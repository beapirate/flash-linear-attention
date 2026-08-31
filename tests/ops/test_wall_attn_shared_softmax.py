# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import pytest
import torch

from fla.ops.wall_attn import combine_wall_attn_outputs, parallel_wall_attn
from fla.utils import assert_close, device

os.environ['TRITON_F32_DEFAULT'] = 'ieee'

RTOL = 5e-3


def _reference_wall(q, k, v, g, scale, cu_seqlens=None):
    B, T, H, _ = q.shape
    if cu_seqlens is None:
        prefix = g.float().cumsum(1)
        same_segment = torch.ones(1, T, T, dtype=torch.bool, device=q.device)
    else:
        prefix = torch.cat(
            [
                g[:, int(cu_seqlens[n]):int(cu_seqlens[n + 1])].float().cumsum(1)
                for n in range(cu_seqlens.numel() - 1)
            ],
            dim=1,
        )
        segment = torch.zeros(T, dtype=torch.long, device=q.device)
        for n in range(cu_seqlens.numel() - 1):
            segment[int(cu_seqlens[n]):int(cu_seqlens[n + 1])] = n
        same_segment = segment.view(1, T, 1) == segment.view(1, 1, T)

    decay = torch.exp(prefix.unsqueeze(2) - prefix.unsqueeze(1))
    scores = (q.float().unsqueeze(2) * k.float().unsqueeze(1) * decay).sum(-1)
    scores = scores.permute(0, 3, 1, 2) * scale
    position = torch.arange(T, device=q.device)
    causal = position.view(1, T, 1) >= position.view(1, 1, T)
    valid = (causal & same_segment).unsqueeze(1).expand(B, H, T, T)
    scores = scores.masked_fill(~valid, float('-inf'))
    probabilities = scores.softmax(-1)
    output = torch.matmul(probabilities, v.float().permute(0, 2, 1, 3)).transpose(1, 2)
    lse = scores.logsumexp(-1).transpose(1, 2)
    return output, lse


def _inputs(T, H, K, V):
    q = torch.randn(1, T, H, K, device=device, requires_grad=True)
    k = torch.randn(1, T, H, K, device=device, requires_grad=True)
    v = torch.randn(1, T, H, V, device=device, requires_grad=True)
    g = (-torch.rand(1, T, H, K, device=device) * 0.05).requires_grad_(True)
    return q, k, v, g


@pytest.mark.parametrize('varlen', [False, True])
def test_parallel_lse_forward_and_backward(varlen):
    torch.manual_seed(42)
    T, H, K, V = 17, 2, 16, 12
    scale = K**-0.5
    cu_seqlens = torch.tensor([0, 7, T], dtype=torch.long, device=device) if varlen else None
    actual_inputs = _inputs(T, H, K, V)
    reference_inputs = tuple(x.detach().clone().requires_grad_(True) for x in actual_inputs)

    output, lse = parallel_wall_attn(*actual_inputs, scale=scale, cu_seqlens=cu_seqlens, return_lse=True)
    output_ref, lse_ref = _reference_wall(*reference_inputs, scale=scale, cu_seqlens=cu_seqlens)
    assert_close('output', output_ref, output, RTOL)
    assert_close('lse', lse_ref, lse, RTOL)

    doutput = torch.randn_like(output)
    dlse = torch.randn_like(lse)
    loss = (output * doutput).sum() + (lse * dlse).sum()
    loss_ref = (output_ref * doutput).sum() + (lse_ref * dlse).sum()
    grads = torch.autograd.grad(loss, actual_inputs)
    grads_ref = torch.autograd.grad(loss_ref, reference_inputs)
    for name, reference, actual in zip(('dq', 'dk', 'dv', 'dg'), grads_ref, grads):
        assert_close(name, reference, actual, RTOL)


def test_shared_softmax_matches_concatenated_reference():
    torch.manual_seed(43)
    T, H, K, V = 13, 2, 16, 12
    scale = K**-0.5
    actual_a = _inputs(T, H, K, V)
    actual_b = _inputs(T, H, K, V)
    reference_a = tuple(x.detach().clone().requires_grad_(True) for x in actual_a)
    reference_b = tuple(x.detach().clone().requires_grad_(True) for x in actual_b)

    output_a, lse_a = parallel_wall_attn(*actual_a, scale=scale, return_lse=True)
    output_b, lse_b = parallel_wall_attn(*actual_b, scale=scale, return_lse=True)
    contribution_a, contribution_b, joint_lse = combine_wall_attn_outputs(output_a, lse_a, output_b, lse_b)

    output_a_ref, lse_a_ref = _reference_wall(*reference_a, scale=scale)
    output_b_ref, lse_b_ref = _reference_wall(*reference_b, scale=scale)
    joint_lse_ref = torch.logaddexp(lse_a_ref, lse_b_ref)
    mass_a_ref = torch.exp(lse_a_ref - joint_lse_ref).unsqueeze(-1)
    mass_b_ref = torch.exp(lse_b_ref - joint_lse_ref).unsqueeze(-1)
    contribution_a_ref = output_a_ref * mass_a_ref
    contribution_b_ref = output_b_ref * mass_b_ref

    assert_close('contribution_a', contribution_a_ref, contribution_a, RTOL)
    assert_close('contribution_b', contribution_b_ref, contribution_b, RTOL)
    assert_close('joint_lse', joint_lse_ref, joint_lse, RTOL)

    gradient = torch.randn_like(torch.cat((contribution_a, contribution_b), dim=-1))
    loss = (torch.cat((contribution_a, contribution_b), dim=-1) * gradient).sum()
    loss_ref = (torch.cat((contribution_a_ref, contribution_b_ref), dim=-1) * gradient).sum()
    grads = torch.autograd.grad(loss, actual_a + actual_b)
    grads_ref = torch.autograd.grad(loss_ref, reference_a + reference_b)
    for name, reference, actual in zip(('dq_a', 'dk_a', 'dv_a', 'dg_a', 'dq_b', 'dk_b', 'dv_b', 'dg_b'), grads_ref, grads):
        assert_close(name, reference, actual, RTOL)
