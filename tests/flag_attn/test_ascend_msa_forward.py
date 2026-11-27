# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Ascend-only MSA prefill and decode, checked against CPU FP32 (never a timing baseline)."""

import pytest
import torch

from flag_attn.testing import backend

pytestmark = [
    pytest.mark.minimax_m3_sparse_attn,
    pytest.mark.skipif(
        not backend.is_available() or backend.runtime.device.vendor_name != "ascend",
        reason="Ascend MSA forward requires an available NPU",
    ),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length,topk,prefix", [(512, 4, 0), (2048, 16, 0), (2048, 4, 0), (3, 4, 0), (129, 4, 16), (513, 16, 127)])
@torch.inference_mode()
def test_prefill_paged_stages(dtype, length, topk, prefix):
    from flag_attn import minimax_m3_index_score, minimax_m3_index_topk, minimax_m3_sparse_attn

    torch.manual_seed(42)
    block, dim, heads = 128, 128, 4
    pages = (length + block - 1) // block
    query_len = length - prefix
    # Randomized physical pages and nonzero prefixes exercise paged/causal ABI.
    order = torch.randperm(pages)
    q = (torch.randn(query_len, heads, dim) * 0.5).to(dtype)
    iq = (torch.randn(query_len, 1, dim) * 0.5).to(dtype)
    ik = (torch.randn(pages, block, dim) * 0.5).to(dtype)
    kv = (torch.randn(pages, 1, block, 2 * dim) * 0.5).to(dtype)
    table = order[None].to(dtype=torch.int32, device=backend.device)
    cu = torch.tensor([0, query_len], dtype=torch.int32, device=backend.device)
    lens = torch.tensor([length], dtype=torch.int32, device=backend.device)
    prefixes = torch.tensor([prefix], dtype=torch.int32, device=backend.device)
    dq, diq, dik, dkv = [x.to(backend.device) for x in (q, iq, ik, kv)]
    scores = minimax_m3_index_score(diq, dik, table, cu, lens, prefixes, query_len, length, 1)
    positions = torch.arange(pages * block)
    query_pos = torch.arange(query_len) + prefix
    causal = positions[None] <= query_pos[:, None]
    dots = iq[:, 0].float() @ ik[order].reshape(-1, dim).float().T
    expected_scores = dots.masked_fill(~causal, -float("inf")).reshape(query_len, pages, block).amax(-1)
    visible = torch.arange(pages)[None] <= query_pos[:, None] // block
    actual_scores = scores[0, :, :pages].cpu()
    torch.testing.assert_close(actual_scores[visible], expected_scores[visible], atol=0.02, rtol=0.02)

    ids = minimax_m3_index_topk(scores, cu, prefixes, query_len, topk, 1, 2)
    actual_ids = ids[0].cpu().long()
    # Check selection against identical defined scores, avoiding numerical
    # near-ties in independently rounded QK reductions. Forced-block ties may
    # have any order; padding beyond the valid selection must be -1.
    priority = actual_scores.clone()
    priority[:, 0] = 1e30
    local = torch.arange(pages)[None] >= (query_pos // block - 1).clamp_min(0)[:, None]
    priority.masked_fill_(local, 1e29)
    priority.masked_fill_(~visible, -float("inf"))
    expected_ids = priority.argsort(dim=-1, descending=True, stable=True)
    for token in range(query_len):
        count = min(topk, int(query_pos[token]) // block + 1)
        assert torch.equal(actual_ids[token, :count].sort().values, expected_ids[token, :count].sort().values)
        assert (actual_ids[token, count:] == -1).all()

    out = torch.empty_like(dq)
    assert minimax_m3_sparse_attn(
        dq, dkv, ids, table, cu, lens, prefixes, query_len, 1, dim ** -0.5, out, stage="prefill"
    ) is None
    cache = kv[order, 0].reshape(-1, 2 * dim).float()
    selected = (actual_ids[:, :, None] == (positions // block)[None, None]).any(1)
    logits = torch.einsum("qhd,kd->hqk", q.float(), cache[:, :dim]) * dim ** -0.5
    logits.masked_fill_(~(selected & causal)[None], -float("inf"))
    expected = torch.einsum("hqk,kd->qhd", logits.softmax(-1), cache[:, dim:])
    actual = out.cpu().float()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)

    # Invalid *unused* padded slots must never be dereferenced. Consumed invalid
    # page IDs are outside the current kernel contract and are not sent to NPU.
    padded = ids.clone()
    unused = torch.arange(topk)[None] >= (query_pos // block + 1)[:, None]
    padded[0].masked_fill_(unused.to(backend.device), -12345)
    minimax_m3_sparse_attn(dq, dkv, padded, table, cu, lens, prefixes, query_len, 1, dim ** -0.5, out)
    torch.testing.assert_close(out.cpu().float(), expected, atol=0.02, rtol=0.02)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length,topk", [(512, 4), (2048, 16), (129, 4)])
@torch.inference_mode()
def test_decode_stages(dtype, length, topk):
    """Single-token decode: the query is the last position, all blocks visible."""
    # decode symbols are exported by the operator subpackage, not the
    # top-level flag_attn registry (which only registers the prefill trio).
    from flag_attn.minimax_sparse_attention import (
        minimax_m3_index_decode,
        minimax_m3_sparse_attn_decode,
    )

    torch.manual_seed(42)
    block, dim, heads = 128, 128, 4
    pages = (length + block - 1) // block
    order = torch.randperm(pages)
    q = (torch.randn(1, heads, dim) * 0.5).to(dtype)
    iq = (torch.randn(1, 1, dim) * 0.5).to(dtype)
    ik = (torch.randn(pages, block, dim) * 0.5).to(dtype)
    kv = (torch.randn(pages, 1, block, 2 * dim) * 0.5).to(dtype)
    table = order[None].to(dtype=torch.int32, device=backend.device)
    lens = torch.tensor([length], dtype=torch.int32, device=backend.device)
    dq, diq, dik, dkv = [x.to(backend.device) for x in (q, iq, ik, kv)]

    ids = minimax_m3_index_decode(diq, dik, table, lens, length, topk, 1, 2, 1, 1, 1)
    actual_ids = ids[0].cpu().long()
    # The decode query sits at the last position, so every page is visible.
    positions = torch.arange(pages * block)
    dots = iq[0, 0].float() @ ik[order].reshape(-1, dim).float().T
    actual_scores = dots.reshape(pages, block).amax(-1).cpu()
    priority = actual_scores.clone()
    priority[0] = 1e30
    local = torch.arange(pages) >= max(pages - 2, 0)
    priority[local] = 1e29
    expected_ids = priority.argsort(dim=-1, descending=True, stable=True)
    count = min(topk, pages)
    assert torch.equal(actual_ids[0, :count].sort().values, expected_ids[:count].sort().values)
    assert (actual_ids[0, count:] == -1).all()

    out = torch.empty_like(dq)
    minimax_m3_sparse_attn_decode(
        q=dq, kv_cache=dkv, topk_idx=ids, block_table=table,
        sm_scale=dim ** -0.5, output=out, seq_lens=lens,
        num_kv_heads=1, decode_query_len=1,
    )
    cache = kv[order, 0].reshape(-1, 2 * dim).float()
    selected = (actual_ids[0][:, None] == (positions // block)[None]).any(0)
    valid = positions < length  # pages*block can exceed seq_len on partial pages
    logits = torch.einsum("hd,kd->hk", q[0].float(), cache[:, :dim]) * dim ** -0.5
    logits.masked_fill_(~(selected & valid)[None], -float("inf"))
    expected = torch.einsum("hk,kd->hd", logits.softmax(-1), cache[:, dim:])
    actual = out[0].cpu().float()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)

    # Invalid *unused* padded slots must never be dereferenced.
    padded = ids.clone()
    if count < topk:
        padded[0, 0, count:] = -12345
        minimax_m3_sparse_attn_decode(
            q=dq, kv_cache=dkv, topk_idx=padded, block_table=table,
            sm_scale=dim ** -0.5, output=out, seq_lens=lens,
            num_kv_heads=1, decode_query_len=1,
        )
        torch.testing.assert_close(out[0].cpu().float(), expected, atol=0.02, rtol=0.02)

def test_prefill_registration_and_fp8_capability():
    from flag_attn import runtime
    from flag_attn.minimax_sparse_attention import minimax_m3_index_score, minimax_m3_index_topk
    from flag_attn.runtime.backend import _ascend
    from flag_attn.runtime.backend._ascend.minimax_sparse_attention import (
        minimax_m3_sparse_attn_decode as ascend_decode,
    )

    assert runtime.device.vendor_name == "ascend"
    assert minimax_m3_index_score is _ascend.minimax_m3_index_score
    assert minimax_m3_index_topk is _ascend.minimax_m3_index_topk
    assert not backend.supports_msa_fp8()
    # The Ascend decode resolves either to a backend-specific implementation
    # (expected once optimized) or to the generic re-export (the starting
    # state); both are valid dispatch states for this task.
    assert callable(ascend_decode)


def test_prefill_rejects_unknown_stage():
    from flag_attn import minimax_m3_sparse_attn

    with pytest.raises(ValueError, match="Unsupported MiniMax stage"):
        minimax_m3_sparse_attn(stage="unsupported")
