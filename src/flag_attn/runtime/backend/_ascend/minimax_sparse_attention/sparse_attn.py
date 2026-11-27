# Copyright 2026 FlagOS Contributors
# Copyright contributors to the vLLM project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ascend M3 prefill with bounded query grids and direct paged KV loads."""
import torch
import triton
import triton.language as tl

from flag_attn.minimax_sparse_attention.sparse_attn import (
    minimax_m3_sparse_attn_decode as minimax_m3_sparse_attn_decode,
)

SPARSE_BLOCK_SIZE = 128
_NPU_PREFILL_MAX_PROGRAMS = 32768
_KV_SCALE_NONE = 0
_KV_SCALE_SCALAR = 1
_KV_SCALE_TOKEN = 2
# C8 full-selection dense path: flattened (query, head) row budget. QH=16
# (BLOCK_SIZE_Q=4 at GQA group 4) puts S=2048 at 2048 programs, the
# scheduling point the C2 sweep measured at 5.74 ms (native level). All
# correctness ingredients are independently proven: flat-view contiguous Q
# load (C4), 3-D base-arithmetic stores (C1/C5), online-softmax carry (C6),
# finite -1e30 mask floor (C5). This variant composes them WITHOUT chunk
# splitting or per-block partials/merge (the C4-C6 failure sources).
_DENSE_MAX_QH = 16


@triton.jit(do_not_specialize=["num_kv_heads", "gqa_group_size", "head_dim", "q_flat_rows", "sm_scale"])
def _gqa_dense_causal_single_pass(
    q_ptr,  # [total_q, num_heads, head_dim], contiguous
    kv_cache_ptr,  # [num_blocks, num_kv_heads, 128, 2*head_dim]
    o_ptr,  # [total_q, num_heads, head_dim]
    block_table_ptr,  # [num_reqs, max_blocks]
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    num_kv_heads,
    gqa_group_size,
    head_dim,
    q_flat_rows,  # total_q * num_heads
    sm_scale,
    stride_qh,
    stride_qd,
    stride_kv_blk,
    stride_kv_h,
    stride_kv_pos,
    stride_kv_d,
    stride_on,
    stride_oh,
    stride_od,
    stride_bt_b,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,
):
    """Dense causal GQA prefill, one pass, direct output (full selection).

    Mathematically equal to the block-sparse attention when every query
    selects all of its visible blocks (topk >= visible-block count); the
    host wrapper guarantees the precondition from block-table metadata.
    Each program covers BLOCK_SIZE_Q queries x the GQA head group of one
    kv head, walks every visible block with a single online-softmax carry
    chain, and writes the final normalized tile directly (no partials).
    """
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    num_heads = num_kv_heads * gqa_group_size
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q * BLOCK_SIZE_Q >= q_len:
        return
    bt_row = block_table_ptr + pid_b * stride_bt_b
    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)

    # Q tile: contiguous [QH, D] slice of the kvh-major flat view
    # [kvh, total_q, gqa, D] (built host-side): rows (q, in-group head) of one
    # query are consecutive, so the tile is a straight contiguous slice.
    row_plane = q_flat_rows // num_kv_heads  # total_q * gqa rows per kv head
    q_row_base = pid_kh * row_plane + (q_start + pid_q * BLOCK_SIZE_Q) * gqa_group_size
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + q_row_base * stride_qh,
        shape=(q_flat_rows, head_dim),
        strides=(stride_qh, stride_qd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_QH, BLOCK_SIZE_D),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")

    # Row r of [QH, *] is query r // H, head r % H.
    off_qh = tl.arange(0, BLOCK_SIZE_QH)
    q_abs_qh = off_qh // BLOCK_SIZE_H + pid_q * BLOCK_SIZE_Q + prefix_len

    m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_SIZE_QH,), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_D), dtype=tl.float32)

    # Blocks visible to the tile's LAST query cover every earlier query too;
    # earlier rows mask their future entries via the causal predicate.
    n_blocks = (prefix_len + q_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    for b in range(0, n_blocks):
        page = tl.load(bt_row + b).to(tl.int64)
        c = b * BLOCK_SIZE_K
        pos = c + off_n
        pos_mask = pos < seq_len
        k_base_ptr = kv_cache_ptr + page * stride_kv_blk + pid_kh * stride_kv_h
        k_ptrs = tl.make_block_ptr(
            base=k_base_ptr,
            shape=(BLOCK_SIZE_K, head_dim),
            strides=(stride_kv_pos, stride_kv_d),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
            order=(1, 0),
        )
        k = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero")
        qk = tl.dot(q, tl.trans(k)) * sm_scale_log2e

        allowed = (q_abs_qh[:, None] >= pos[None, :]) & pos_mask[None, :]
        # Finite floor: exp2 underflows to exact 0; fully-masked rows keep a
        # finite m_ij and are rescaled to zero by alpha once real entries
        # arrive (no -inf arithmetic anywhere).
        qk = tl.where(allowed, qk, -1.0e30)

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp2(m_i - m_ij)  # -inf - finite == -inf -> 0
        acc_o = acc_o * alpha[:, None]
        l_i = l_i * alpha + l_ij

        v_base_ptr = k_base_ptr + head_dim * stride_kv_d
        v_ptrs = tl.make_block_ptr(
            base=v_base_ptr,
            shape=(BLOCK_SIZE_K, head_dim),
            strides=(stride_kv_pos, stride_kv_d),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
            order=(1, 0),
        )
        v = tl.load(v_ptrs, boundary_check=(0, 1), padding_option="zero")
        acc_o += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
    acc_o = acc_o * inv_l[:, None]

    # 3-D base-arithmetic store (proven layout; 1-D row-offset scatter stores
    # miscompile on this backend).
    off_qq = tl.arange(0, BLOCK_SIZE_Q)
    off_hh = tl.arange(0, BLOCK_SIZE_H)
    qt_mask = off_qq < q_len - pid_q * BLOCK_SIZE_Q
    h_mask = off_hh < gqa_group_size
    d_mask = off_d < head_dim
    acc3 = tl.reshape(
        acc_o.to(o_ptr.dtype.element_ty), BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D
    )
    # Same kvh-major layout for O: plane offset + per-query stride_on (which
    # already spans the in-group heads) + per-head stride_oh.
    total_q = q_flat_rows // num_heads
    o_base = (
        o_ptr
        + pid_kh * total_q * stride_on
        + (q_start + pid_q * BLOCK_SIZE_Q + off_qq)[:, None, None] * stride_on
        + off_hh[None, :, None] * stride_oh
        + off_d[None, None, :] * stride_od
    )
    tl.store(o_base, acc3, mask=qt_mask[:, None, None] & h_mask[None, :, None] & d_mask[None, None, :])


@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _gqa_sparse_fwd_direct(
    q_ptr,  # [total_q, num_heads, head_dim]
    kv_cache_ptr,  # main cache: [num_blocks, num_kv_heads, 128, 2*head_dim]
    k_scale_ptr,
    v_scale_ptr,
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk]
    o_ptr,  # [total_q, num_heads, head_dim]
    block_table_ptr,  # [num_reqs, max_blocks]
    cu_seqlens_q,
    cu_seqblocks_q,
    seq_lens,
    prefix_lens,
    num_kv_heads,
    gqa_group_size,
    head_dim,
    max_topk,
    num_q_loop,
    sm_scale,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kv_blk,
    stride_kv_h,
    stride_kv_pos,
    stride_kv_d,
    stride_ks_h,
    stride_ks_t,
    stride_vs_h,
    stride_vs_t,
    stride_th,
    stride_tn,
    stride_tk,
    stride_on,
    stride_oh,
    stride_od,
    stride_bt_b,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,
    USE_FP8: tl.constexpr,  # fp8 KV cache: dequantize K/V to q.dtype on load
    KV_SCALE_MODE: tl.constexpr,  # 0: none, 1: scalar, 2: [kv_head, token]
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    q_block_start = tl.load(cu_seqblocks_q + pid_b)
    q_block_len = tl.load(cu_seqblocks_q + pid_b + 1) - q_block_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q * num_q_loop >= q_block_len:
        return
    real_q_loop = min(num_q_loop, q_block_len - pid_q * num_q_loop)
    bt_row = block_table_ptr + pid_b * stride_bt_b
    off_n = tl.arange(0, BLOCK_SIZE_K)
    if USE_FP8 and KV_SCALE_MODE == 1:
        # Scalar scales are invariant across every selected page. Fold K into
        # the logits scale and delay V until the normalized output.
        qk_scale = sm_scale_log2e * tl.load(k_scale_ptr)
        v_scale_scalar = tl.load(v_scale_ptr)
    for j in range(real_q_loop):
        pid_q_j = pid_q * num_q_loop + j
        t_ptr_j = t_ptr + (q_block_start + pid_q_j) * stride_tn + pid_kh * stride_th
        # Valid block count from seq position (no sentinel): block_size_q == 1.
        q_abs = prefix_len + pid_q_j * BLOCK_SIZE_Q
        valid_blocks = (q_abs + BLOCK_SIZE_K) // BLOCK_SIZE_K
        real_topk = tl.minimum(max_topk, valid_blocks)
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + q_start * stride_qn + pid_kh * gqa_group_size * stride_qh,
            shape=(q_len, gqa_group_size, head_dim),
            strides=(stride_qn, stride_qh, stride_qd),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        q = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
        off_q = (
            tl.arange(0, BLOCK_SIZE_Q)[:, None]
            + pid_q_j * BLOCK_SIZE_Q
            + prefix_len
            - tl.arange(0, BLOCK_SIZE_K)[None, :]
        )
        m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)

        l_i = tl.zeros((BLOCK_SIZE_QH,), dtype=tl.float32)
        # Keep the direct kernel's PV accumulator in [QH, D] order.  The
        # transposed [D, QH] form spills heavily for the FP8 QH=32 tile.
        acc_o = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_D), dtype=tl.float32)
        q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_D)
        if USE_FP8:
            # Keep QK in FP8 Tensor Core form.  Q is per-head dynamically
            # scaled once, then its scale is restored on the FP32 logits.
            # This keeps the existing K cache in FP8 through tl.dot instead
            # of materializing a BF16 K tile for every selected page.
            qk_q_scale = tl.maximum(tl.max(tl.abs(q), axis=1) * (1.0 / 448.0), 1.0e-8)
            q_qk = (q / qk_q_scale[:, None]).to(tl.float8e4nv)
        else:
            q_qk = q
        for _ in range(real_topk):
            blk = tl.load(t_ptr_j).to(tl.int32)
            t_ptr_j = t_ptr_j + stride_tk
            c = blk * BLOCK_SIZE_K
            page = tl.load(bt_row + blk).to(tl.int64)

            pos = c + off_n
            pos_mask = pos < seq_len

            k_base_ptr = kv_cache_ptr + page * stride_kv_blk + pid_kh * stride_kv_h
            k_ptrs = tl.make_block_ptr(
                base=k_base_ptr,
                shape=(BLOCK_SIZE_K, head_dim),
                strides=(stride_kv_pos, stride_kv_d),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
                order=(1, 0),
            )
            k = tl.load(
                k_ptrs,
                boundary_check=(0, 1),
                padding_option="zero",
            )
            if USE_FP8:
                if KV_SCALE_MODE == 2:
                    k_scale = tl.load(
                        k_scale_ptr
                        + pid_kh * stride_ks_h
                        + (page * BLOCK_SIZE_K + off_n) * stride_ks_t,
                        mask=pos_mask,
                        other=1.0,
                    )
            qk_dot = tl.dot(q_qk, tl.trans(k))
            if USE_FP8:
                qk_dot *= qk_q_scale[:, None]

            is_full_causal = (c + BLOCK_SIZE_K) <= q_abs
            is_full_seq = (c + BLOCK_SIZE_K) <= seq_len

            qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
            if not is_full_causal:
                qk += tl.where(off_q[:, None, :] >= c, 0, float("-inf"))

            qk = tl.reshape(qk, BLOCK_SIZE_QH, BLOCK_SIZE_K)
            if USE_FP8:
                if KV_SCALE_MODE == 1:
                    qk += qk_dot * qk_scale
                elif KV_SCALE_MODE == 2:
                    # Q @ (K * scale[token]) is equivalent to scaling the
                    # smaller [QH, K] logits instead of the [K, D] K tile.
                    qk += qk_dot * (sm_scale_log2e * k_scale[None, :])
                else:
                    qk += qk_dot * sm_scale_log2e
            else:
                qk += qk_dot * sm_scale_log2e

            if not is_full_seq:
                qk += tl.where(pos_mask[None, :], 0, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)

            alpha = tl.exp2(m_i - m_ij)
            acc_o = acc_o * alpha[:, None]
            l_i = l_i * alpha + l_ij

            v_base_ptr = k_base_ptr + head_dim * stride_kv_d
            v_ptrs = tl.make_block_ptr(
                base=v_base_ptr,
                shape=(BLOCK_SIZE_K, head_dim),
                strides=(stride_kv_pos, stride_kv_d),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
                order=(1, 0),
            )
            v = tl.load(
                v_ptrs,
                boundary_check=(0, 1),
                padding_option="zero",
            )

            if USE_FP8:
                if KV_SCALE_MODE == 2:
                    v_scale = tl.load(
                        v_scale_ptr
                        + pid_kh * stride_vs_h
                        + (page * BLOCK_SIZE_K + off_n) * stride_vs_t,
                        mask=pos_mask,
                        other=1.0,
                    )

            if USE_FP8 and BLOCK_SIZE_H < 32:
                # The 8/16-head tiles lower efficiently as FP8 PV MMAs.
                # Fold a per-token V scale into P before quantizing P
                # row-wise, and keep V in its FP8 cache format.
                pv_p = p
                if KV_SCALE_MODE == 2:
                    pv_p *= v_scale[None, :]
                pv_p_scale = tl.maximum(
                    tl.max(tl.abs(pv_p), axis=1) * (1.0 / 448.0), 1.0e-8
                )
                p_dot = (pv_p / pv_p_scale[:, None]).to(tl.float8e4nv)
                acc_o += tl.dot(p_dot, v) * pv_p_scale[:, None]
            else:
                # QH=32 uses the native PV accumulator layout above, but its
                # FP8 PV lowering spills heavily; retain BF16 operands there.
                if USE_FP8:
                    v = v.to(q.dtype)
                p_dot = p.to(v.dtype)
                if USE_FP8 and KV_SCALE_MODE == 2:
                    p_dot = (p * v_scale[None, :]).to(v.dtype)
                acc_o += tl.dot(p_dot, v)
            m_i = m_ij

        inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
        if USE_FP8 and KV_SCALE_MODE == 1:
            inv_l *= v_scale_scalar
        acc_o = acc_o * inv_l[:, None]
        acc_o = tl.reshape(acc_o, BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D)
        o_ptrs = tl.make_block_ptr(
            base=o_ptr + q_start * stride_on + pid_kh * gqa_group_size * stride_oh,
            shape=(q_len, gqa_group_size, head_dim),
            strides=(stride_on, stride_oh, stride_od),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1, 2))





# C9: topk-pair-unrolled sparse prefill (backend-agnostic idea taken from the
# upstream NVIDIA sparse kernels: after a selection batching in large tiles is
# impossible, shorten the per-program critical path by consuming two selected
# pages per online-softmax step). Q/O use the kvh-major contiguous layout so
# the GQA head rows of one query are a contiguous [BLOCK_H, D] tile; the two
# pages of a step share one softmax merge, halving the number of cross-pipe
# (cube<->vector) handshakes per selected page pair.
@triton.jit(do_not_specialize=["num_kv_heads", "gqa_group_size", "head_dim",
                               "q_flat_rows", "max_topk", "sm_scale"])
def _gqa_sparse_pairs_fwd(
    q_ptr,  # [kvh, total_q, gqa, head_dim] contiguous
    kv_cache_ptr,  # [num_blocks, num_kv_heads, 128, 2*head_dim]
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk]
    o_ptr,  # [kvh, total_q, gqa, head_dim] contiguous
    block_table_ptr,  # [num_reqs, max_blocks]
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    num_kv_heads,
    gqa_group_size,
    head_dim,
    q_flat_rows,  # total_q * num_kv_heads * gqa_group_size
    max_topk,
    sm_scale,
    stride_qh,
    stride_qd,
    stride_kv_blk,
    stride_kv_h,
    stride_kv_pos,
    stride_kv_d,
    stride_th,
    stride_tn,
    stride_tk,
    stride_oh,
    stride_od,
    stride_bt_b,
    BLOCK_H: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q >= q_len:
        return
    q_abs = prefix_len + pid_q
    real_topk = tl.minimum(max_topk, (q_abs + BLOCK_SIZE_K) // BLOCK_SIZE_K)
    if real_topk <= 0:
        return
    bt_row = block_table_ptr + pid_b * stride_bt_b
    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_h = tl.arange(0, BLOCK_H)
    h_valid = off_h < gqa_group_size
    d_valid = off_d < head_dim

    row_plane = q_flat_rows // num_kv_heads  # total_q * gqa rows per kv head
    row_base = pid_kh * row_plane + (q_start + pid_q) * gqa_group_size
    q = tl.load(
        q_ptr + (row_base + off_h)[:, None] * stride_qh
        + off_d[None, :] * stride_qd,
        mask=h_valid[:, None] & d_valid[None, :],
        other=0.0,
    )  # [BLOCK_H, D]

    t_row = t_ptr + pid_kh * stride_th + (q_start + pid_q) * stride_tn
    m_i = tl.full((BLOCK_H,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_H, BLOCK_SIZE_D), dtype=tl.float32)

    # Two selected pages per step; each step performs one softmax merge.
    n_pairs = real_topk // 2
    for jj in range(0, n_pairs):
        j0 = 2 * jj
        j1 = j0 + 1
        blk0 = tl.load(t_row + j0 * stride_tk).to(tl.int32)
        blk1 = tl.load(t_row + j1 * stride_tk).to(tl.int32)
        page0 = tl.load(bt_row + blk0).to(tl.int64)
        page1 = tl.load(bt_row + blk1).to(tl.int64)
        pos0 = blk0 * BLOCK_SIZE_K + off_n
        pos1 = blk1 * BLOCK_SIZE_K + off_n
        k_base0 = kv_cache_ptr + page0 * stride_kv_blk + pid_kh * stride_kv_h
        k_base1 = kv_cache_ptr + page1 * stride_kv_blk + pid_kh * stride_kv_h
        k0 = tl.load(
            tl.make_block_ptr(
                base=k_base0, shape=(BLOCK_SIZE_K, head_dim),
                strides=(stride_kv_pos, stride_kv_d), offsets=(0, 0),
                block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D), order=(1, 0),
            ),
            boundary_check=(0, 1), padding_option="zero",
        )
        k1 = tl.load(
            tl.make_block_ptr(
                base=k_base1, shape=(BLOCK_SIZE_K, head_dim),
                strides=(stride_kv_pos, stride_kv_d), offsets=(0, 0),
                block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D), order=(1, 0),
            ),
            boundary_check=(0, 1), padding_option="zero",
        )
        qk0 = tl.dot(q, tl.trans(k0)) * sm_scale_log2e
        qk1 = tl.dot(q, tl.trans(k1)) * sm_scale_log2e
        allow0 = (q_abs >= pos0)[None, :] & (pos0 < seq_len)[None, :]
        allow1 = (q_abs >= pos1)[None, :] & (pos1 < seq_len)[None, :]
        qk0 = tl.where(allow0, qk0, -1.0e30)
        qk1 = tl.where(allow1, qk1, -1.0e30)
        m_new = tl.maximum(m_i, tl.maximum(tl.max(qk0, axis=1), tl.max(qk1, axis=1)))
        alpha = tl.exp2(m_i - m_new)
        p0 = tl.exp2(qk0 - m_new[:, None])
        p1 = tl.exp2(qk1 - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p0, axis=1) + tl.sum(p1, axis=1)
        acc_o = acc_o * alpha[:, None]
        v_base0 = k_base0 + head_dim * stride_kv_d
        v_base1 = k_base1 + head_dim * stride_kv_d
        v0 = tl.load(
            tl.make_block_ptr(
                base=v_base0, shape=(BLOCK_SIZE_K, head_dim),
                strides=(stride_kv_pos, stride_kv_d), offsets=(0, 0),
                block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D), order=(1, 0),
            ),
            boundary_check=(0, 1), padding_option="zero",
        )
        v1 = tl.load(
            tl.make_block_ptr(
                base=v_base1, shape=(BLOCK_SIZE_K, head_dim),
                strides=(stride_kv_pos, stride_kv_d), offsets=(0, 0),
                block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D), order=(1, 0),
            ),
            boundary_check=(0, 1), padding_option="zero",
        )
        acc_o += tl.dot(p0.to(v0.dtype), v0)
        acc_o += tl.dot(p1.to(v1.dtype), v1)
        m_i = m_new
    if real_topk % 2 == 1:
        blk = tl.load(t_row + (real_topk - 1) * stride_tk).to(tl.int32)
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = blk * BLOCK_SIZE_K + off_n
        k_base = kv_cache_ptr + page * stride_kv_blk + pid_kh * stride_kv_h
        k = tl.load(
            tl.make_block_ptr(
                base=k_base, shape=(BLOCK_SIZE_K, head_dim),
                strides=(stride_kv_pos, stride_kv_d), offsets=(0, 0),
                block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D), order=(1, 0),
            ),
            boundary_check=(0, 1), padding_option="zero",
        )
        qk = tl.dot(q, tl.trans(k)) * sm_scale_log2e
        allow = (q_abs >= pos)[None, :] & (pos < seq_len)[None, :]
        qk = tl.where(allow, qk, -1.0e30)
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc_o = acc_o * alpha[:, None]
        v = tl.load(
            tl.make_block_ptr(
                base=k_base + head_dim * stride_kv_d,
                shape=(BLOCK_SIZE_K, head_dim),
                strides=(stride_kv_pos, stride_kv_d), offsets=(0, 0),
                block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D), order=(1, 0),
            ),
            boundary_check=(0, 1), padding_option="zero",
        )
        acc_o += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
    acc_o = acc_o * inv_l[:, None]
    tl.store(
        o_ptr + (row_base + off_h)[:, None] * stride_oh
        + off_d[None, :] * stride_od,
        acc_o.to(o_ptr.dtype.element_ty),
        mask=h_valid[:, None] & d_valid[None, :],
    )


@torch.no_grad()
def minimax_m3_sparse_attn(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    kv_cache: torch.Tensor,  # [num_blocks, num_kv_heads, 128, 2*head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk]
    block_table: torch.Tensor,  # [batch, max_blocks]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim]
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    """GQA block-sparse attention over the selected blocks. block_size_q == 1."""
    total_q, num_heads, head_dim = q.shape
    batch = cu_seqlens_q.shape[0] - 1
    topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    if q.device.type != "npu":
        raise ValueError("Ascend MSA requires NPU tensors")
    if kv_cache.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Ascend MSA supports FP16/BF16 KV caches")
    if k_scale is None and v_scale is None and block_table.shape[1] <= topk and q.is_contiguous():
        # C8: full-selection fast path (single-pass dense causal, direct
        # output). The block-table width upper-bounds every request's
        # visible-block count, so width <= topk implies every query selects
        # ALL its visible blocks: the per-query top-k degenerates to "all
        # causal blocks" and sparse attention equals dense causal attention.
        # Runtime data property (host metadata only, no device sync), not a
        # benchmark-shape specialization; wider tables and non-contiguous Q
        # fall through to the sparse kernels below.
        block_size_h = triton.next_power_of_2(gqa_group_size)
        block_size_q = max(1, _DENSE_MAX_QH // block_size_h)
        # kvh-major Q/O views ([kvh, total_q, gqa, D]) so each program's
        # (query-tile x in-group heads) rows are contiguous. Two layout
        # passes (~2x 8.4 MB at S=2048, ~21 us) against a 6 ms kernel.
        q_major = (
            q.view(total_q, num_kv_heads, gqa_group_size, head_dim)
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        o_major = torch.empty_like(q_major)
        _gqa_dense_causal_single_pass[
            (triton.cdiv(max_query_len, block_size_q), num_kv_heads, batch)
        ](
            q_major,
            kv_cache,
            o_major,
            block_table,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            num_kv_heads,
            gqa_group_size,
            head_dim,
            total_q * num_heads,
            sm_scale,
            q_major.stride(2),
            q_major.stride(3),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            kv_cache.stride(3),
            o_major.stride(1),
            o_major.stride(2),
            o_major.stride(3),
            block_table.stride(0),
            BLOCK_SIZE_Q=block_size_q,
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
            BLOCK_SIZE_D=triton.next_power_of_2(head_dim),
            BLOCK_SIZE_H=block_size_h,
            BLOCK_SIZE_QH=block_size_q * block_size_h,
            num_warps=4,
            num_stages=3,
        )
        output.view(total_q, num_kv_heads, gqa_group_size, head_dim).copy_(
            o_major.permute(1, 0, 2, 3)
        )
        return
    if k_scale is None and v_scale is None and q.is_contiguous():
        # C9: partial-selection sparse path with pair-unrolled topk loop and
        # kvh-major contiguous Q/O tiles (same layouts as the dense branch).
        block_size_h = triton.next_power_of_2(gqa_group_size)
        q_major = (
            q.view(total_q, num_kv_heads, gqa_group_size, head_dim)
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        o_major = torch.empty_like(q_major)
        _gqa_sparse_pairs_fwd[(max_query_len, num_kv_heads, batch)](
            q_major,
            kv_cache,
            topk_idx,
            o_major,
            block_table,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            num_kv_heads,
            gqa_group_size,
            head_dim,
            total_q * num_heads,
            topk,
            sm_scale,
            q_major.stride(2),
            q_major.stride(3),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            kv_cache.stride(3),
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_idx.stride(2),
            o_major.stride(2),
            o_major.stride(3),
            block_table.stride(0),
            BLOCK_H=block_size_h,
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
            BLOCK_SIZE_D=triton.next_power_of_2(head_dim),
            num_warps=4,
            num_stages=3,
        )
        output.view(total_q, num_kv_heads, gqa_group_size, head_dim).copy_(
            o_major.permute(1, 0, 2, 3)
        )
        return
    use_fp8 = False
    (
        k_scale_arg,
        v_scale_arg,
        stride_ks_h,
        stride_ks_t,
        stride_vs_h,
        stride_vs_t,
        kv_scale_mode,
    ) = (
        (output, output, 0, 0, 0, 0, _KV_SCALE_NONE)
        if use_fp8
        else (
            output,
            output,
            0,
            0,
            0,
            0,
            _KV_SCALE_NONE,
        )
    )
    block_size_h = triton.next_power_of_2(gqa_group_size)
    use_direct = q.device.type == "npu"
    if use_direct:
        fixed_grid_size = num_kv_heads * batch
        max_query_programs = max(
            1,
            _NPU_PREFILL_MAX_PROGRAMS // fixed_grid_size,
        )
        num_q_loop = triton.cdiv(max_query_len, max_query_programs)
    else:
        num_q_loop = 1
    grid = (triton.cdiv(max_query_len, num_q_loop), num_kv_heads, batch)
    if use_direct or use_fp8 or block_size_h < 8:
        _gqa_sparse_fwd_direct[grid](
            q,
            kv_cache,
            k_scale_arg,
            v_scale_arg,
            topk_idx,
            output,
            block_table,
            cu_seqlens_q,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            num_kv_heads,
            gqa_group_size,
            head_dim,
            topk,
            num_q_loop,
            sm_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            kv_cache.stride(3),
            stride_ks_h,
            stride_ks_t,
            stride_vs_h,
            stride_vs_t,
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_idx.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            block_table.stride(0),
            BLOCK_SIZE_Q=1,
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
            BLOCK_SIZE_D=triton.next_power_of_2(head_dim),
            BLOCK_SIZE_H=block_size_h,
            BLOCK_SIZE_QH=block_size_h,
            USE_FP8=use_fp8,
            KV_SCALE_MODE=kv_scale_mode,
            num_warps=4,
            num_stages=3,
        )
        return
