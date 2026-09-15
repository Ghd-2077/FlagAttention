# Copyright 2026 FlagOS Contributors
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

"""Ascend dense KV-cache decode and sparse decode over already decoded values.

The FP8 byte-cache CPU conversion in the source is a baseline adapter, not an
optimized device kernel, and is intentionally outside this module.
"""
import torch
import triton
import triton.language as tl

_DENSE_DECODE_CONFIGS = [
    triton.Config({"BLOCK_H": 16, "BLOCK_N": 32}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_H": 32, "BLOCK_N": 32}, num_warps=4, num_stages=1),
]
_SPARSE_DECODE_CONFIGS = [
    triton.Config({"BK": 16, "BH": 16}, num_warps=2, num_stages=1),
    triton.Config({"BK": 32, "BH": 16}, num_warps=4, num_stages=1),
]


@triton.autotune(
    configs=_DENSE_DECODE_CONFIGS,
    key=["HQ", "DQK", "HAVE_CAUSAL", "EMULATED_TLE"],
)
@triton.jit
def _dense_decode_kernel(
    Q_ptr,
    stride_q_b,
    stride_q_sq,
    stride_q_h,
    KV_cache,
    stride_kv_bs,
    Block_table,
    stride_bt_b,
    Seq_lens,
    Out,
    stride_o_b,
    stride_o_sq,
    stride_o_h,
    LSE,
    stride_lse_b,
    stride_lse_h,
    sm_scale,
    SQ,
    HQ: tl.constexpr,
    DQK: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HAVE_CAUSAL: tl.constexpr,
    EMULATED_TLE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Dense decode kernel with paged attention and online softmax.
    Grid: (ceil(HQ / BLOCK_H), batch_size * seq_q)
    """
    pid_h_block = tl.program_id(0)
    pid_b_sq = tl.program_id(1)
    i_b = pid_b_sq // SQ
    i_sq = pid_b_sq % SQ

    cur_head = pid_h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_head = cur_head < HQ

    # Load Q: NoPE part [BLOCK_H, HEAD_DIM_V] and RoPE part [BLOCK_H, DQK-HEAD_DIM_V]
    offs_d_nope = tl.arange(0, HEAD_DIM_V)
    offs_q_nope = (
        i_b * stride_q_b
        + i_sq * stride_q_sq
        + cur_head[:, None] * stride_q_h
        + offs_d_nope[None, :]
    )
    q_nope = tl.load(Q_ptr + offs_q_nope, mask=mask_head[:, None], other=0.0)

    offs_d_pe = tl.arange(HEAD_DIM_V, DQK)
    offs_q_pe = (
        i_b * stride_q_b
        + i_sq * stride_q_sq
        + cur_head[:, None] * stride_q_h
        + offs_d_pe[None, :]
    )
    q_pe = tl.load(Q_ptr + offs_q_pe, mask=mask_head[:, None], other=0.0)

    # Online softmax accumulators
    e_max = tl.full([BLOCK_H], value=float("-inf"), dtype=tl.float32)
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, HEAD_DIM_V], dtype=tl.float32)

    cur_batch_seq_len = tl.load(Seq_lens + i_b)
    if HAVE_CAUSAL:
        cur_batch_seq_len = tl.maximum(cur_batch_seq_len - SQ + i_sq + 1, 0)
    Block_table += i_b * stride_bt_b

    offs_n = tl.arange(0, BLOCK_N)

    if EMULATED_TLE:
        # The native TLE kernel overlaps two pipe slots.  Ascend Triton 3.5.1
        # cannot lower the dynamic nested condition that was used here, so the
        # emulation keeps the same pair order with two explicit, masked tile
        # bodies.  Invalid second tiles are all-masked and therefore contribute
        # zero to the online softmax.
        num_tiles = tl.cdiv(cur_batch_seq_len, BLOCK_N)
        num_pairs = tl.cdiv(num_tiles, 2)
        for pair in range(num_pairs):
            tile0 = pair * 2
            tile0_offs_n = offs_n + tile0 * BLOCK_N
            tile0_mask = tile0_offs_n < cur_batch_seq_len
            tile0_page = tl.load(
                Block_table + tile0_offs_n // PAGE_SIZE,
                mask=tile0_mask,
                other=0,
            )
            tile0_loc = tile0_page * PAGE_SIZE + tile0_offs_n % PAGE_SIZE
            tile0_v = tl.load(
                KV_cache + tile0_loc[:, None] * stride_kv_bs + offs_d_nope[None, :],
                mask=tile0_mask[:, None],
                other=0.0,
            )
            tile0_qk = tl.dot(q_nope, tl.trans(tile0_v))
            tile0_k_pe = tl.load(
                KV_cache + tile0_loc[None, :] * stride_kv_bs + offs_d_pe[:, None],
                mask=tile0_mask[None, :],
                other=0.0,
            )
            tile0_qk = tl.dot(q_pe, tile0_k_pe, acc=tile0_qk) * sm_scale
            tile0_qk = tl.where(tile0_mask[None, :], tile0_qk, float("-inf"))
            tile0_max = tl.maximum(tl.max(tile0_qk, 1), e_max)
            tile0_scale = tl.exp(e_max - tile0_max)
            tile0_p = tl.exp(tile0_qk - tile0_max[:, None])
            acc *= tile0_scale[:, None]
            acc = tl.dot(tile0_p.to(tile0_v.dtype), tile0_v, acc=acc)
            e_sum = e_sum * tile0_scale + tl.sum(tile0_p, 1)
            e_max = tile0_max

            tile1 = tile0 + 1
            tile1_offs_n = offs_n + tile1 * BLOCK_N
            tile1_mask = tile1_offs_n < cur_batch_seq_len
            tile1_page = tl.load(
                Block_table + tile1_offs_n // PAGE_SIZE,
                mask=tile1_mask,
                other=0,
            )
            tile1_loc = tile1_page * PAGE_SIZE + tile1_offs_n % PAGE_SIZE
            tile1_v = tl.load(
                KV_cache + tile1_loc[:, None] * stride_kv_bs + offs_d_nope[None, :],
                mask=tile1_mask[:, None],
                other=0.0,
            )
            tile1_qk = tl.dot(q_nope, tl.trans(tile1_v))
            tile1_k_pe = tl.load(
                KV_cache + tile1_loc[None, :] * stride_kv_bs + offs_d_pe[:, None],
                mask=tile1_mask[None, :],
                other=0.0,
            )
            tile1_qk = tl.dot(q_pe, tile1_k_pe, acc=tile1_qk) * sm_scale
            tile1_qk = tl.where(tile1_mask[None, :], tile1_qk, float("-inf"))
            tile1_max = tl.maximum(tl.max(tile1_qk, 1), e_max)
            tile1_scale = tl.exp(e_max - tile1_max)
            tile1_p = tl.exp(tile1_qk - tile1_max[:, None])
            acc *= tile1_scale[:, None]
            acc = tl.dot(tile1_p.to(tile1_v.dtype), tile1_v, acc=acc)
            e_sum = e_sum * tile1_scale + tl.sum(tile1_p, 1)
            e_max = tile1_max
    else:
        loop_time = cur_batch_seq_len // BLOCK_N
        remainder = cur_batch_seq_len % BLOCK_N
        for _ in range(loop_time):
            kv_page_number = tl.load(Block_table + offs_n // PAGE_SIZE)
            kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE

            offs_v_c = kv_loc[:, None] * stride_kv_bs + offs_d_nope[None, :]
            v_c = tl.load(KV_cache + offs_v_c)
            k_c = tl.trans(v_c)
            qk = tl.dot(q_nope, k_c)
            offs_k_pe = kv_loc[None, :] * stride_kv_bs + offs_d_pe[:, None]
            k_pe = tl.load(KV_cache + offs_k_pe)
            qk = tl.dot(q_pe, k_pe, acc=qk)
            qk *= sm_scale
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc = tl.dot(p.to(v_c.dtype), v_c, acc=acc)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max
            offs_n += BLOCK_N

        if remainder:
            # offs_n already advanced in the full-tile loop.
            remainder_offs_n = offs_n
            mask_kvsplit = remainder_offs_n < cur_batch_seq_len
            kv_page_number = tl.load(
                Block_table + remainder_offs_n // PAGE_SIZE,
                mask=mask_kvsplit,
                other=0,
            )
            kv_loc = kv_page_number * PAGE_SIZE + remainder_offs_n % PAGE_SIZE
            offs_v_c = kv_loc[:, None] * stride_kv_bs + offs_d_nope[None, :]
            v_c = tl.load(
                KV_cache + offs_v_c,
                mask=mask_kvsplit[:, None],
                other=0.0,
            )
            k_c = tl.trans(v_c)
            qk = tl.dot(q_nope, k_c)
            offs_k_pe = kv_loc[None, :] * stride_kv_bs + offs_d_pe[:, None]
            k_pe = tl.load(
                KV_cache + offs_k_pe,
                mask=mask_kvsplit[None, :],
                other=0.0,
            )
            qk = tl.dot(q_pe, k_pe, acc=qk)
            qk *= sm_scale
            qk = tl.where(mask_kvsplit[None, :], qk, float("-inf"))
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc = tl.dot(p.to(v_c.dtype), v_c, acc=acc)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

    # Store output
    offs_o = (
        i_b * stride_o_b
        + i_sq * stride_o_sq
        + cur_head[:, None] * stride_o_h
        + offs_d_nope[None, :]
    )
    tl.store(
        Out + offs_o,
        tl.where(e_sum[:, None] > 0, acc / tl.maximum(e_sum[:, None], 1e-20), 0.0).to(
            Out.dtype.element_ty
        ),
        mask=mask_head[:, None],
    )

    # Store LSE
    lse_val = e_max + tl.math.log(e_sum)
    lse_offset = i_b * stride_lse_b + cur_head * stride_lse_h + i_sq
    tl.store(LSE + lse_offset, lse_val, mask=mask_head)


@triton.autotune(
    configs=_SPARSE_DECODE_CONFIGS,
    key=[
        "HQ",
        "DQK",
        "TOPK",
        "HAVE_ATTN_SINK",
        "HAVE_TOPK_LENGTH",
        "IS_FP8",
        "EMULATED_TLE",
    ],
)
@triton.jit
def _sparse_decode_kernel(
    q,
    kv,
    kv_scales,
    kv_rope,
    indices,
    attn_sink,
    topk_length,
    sm_scale: tl.constexpr,
    output,
    lse,
    stride_qb,
    stride_qsq,
    stride_qh,
    stride_kvn,
    stride_scales_n,
    stride_rope_n,
    stride_ib,
    stride_isq,
    stride_ob,
    stride_osq,
    stride_oh,
    stride_lseb,
    stride_lseh,
    SQ,
    HQ: tl.constexpr,
    DQK: tl.constexpr,
    SKV,
    TOPK: tl.constexpr,
    HAVE_ATTN_SINK: tl.constexpr,
    HAVE_TOPK_LENGTH: tl.constexpr,
    IS_FP8: tl.constexpr,
    EMULATED_TLE: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
):
    """
    Sparse decode kernel with online softmax.
    Grid: (batch_size * seq_q * ceil(HQ / BH),)
    Each program handles BH heads for one (batch, seq_q) position.

    For FP8 mode:
      - kv: [num_tokens, 512] float8_e4m3fn (NoPE part)
      - kv_scales: [num_tokens, 4] float32 (per-128-element scales)
      - kv_rope: [num_tokens, 64] bfloat16 (RoPE part)
    For BF16 mode:
      - kv: [num_tokens, DQK] bfloat16 (full KV)
      - kv_scales, kv_rope: unused
    """
    num_head_blocks: tl.constexpr = (HQ + BH - 1) // BH
    pid = tl.program_id(0)
    i_b = pid // (SQ * num_head_blocks)
    remainder = pid % (SQ * num_head_blocks)
    i_sq = remainder // num_head_blocks
    i_sq = i_sq.to(tl.int64)
    i_gbh = remainder % num_head_blocks
    gbh_base = i_gbh * BH

    DP: tl.constexpr = 512
    BDP: tl.constexpr = 256

    # Base pointers
    q_base = q + i_b * stride_qb + i_sq * stride_qsq + gbh_base * stride_qh
    kv_base = kv
    t_base = indices + i_b * stride_ib + i_sq * stride_isq
    attn_sink_ptr = attn_sink + gbh_base if HAVE_ATTN_SINK else 0
    topk_length_ptr = topk_length + i_b if HAVE_TOPK_LENGTH else 0
    o_base = output + i_b * stride_ob + i_sq * stride_osq + gbh_base * stride_oh
    l_base = lse + i_b * stride_lseb + gbh_base * stride_lseh + i_sq

    offs_h = tl.arange(0, BH)
    offs_d = tl.arange(0, BDP)
    if DQK == 576:
        offs_td = tl.arange(0, 64)
    offs_t = tl.arange(0, BK)

    # Load Q in two halves [BH, 256] x 2
    q_ptr = q_base + offs_h[:, None] * stride_qh + offs_d[None, :]
    q_blk0 = tl.load(q_ptr, eviction_policy="evict_first")
    q_blk1 = tl.load(q_ptr + BDP, eviction_policy="evict_first")
    if DQK == 576:
        tq_ptr = q_base + DP + offs_h[:, None] * stride_qh + offs_td[None, :]
        tq_blk = tl.load(tq_ptr, eviction_policy="evict_first")
    if EMULATED_TLE:
        # The emulated Ascend path keeps decoded FP8 values in FP32 so its
        # score/softmax accumulation matches the reference implementation.
        q_blk0 = q_blk0.to(tl.float32)
        q_blk1 = q_blk1.to(tl.float32)
        if DQK == 576:
            tq_blk = tq_blk.to(tl.float32)

    # Online softmax accumulators
    max_log = tl.full([BH], float("-inf"), dtype=tl.float32)
    sum_exp = tl.full([BH], 0.0, dtype=tl.float32)
    acc0 = tl.zeros([BH, BDP], dtype=tl.float32)
    acc1 = tl.zeros([BH, BDP], dtype=tl.float32)

    topk_len = tl.load(topk_length_ptr) if HAVE_TOPK_LENGTH else TOPK
    # Native TLE uses two pipe slots and advances the producer/consumer pair
    # together.  On Ascend/Triton 3.5.1, emulate the same ordering with a
    # sequential two-tile loop while keeping all data in registers.
    pair_blocks: tl.constexpr = 2 if EMULATED_TLE else 1
    # Use the compile-time TOPK capacity for the loop bound.  A runtime
    # ``range(tl.cdiv(topk_length, ...))`` becomes an expensive dynamic loop on
    # Ascend for large top-k (and can remain in that loop for many minutes).
    # Invalid lanes/tiles are already masked by ``t_msk`` and ``mask_ids``.
    num_pairs: tl.constexpr = (TOPK + BK * pair_blocks - 1) // (BK * pair_blocks)
    for pair in range(num_pairs):
        for pair_offset in range(pair_blocks):
            ck = pair * pair_blocks + pair_offset
            # The native TLE producer masks its final partial tile.  Keep the
            # same semantics without a dynamic ``if ck < NK`` branch: invalid
            # lanes load token zero, but remain masked out of the dot/softmax.
            t_ptr = BK * ck + offs_t
            t_msk = t_ptr < topk_len
            t_ptr += t_base
            kv_ids = tl.load(t_ptr, t_msk, other=-1)
            mask_ids = t_msk & (kv_ids < SKV) & (kv_ids >= 0)
            kv_ids = tl.where(mask_ids, kv_ids, 0)

            if IS_FP8:
                # FP8 mode: load FP8 values and dequantize with per-128-element scales
                # Load NoPE FP8 data: [BDP, BK] for each half
                kv_ptr = kv_base + offs_d[:, None] + kv_ids[None, :] * stride_kvn
                kv_fp8_0 = tl.load(
                    kv_ptr, mask=mask_ids[None, :], other=0.0, cache_modifier=".cg"
                )  # [256, BK] float8
                kv_fp8_1 = tl.load(
                    kv_ptr + BDP,
                    mask=mask_ids[None, :],
                    other=0.0,
                    cache_modifier=".cg",
                )  # [256, BK] float8

                # Load 4 scales per token separately
                # Scale layout: [num_tokens, 4] float32
                scale0 = tl.load(
                    kv_scales + kv_ids * stride_scales_n + 0,
                    mask=mask_ids,
                    other=0.0,
                )  # [BK]
                scale1 = tl.load(
                    kv_scales + kv_ids * stride_scales_n + 1,
                    mask=mask_ids,
                    other=0.0,
                )  # [BK]
                scale2 = tl.load(
                    kv_scales + kv_ids * stride_scales_n + 2,
                    mask=mask_ids,
                    other=0.0,
                )  # [BK]
                scale3 = tl.load(
                    kv_scales + kv_ids * stride_scales_n + 3,
                    mask=mask_ids,
                    other=0.0,
                )  # [BK]

                # Dequantize first half [256, BK]:
                #   elements [0:128] use scale0, elements [128:256] use scale1
                mask_lo = offs_d[:, None] < 128
                kv_blk0 = tl.where(
                    mask_lo,
                    kv_fp8_0.to(tl.float32) * scale0[None, :],
                    kv_fp8_0.to(tl.float32) * scale1[None, :],
                ).to(tl.bfloat16)

                # Dequantize second half [256, BK]:
                #   elements [0:128] use scale2, elements [128:256] use scale3
                kv_blk1 = tl.where(
                    mask_lo,
                    kv_fp8_1.to(tl.float32) * scale2[None, :],
                    kv_fp8_1.to(tl.float32) * scale3[None, :],
                ).to(tl.bfloat16)
            else:
                # BF16/float32 decoded cache used by the Ascend emulation.
                kv_ptr = kv_base + offs_d[:, None] + kv_ids[None, :] * stride_kvn
                kv_blk0 = tl.load(
                    kv_ptr, mask=mask_ids[None, :], other=0.0, cache_modifier=".cg"
                )  # [BDP, BK]
                kv_blk1 = tl.load(
                    kv_ptr + BDP,
                    mask=mask_ids[None, :],
                    other=0.0,
                    cache_modifier=".cg",
                )  # [BDP, BK]

            # Compute QK^T
            qk = tl.dot(q_blk0, kv_blk0, out_dtype=tl.float32)
            qk = tl.dot(q_blk1, kv_blk1, qk, out_dtype=tl.float32)
            if DQK == 576:
                if IS_FP8:
                    # RoPE part from separate tensor
                    rope_ptr = (
                        kv_rope + offs_td[:, None] + kv_ids[None, :] * stride_rope_n
                    )
                    tkv_blk = tl.load(
                        rope_ptr,
                        mask=mask_ids[None, :],
                        other=0.0,
                        cache_modifier=".cg",
                    )
                else:
                    tkv_ptr = (
                        kv_base + DP + offs_td[:, None] + kv_ids[None, :] * stride_kvn
                    )
                    tkv_blk = tl.load(
                        tkv_ptr,
                        mask=mask_ids[None, :],
                        other=0.0,
                        cache_modifier=".cg",
                    )
                qk = tl.dot(tq_blk, tkv_blk, qk, out_dtype=tl.float32)
            qk *= sm_scale

            # Mask invalid tokens
            qk = tl.where(mask_ids[None, :], qk, float("-inf"))

            # Online softmax
            new_max = tl.maximum(max_log, tl.max(qk, axis=1))
            exp_qk = tl.math.exp(qk - new_max[:, None])
            sum_qk = tl.sum(exp_qk, axis=1)
            alpha = tl.math.exp(max_log - new_max)
            sum_exp = sum_exp * alpha + sum_qk

            # Accumulate P @ V (V = K NoPE for MLA)
            prob_blk = exp_qk if EMULATED_TLE else exp_qk.to(tl.bfloat16)
            acc0 = tl.dot(
                prob_blk,
                kv_blk0.trans(),
                acc0 * alpha[:, None],
                out_dtype=tl.float32,
            )
            acc1 = tl.dot(
                prob_blk,
                kv_blk1.trans(),
                acc1 * alpha[:, None],
                out_dtype=tl.float32,
            )
            max_log = new_max

    # Finalize output
    valid_mask = max_log != float("-inf")
    max_log = tl.where(valid_mask, max_log, float("-inf"))

    orig_lse = max_log + tl.math.log(sum_exp)
    lse_out = tl.where(valid_mask, orig_lse, float("inf"))
    tl.store(l_base + offs_h * stride_lseh, lse_out)

    if HAVE_ATTN_SINK:
        sink = tl.load(attn_sink_ptr + offs_h)
        sum_exp_new_lse = tl.math.exp(orig_lse) + tl.math.exp(sink)
        factor = tl.math.exp(max_log) / sum_exp_new_lse
    else:
        factor = 1.0 / sum_exp

    out_vals0 = tl.where(valid_mask[:, None], acc0 * factor[:, None], 0.0)
    out_vals1 = tl.where(valid_mask[:, None], acc1 * factor[:, None], 0.0)

    # Store output
    o_ptr = o_base + offs_h[:, None] * stride_oh + offs_d[None, :]
    tl.store(o_ptr, out_vals0.to(tl.bfloat16))
    tl.store(o_ptr + BDP, out_vals1.to(tl.bfloat16))


@torch.no_grad()
def flash_mla_with_kvcache(
    q,
    k_cache,
    block_table,
    cache_seqlens,
    head_dim_v=512,
    *,
    softmax_scale=None,
    causal=False,
    paired_tiles=True,
    out=None,
):
    """Dense paged FP16/BF16 MLA returning output and [B, H, Tq] LSE.

    FP8 byte-cache decoding is deliberately not performed by this device API.
    """
    if q.device.type != "npu" or k_cache.device != q.device:
        raise ValueError("Ascend MLA requires Q and KV on the same NPU")
    if q.dtype not in (torch.float16, torch.bfloat16) or k_cache.dtype != q.dtype:
        raise TypeError("Dense KV-cache MLA supports matching FP16/BF16 inputs")
    b, sq, h, d = q.shape
    if k_cache.shape[2:] != (1, d) or d != head_dim_v + 64:
        raise ValueError("Require one KV head and a 64-dimensional RoPE tail")
    if q.stride(-1) != 1 or not k_cache.is_contiguous():
        raise ValueError("Require contiguous KV and contiguous Q head dimensions")
    if block_table.dtype != torch.int32 or cache_seqlens.dtype != torch.int32:
        raise TypeError("Block table and sequence lengths must be int32")
    if block_table.shape[0] != b or cache_seqlens.shape != (b,):
        raise ValueError("Metadata batch dimension mismatch")
    if block_table.device != q.device or cache_seqlens.device != q.device:
        raise ValueError("Metadata must be on the input NPU")
    if out is None:
        out = torch.empty((b, sq, h, head_dim_v), dtype=q.dtype, device=q.device)
    elif (
        out.shape != (b, sq, h, head_dim_v)
        or out.dtype != q.dtype
        or out.device != q.device
    ):
        raise ValueError("Output shape, dtype or device mismatch")
    if out.stride(-1) != 1:
        raise ValueError("Output head dimension must be contiguous")
    lse = torch.empty((b, h, sq), dtype=torch.float32, device=q.device)
    table = block_table.contiguous()
    kv = k_cache.view(-1, d)
    scale = d**-0.5 if softmax_scale is None else softmax_scale

    def grid(meta):
        return (triton.cdiv(h, meta["BLOCK_H"]), b * sq)

    _dense_decode_kernel[grid](
        q,
        *q.stride()[:3],
        kv,
        kv.stride(0),
        table,
        table.stride(0),
        cache_seqlens,
        out,
        *out.stride()[:3],
        lse,
        *lse.stride()[:2],
        scale,
        sq,
        h,
        d,
        head_dim_v,
        k_cache.shape[1],
        causal,
        paired_tiles,
    )
    return out, lse


@torch.no_grad()
def flash_mla_sparse_decode(
    q, decoded_kv, indices, *, softmax_scale=None, attn_sink=None, topk_length=None
):
    """Sparse decode over pre-decoded FP32 values using the source paired schedule.

    Q is BF16 [B,T,H,D]; decoded_kv is FP32 [tokens,D]; indices is int32
    [B,T,K]. CPU FP8 decoding and extra-cache packing belong to baseline adapters.
    """
    if (
        q.device.type != "npu"
        or decoded_kv.device != q.device
        or indices.device != q.device
    ):
        raise ValueError("Sparse MLA inputs must be on the same NPU")
    b, sq, h, d = q.shape
    if (
        q.dtype != torch.bfloat16
        or decoded_kv.dtype != torch.float32
        or indices.dtype != torch.int32
    ):
        raise TypeError("Require BF16 Q, FP32 decoded values and int32 indices")
    if d not in (512, 576) or h <= 0 or h % 16:
        raise ValueError("Require D=512/576 and a positive multiple of 16 heads")
    if decoded_kv.ndim != 2 or decoded_kv.shape[1] != d or indices.shape[:2] != (b, sq):
        raise ValueError("KV/indices shape mismatch")
    if not all(x.is_contiguous() for x in (q, decoded_kv, indices)):
        raise ValueError("Sparse MLA inputs must be contiguous")
    if attn_sink is not None and (
        attn_sink.shape != (h,) or attn_sink.dtype != torch.float32
    ):
        raise ValueError("Attention sink must be FP32 [H]")
    if topk_length is not None and (
        topk_length.shape != (b,) or topk_length.dtype != torch.int32
    ):
        raise ValueError("Top-k lengths must be int32 [B]")
    out = torch.empty((b, sq, h, 512), dtype=q.dtype, device=q.device)
    lse = torch.empty((b, h, sq), dtype=torch.float32, device=q.device)
    scale = d**-0.5 if softmax_scale is None else softmax_scale
    grid = (b * sq * triton.cdiv(h, 16),)
    _sparse_decode_kernel[grid](
        q,
        decoded_kv,
        decoded_kv,
        decoded_kv,
        indices,
        attn_sink,
        topk_length,
        scale,
        out,
        lse,
        *q.stride()[:3],
        decoded_kv.stride(0),
        decoded_kv.stride(0),
        decoded_kv.stride(0),
        *indices.stride()[:2],
        *out.stride()[:3],
        *lse.stride()[:2],
        sq,
        h,
        d,
        decoded_kv.shape[0],
        indices.shape[-1],
        attn_sink is not None,
        topk_length is not None,
        False,
        True,
    )
    return out, lse
