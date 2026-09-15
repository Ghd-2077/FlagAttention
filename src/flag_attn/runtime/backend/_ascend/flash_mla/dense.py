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

"""Ascend paged MLA with workload partitioning and compact split reduction.

This is the source's ordinary-Triton equivalent schedule, not native TLE pipes.
"""
import math
from functools import lru_cache
import torch
import triton
import triton.language as tl

FLASH_MLA_META_FIELDS = 8
FLASH_MLA_BLOCK_N = 64
FLASH_MLA_FIXED_OVERHEAD_BLOCKS = 5
FLASH_MLA_COMBINE_BLOCK_H = 8
FLASH_MLA_COMBINE_BLOCK_D = 256


@triton.jit
def _flash_mla_scalar_load(ptr, USE_PARTIAL_TLE: tl.constexpr):
    return tl.load(ptr)


def _contiguous_if_needed(value):
    return value if value.is_contiguous() else value.contiguous()


def _same_device(lhs, rhs):
    return lhs == rhs


@triton.jit
def flash_mla_sched_meta_kernel_v3(
    B_seq_len,
    Sched_meta,
    Num_splits,
    CombineReqIds,
    NumCombineReqs,
    BLOCK_B: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    FIXED_OVERHEAD_NUM_BLOCKS: tl.constexpr,
    NUM_SM_PARTS: tl.constexpr,
    META_FIELDS: tl.constexpr,
):
    offs_b = tl.arange(0, BLOCK_B)
    mask_b = offs_b < BATCH_SIZE
    seqlens = tl.load(B_seq_len + offs_b, mask=mask_b, other=0)
    num_blocks_vec = tl.cdiv(tl.maximum(seqlens, 1), BLOCK_SIZE_N)
    total_num_blocks = tl.sum(
        tl.where(mask_b, num_blocks_vec + FIXED_OVERHEAD_NUM_BLOCKS, 0), axis=0
    )
    payload = tl.maximum(
        tl.cdiv(total_num_blocks, NUM_SM_PARTS) + FIXED_OVERHEAD_NUM_BLOCKS,
        FIXED_OVERHEAD_NUM_BLOCKS + 2,
    )

    now_req_idx = 0
    now_block = 0
    now_n_split_idx = 0
    cum_num_splits = 0
    combine_req_count = 0
    tl.store(Num_splits, 0)

    for part in tl.range(0, NUM_SM_PARTS, 1):
        begin_req_idx = now_req_idx
        begin_block_idx = now_block
        begin_split_idx = now_n_split_idx
        is_first_req_splitted = now_block != 0
        remain_payload = payload

        while (now_req_idx < BATCH_SIZE) & (remain_payload > 0):
            cur_seq_len = tl.load(B_seq_len + now_req_idx)
            cur_num_blocks = tl.cdiv(tl.maximum(cur_seq_len, 1), BLOCK_SIZE_N)
            now_remain_blocks = cur_num_blocks - now_block
            if remain_payload + 1 >= now_remain_blocks + FIXED_OVERHEAD_NUM_BLOCKS:
                req_num_splits = now_n_split_idx + 1
                if req_num_splits != 1:
                    tl.store(CombineReqIds + combine_req_count, now_req_idx)
                    combine_req_count += 1
                cum_num_splits += req_num_splits
                tl.store(Num_splits + now_req_idx + 1, cum_num_splits)
                remain_payload -= now_remain_blocks + FIXED_OVERHEAD_NUM_BLOCKS
                now_req_idx += 1
                now_block = 0
                now_n_split_idx = 0
            else:
                if remain_payload - FIXED_OVERHEAD_NUM_BLOCKS > 0:
                    split_blocks = remain_payload - FIXED_OVERHEAD_NUM_BLOCKS
                    # The WS TLE kernel cannot safely handle a one-block
                    # partial split in this schedule.
                    split_blocks = tl.where(
                        (split_blocks > 1) & (now_remain_blocks - split_blocks == 1),
                        split_blocks - 1,
                        split_blocks,
                    )
                    now_block += split_blocks
                    now_n_split_idx += 1
                remain_payload = 0

        if now_block > 0:
            end_req_idx = now_req_idx
            end_block_idx = now_block
        else:
            end_req_idx = now_req_idx - 1
            if end_req_idx >= 0:
                end_seq_len = tl.load(B_seq_len + end_req_idx)
                end_block_idx = tl.where(
                    end_seq_len == 0, 0, tl.cdiv(end_seq_len, BLOCK_SIZE_N)
                )
            else:
                end_block_idx = 0

        meta = Sched_meta + part * META_FIELDS
        if begin_req_idx >= BATCH_SIZE:
            tl.store(meta + 0, BATCH_SIZE)
            tl.store(meta + 1, BATCH_SIZE - 1)
            tl.store(meta + 2, 0)
            tl.store(meta + 3, 0)
            tl.store(meta + 4, 0)
            tl.store(meta + 5, 0)
            tl.store(meta + 6, 0)
            tl.store(meta + 7, 0)
        else:
            end_seq_len = tl.load(B_seq_len + end_req_idx)
            last_block_exclusive = tl.where(
                end_seq_len == 0, 0, tl.cdiv(end_seq_len, BLOCK_SIZE_N)
            )
            is_last_req_splitted = (end_block_idx != last_block_exclusive) & (
                end_seq_len != 0
            )
            if begin_req_idx == end_req_idx:
                same_req_split = is_first_req_splitted | is_last_req_splitted
                is_first_req_splitted = same_req_split
                is_last_req_splitted = same_req_split

            tl.store(meta + 0, begin_req_idx)
            tl.store(meta + 1, end_req_idx)
            tl.store(meta + 2, begin_block_idx)
            tl.store(meta + 3, end_block_idx)
            tl.store(meta + 4, begin_split_idx)
            tl.store(meta + 5, is_first_req_splitted.to(tl.int32))
            tl.store(meta + 6, is_last_req_splitted.to(tl.int32))
            tl.store(meta + 7, 0)

    tl.store(NumCombineReqs, combine_req_count)


@triton.jit
def flash_mla_splitkv_native_equiv_kernel(
    Q,
    Kv,
    Block_table,
    B_seq_len,
    Sched_meta,
    Num_splits,
    O,
    OAccum,
    LSE_accum,
    sm_scale,
    head_num,
    stride_q_row,
    stride_kv_token,
    stride_block_table_b,
    stride_o_row,
    stride_oaccum_split,
    stride_oaccum_h,
    stride_lseaccum_split,
    stride_lseaccum_h,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCKS_PER_PAGE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    D_CHUNK: tl.constexpr,
    META_FIELDS: tl.constexpr,
    USE_PARTIAL_TLE: tl.constexpr,
):
    """Native Triton replacement for the FlashMLA TLE split-KV pipeline.

    The TLE implementation uses a producer, two warp-specialized consumers,
    and CTA pipes to stage each KV pair.  Triton 3.5.1 on Ascend has no pipe
    or warp-specialization compiler primitives, so this kernel computes the
    same partitioned softmax directly from global memory.  Each program owns
    one head block and one scheduler partition; partitions remain independent
    and are merged by ``flash_mla_combine_kernel_compact``.
    """
    partition_idx = tl.program_id(1)
    m_block_idx = tl.program_id(0)
    meta_base = Sched_meta + partition_idx * META_FIELDS
    begin_req_idx = _flash_mla_scalar_load(meta_base + 0, USE_PARTIAL_TLE)
    end_req_idx = _flash_mla_scalar_load(meta_base + 1, USE_PARTIAL_TLE)
    begin_block_idx_meta = _flash_mla_scalar_load(meta_base + 2, USE_PARTIAL_TLE)
    end_block_idx_meta = _flash_mla_scalar_load(meta_base + 3, USE_PARTIAL_TLE)
    begin_split_idx = _flash_mla_scalar_load(meta_base + 4, USE_PARTIAL_TLE)
    is_first_req_splitted = _flash_mla_scalar_load(meta_base + 5, USE_PARTIAL_TLE) != 0
    is_last_req_splitted = _flash_mla_scalar_load(meta_base + 6, USE_PARTIAL_TLE) != 0

    head_base = m_block_idx * BLOCK_M
    offs_h = tl.arange(0, BLOCK_M)
    head_offsets = head_base + offs_h
    mask_h = head_offsets < head_num
    offs_d_v = tl.arange(0, HEAD_DIM_V)
    offs_d_k = tl.arange(0, D_CHUNK)

    for batch_idx in tl.range(begin_req_idx, end_req_idx + 1):
        seq_len = _flash_mla_scalar_load(B_seq_len + batch_idx, USE_PARTIAL_TLE)
        start_block_idx = tl.where(batch_idx == begin_req_idx, begin_block_idx_meta, 0)
        full_end_block_idx = tl.cdiv(seq_len, PAGE_SIZE)
        end_block_idx = tl.where(
            batch_idx == end_req_idx, end_block_idx_meta, full_end_block_idx
        )

        n_split_idx = tl.where(batch_idx == begin_req_idx, begin_split_idx, 0)
        no_split_middle = (batch_idx != begin_req_idx) & (batch_idx != end_req_idx)
        no_split_first = (batch_idx == begin_req_idx) & (~is_first_req_splitted)
        no_split_last = (batch_idx == end_req_idx) & (~is_last_req_splitted)
        is_no_split = no_split_middle | no_split_first | no_split_last
        if begin_req_idx == end_req_idx:
            is_no_split = ~is_first_req_splitted
        split_idx = (
            _flash_mla_scalar_load(Num_splits + batch_idx, USE_PARTIAL_TLE)
            + n_split_idx
        )

        q_row = batch_idx * head_num + head_offsets
        q_nope = tl.load(
            Q + q_row[:, None] * stride_q_row + offs_d_v[None, :],
            mask=mask_h[:, None],
            other=0.0,
        )
        q_pe = tl.load(
            Q + q_row[:, None] * stride_q_row + (HEAD_DIM_V + offs_d_k)[None, :],
            mask=mask_h[:, None],
            other=0.0,
        )

        e_max = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
        e_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM_V], dtype=tl.float32)

        for block_idx in tl.range(start_block_idx, end_block_idx):
            page = _flash_mla_scalar_load(
                Block_table + batch_idx * stride_block_table_b + block_idx,
                USE_PARTIAL_TLE,
            )
            # One scheduler block is a page.  The native kernel uses two
            # half-page tiles to stay within the Ascend 910B UB budget.
            for page_part in range(BLOCKS_PER_PAGE):
                offs_n = tl.arange(0, BLOCK_N)
                token_in_page = page_part * BLOCK_N + offs_n
                logical_token = block_idx * PAGE_SIZE + token_in_page
                valid_n = logical_token < seq_len
                kv_row = page * PAGE_SIZE + token_in_page

                k_nope = tl.load(
                    Kv + kv_row[:, None] * stride_kv_token + offs_d_v[None, :],
                    mask=valid_n[:, None],
                    other=0.0,
                )
                k_pe = tl.load(
                    Kv
                    + kv_row[:, None] * stride_kv_token
                    + (HEAD_DIM_V + offs_d_k)[None, :],
                    mask=valid_n[:, None],
                    other=0.0,
                )

                qk = tl.dot(q_nope, tl.trans(k_nope), out_dtype=tl.float32)
                qk = tl.dot(q_pe, tl.trans(k_pe), acc=qk, out_dtype=tl.float32)
                qk *= sm_scale
                qk = tl.where(valid_n[None, :], qk, float("-inf"))

                new_max = tl.maximum(tl.max(qk, axis=1), e_max)
                re_scale = tl.exp(e_max - new_max)
                p = tl.exp(qk - new_max[:, None])
                e_sum = e_sum * re_scale + tl.sum(p, axis=1)
                acc = acc * re_scale[:, None]
                acc = tl.dot(
                    p.to(Kv.dtype.element_ty),
                    k_nope,
                    acc=acc,
                    out_dtype=tl.float32,
                )
                e_max = new_max

        valid = e_sum > 0.0
        safe_sum = tl.where(valid, e_sum, 1.0)
        inv_sum = tl.fdiv(1.0, safe_sum)
        out_vals = tl.where(valid[:, None], acc * inv_sum[:, None], 0.0)

        if is_no_split:
            tl.store(
                O
                + q_row[:, None] * 0
                + batch_idx * stride_o_row * head_num
                + head_offsets[:, None] * stride_o_row
                + offs_d_v[None, :],
                out_vals.to(O.dtype.element_ty),
                mask=mask_h[:, None],
            )
        else:
            tl.store(
                OAccum
                + split_idx * stride_oaccum_split
                + head_offsets[:, None] * stride_oaccum_h
                + offs_d_v[None, :],
                out_vals.to(OAccum.dtype.element_ty),
                mask=mask_h[:, None],
            )
            lse_vals = tl.where(valid, tl.log(safe_sum) + e_max, float("-inf"))
            tl.store(
                LSE_accum
                + split_idx * stride_lseaccum_split
                + head_offsets * stride_lseaccum_h,
                lse_vals,
                mask=mask_h,
            )


@triton.jit
def flash_mla_combine_kernel_compact(
    O_accum,
    LSE_accum,
    Num_splits,
    CombineReqIds,
    NumCombineReqs,
    O,
    head_num,
    stride_oaccum_split,
    stride_oaccum_h,
    stride_lseaccum_split,
    stride_lseaccum_h,
    stride_o_b,
    stride_o_h,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
):
    task_idx = tl.program_id(0)
    h_block_idx = tl.program_id(1)
    d_block_idx = tl.program_id(2)

    num_tasks = tl.load(NumCombineReqs)
    if task_idx < num_tasks:
        batch_idx = tl.load(CombineReqIds + task_idx)

        offs_h = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_d = d_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_h = offs_h < head_num
        mask_d = offs_d < HEAD_DIM_V

        start_split = tl.load(Num_splits + batch_idx)
        end_split = tl.load(Num_splits + batch_idx + 1)
        my_num_splits = end_split - start_split

        if my_num_splits > 1:
            max_lse = tl.full([BLOCK_H], value=float("-inf"), dtype=tl.float32)
            for s in tl.range(0, my_num_splits):
                lse_s = tl.load(
                    LSE_accum
                    + (start_split + s) * stride_lseaccum_split
                    + offs_h * stride_lseaccum_h,
                    mask=mask_h,
                    other=float("-inf"),
                )
                max_lse = tl.maximum(max_lse, lse_s)

            acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
            sum_w = tl.zeros([BLOCK_H], dtype=tl.float32)
            valid_row = max_lse != float("-inf")
            for s in tl.range(0, my_num_splits):
                lse_s = tl.load(
                    LSE_accum
                    + (start_split + s) * stride_lseaccum_split
                    + offs_h * stride_lseaccum_h,
                    mask=mask_h,
                    other=float("-inf"),
                )
                w = tl.where(valid_row, tl.exp(lse_s - max_lse), 0.0)
                sum_w += w

                o_s = tl.load(
                    O_accum
                    + (start_split + s) * stride_oaccum_split
                    + offs_h[:, None] * stride_oaccum_h
                    + offs_d[None, :],
                    mask=mask_h[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                acc += w[:, None] * o_s

            inv_sum = tl.where(sum_w > 0.0, tl.fdiv(1.0, sum_w), 0.0)
            acc = acc * inv_sum[:, None]

            tl.store(
                O
                + batch_idx * stride_o_b
                + offs_h[:, None] * stride_o_h
                + offs_d[None, :],
                acc.to(O.dtype.element_ty),
                mask=mask_h[:, None] & mask_d[None, :],
            )


class FlashMLADecodePlan:
    """Reusable Ascend decode workspace; one plan must not run concurrently.

    The source schedule supports one query and one KV head per request, 64-token
    pages and a 64-dimensional RoPE tail. Workspaces are owned by the caller.
    """

    def __init__(self, *, b, h_q, dv, dtype, device):
        device = torch.device(device)
        if device.type != "npu":
            raise ValueError("Ascend MLA requires an NPU device")
        if device.index is None:
            device = torch.device("npu", torch.npu.current_device())
        if dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("Ascend MLA supports FP16 and BF16")
        if b <= 0 or h_q <= 0 or dv < 16 or dv & (dv - 1):
            raise ValueError(
                "Require positive batch/heads and a power-of-two value dimension >= 16"
            )
        self.b, self.s_q, self.h_q, self.h_kv = b, 1, h_q, 1
        self.d, self.dv, self.block_size = dv + 64, dv, 64
        self.dtype, self.device = dtype, device
        self.reuse_output, self.out = False, None
        self.partial_tle, self.native_equiv = False, True
        self.d_chunk, self.kernel_block_m, self.kernel_block_n = 64, 16, 32
        self.sm_scale = 1 / math.sqrt(self.d)
        self.num_m_blocks = triton.cdiv(h_q, self.kernel_block_m)
        self.num_sms = torch.npu.get_device_properties(
            device.index
        ).multi_processor_count
        self.num_sm_parts = max(int(self.num_sms) // self.num_m_blocks, 1)
        self.total_num_splits = b + self.num_sm_parts
        self.max_combine_reqs = min(b, self.num_sm_parts)
        self.block_b = triton.next_power_of_2(b)
        self.sched_meta = torch.empty(
            (self.num_sm_parts, 8), dtype=torch.int32, device=device
        )
        self.num_splits = torch.empty((b + 1,), dtype=torch.int32, device=device)
        self.combine_req_ids = torch.empty(
            (self.max_combine_reqs,), dtype=torch.int32, device=device
        )
        self.num_combine_reqs = torch.empty((1,), dtype=torch.int32, device=device)
        self.out_accum = torch.empty(
            (self.total_num_splits, h_q, dv), dtype=dtype, device=device
        )
        self.lse_accum = torch.empty(
            (self.total_num_splits, h_q), dtype=torch.float32, device=device
        )
        self.metadata_valid = False
        self._last_cache_seqlens_ref = None

    def invalidate_metadata(self) -> None:
        self.metadata_valid = False
        self._last_cache_seqlens_ref = None

    def plan(self, cache_seqlens: torch.Tensor) -> None:
        if cache_seqlens.dtype != torch.int32:
            raise TypeError("cache_seqlens must be int32")
        if not _same_device(cache_seqlens.device, self.device):
            raise ValueError("cache_seqlens device mismatch")
        if cache_seqlens.ndim != 1 or cache_seqlens.shape[0] != self.b:
            raise ValueError("cache_seqlens shape mismatch")

        cache_seqlens_tle = _contiguous_if_needed(cache_seqlens)
        flash_mla_sched_meta_kernel_v3[(1,)](
            cache_seqlens_tle,
            self.sched_meta,
            self.num_splits,
            self.combine_req_ids,
            self.num_combine_reqs,
            BLOCK_B=self.block_b,
            BATCH_SIZE=self.b,
            BLOCK_SIZE_N=self.block_size,
            FIXED_OVERHEAD_NUM_BLOCKS=FLASH_MLA_FIXED_OVERHEAD_BLOCKS,
            NUM_SM_PARTS=self.num_sm_parts,
            META_FIELDS=FLASH_MLA_META_FIELDS,
            num_warps=1,
            num_stages=1,
        )

        self.metadata_valid = True
        self._last_cache_seqlens_ref = cache_seqlens_tle

    def _check_run_inputs(
        self,
        q: torch.Tensor,
        blocked_k: torch.Tensor,
        block_table: torch.Tensor,
    ) -> None:
        if not (
            _same_device(q.device, self.device)
            and _same_device(blocked_k.device, self.device)
            and _same_device(block_table.device, self.device)
        ):
            raise ValueError("device mismatch")
        if q.dtype != self.dtype or blocked_k.dtype != self.dtype:
            raise TypeError("dtype mismatch")
        if block_table.dtype != torch.int32:
            raise TypeError("block_table must be int32")
        if q.ndim != 4 or tuple(q.shape) != (
            self.b,
            self.s_q,
            self.h_q,
            self.d,
        ):
            raise ValueError("q shape mismatch")
        if (
            blocked_k.ndim != 4
            or blocked_k.shape[1] != self.block_size
            or blocked_k.shape[2] != self.h_kv
            or blocked_k.shape[3] != self.d
        ):
            raise ValueError("blocked_k shape mismatch")
        if block_table.ndim != 2 or block_table.shape[0] != self.b:
            raise ValueError("block_table shape mismatch")

    def _get_out_tensor(self, out: torch.Tensor | None) -> torch.Tensor:
        if out is not None:
            if not _same_device(out.device, self.device):
                raise ValueError("out device mismatch")
            if out.dtype != self.dtype:
                raise TypeError("out dtype mismatch")
            if tuple(out.shape) != (self.b * self.s_q, self.h_q, self.dv):
                raise ValueError("out shape must be (b * s_q, h_q, dv)")
            return out
        if self.reuse_output:
            if self.out is None:
                self.out = torch.empty(
                    (self.b * self.s_q, self.h_q, self.dv),
                    dtype=self.dtype,
                    device=self.device,
                )
            return self.out
        return torch.empty(
            (self.b * self.s_q, self.h_q, self.dv),
            dtype=self.dtype,
            device=self.device,
        )

    def _run_native_equiv(
        self,
        q: torch.Tensor,
        blocked_k: torch.Tensor,
        block_table: torch.Tensor,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the TLE algorithm with ordinary Triton global-memory loads.

        This is deliberately separate from the common FlashMLA path.  It
        retains the TLE scheduler/partition/combine protocol while replacing
        CTA pipes and warp-specialized consumers with one native Triton
        program per (head block, scheduler partition).
        """
        q_native = _contiguous_if_needed(q).view(self.b * self.s_q * self.h_q, self.d)
        blocked_k_native = _contiguous_if_needed(blocked_k)
        kv_flat = blocked_k_native.view(-1, self.d)
        block_table_native = _contiguous_if_needed(block_table)
        out_native = self._get_out_tensor(out)
        out_flat = out_native.view(self.b * self.s_q * self.h_q, self.dv)

        if self.block_size % self.kernel_block_n:
            raise ValueError(
                "FlashMLA native TLE replacement requires block_size divisible "
                f"by {self.kernel_block_n}, got {self.block_size}"
            )

        flash_mla_splitkv_native_equiv_kernel[(self.num_m_blocks, self.num_sm_parts)](
            q_native,
            kv_flat,
            block_table_native,
            self._last_cache_seqlens_ref,
            self.sched_meta,
            self.num_splits,
            out_flat,
            self.out_accum,
            self.lse_accum,
            self.sm_scale,
            self.h_q,
            q_native.stride(0),
            kv_flat.stride(0),
            block_table_native.stride(0),
            out_flat.stride(0),
            self.out_accum.stride(0),
            self.out_accum.stride(1),
            self.lse_accum.stride(0),
            self.lse_accum.stride(1),
            BLOCK_M=self.kernel_block_m,
            BLOCK_N=self.kernel_block_n,
            BLOCKS_PER_PAGE=self.block_size // self.kernel_block_n,
            PAGE_SIZE=self.block_size,
            HEAD_DIM_V=self.dv,
            HEAD_DIM=self.d,
            D_CHUNK=self.d_chunk,
            META_FIELDS=FLASH_MLA_META_FIELDS,
            USE_PARTIAL_TLE=self.partial_tle,
            num_warps=4,
            num_stages=1,
        )

        flash_mla_combine_kernel_compact[
            (
                self.max_combine_reqs,
                triton.cdiv(self.h_q, FLASH_MLA_COMBINE_BLOCK_H),
                triton.cdiv(self.dv, FLASH_MLA_COMBINE_BLOCK_D),
            )
        ](
            self.out_accum,
            self.lse_accum,
            self.num_splits,
            self.combine_req_ids,
            self.num_combine_reqs,
            out_native,
            self.h_q,
            self.out_accum.stride(0),
            self.out_accum.stride(1),
            self.lse_accum.stride(0),
            self.lse_accum.stride(1),
            out_native.stride(0),
            out_native.stride(1),
            BLOCK_H=FLASH_MLA_COMBINE_BLOCK_H,
            BLOCK_D=FLASH_MLA_COMBINE_BLOCK_D,
            HEAD_DIM_V=self.dv,
            num_warps=4,
            num_stages=1,
        )
        return out_native.view(self.b, self.s_q, self.h_q, self.dv)

    def run(
        self,
        q: torch.Tensor,
        blocked_k: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor | None = None,
        *,
        update_metadata: bool = True,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._check_run_inputs(q, blocked_k, block_table)
        if update_metadata:
            if cache_seqlens is None:
                raise ValueError("cache_seqlens is required when update_metadata=True")
            self.plan(cache_seqlens)
        elif not self.metadata_valid or self._last_cache_seqlens_ref is None:
            raise RuntimeError("metadata is not valid; call plan(cache_seqlens) first")

        return self._run_native_equiv(q, blocked_k, block_table, out)


@lru_cache(maxsize=32)
def _cached_decode_plan(b, h_q, dv, dtype, device, stream):
    # Include the NPU stream in the cache key so concurrent streams do not share
    # writable scheduler/partial-output buffers. Same-stream work stays ordered.
    return FlashMLADecodePlan(b=b, h_q=h_q, dv=dv, dtype=dtype, device=device)


@torch.no_grad()
def flash_mla(
    q,
    block_table,
    blocked_k,
    max_seqlen_pad,
    block_size,
    b,
    s_q,
    cache_seqlens,
    h_q,
    h_kv,
    d,
    dv,
    causal,
    *,
    plan=None,
):
    """Source-compatible Ascend single-query paged MLA entry point.

    Retains the bounded plan cache and refreshes scheduling metadata each call.
    Explicit plans let callers control workspace lifetime and graph capture.
    """
    if q.device.type != "npu":
        raise ValueError("Ascend MLA requires NPU tensors")
    if not causal or s_q != 1 or h_kv != 1 or block_size != 64 or d != dv + 64:
        raise ValueError(
            "Require causal single-query MQA, 64-token pages and a 64-dimensional RoPE tail"
        )
    if tuple(q.shape) != (b, s_q, h_q, d):
        raise ValueError("Q shape disagrees with the supplied dimensions")
    if plan is None:
        stream = torch.npu.current_stream(q.device).npu_stream
        plan = _cached_decode_plan(b, h_q, dv, q.dtype, q.device, stream)
    return plan.run(q, blocked_k, block_table, cache_seqlens)
