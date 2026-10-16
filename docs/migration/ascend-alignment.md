# Ascend alignment to locked upstream (2026-10-16)

## Scope and rollback

- Fork baseline: `cc675a5f7f456416ab75e08ac48b079c774d533f`.
- Locked upstream: `17a2f54463e61f1906f3a684dc1da7413938fffd`.
- Common ancestor: `d9adc17cfc2b4ac3776537548886da3de20e8085`.
- Divergence: fork 15 commits, upstream 20 commits.
- Local migration branch: `chore/align-upstream-20261016`.
- Local backup: `backup/npu-ghd-pre-merge` at the fork baseline.
- Original `npu-ghd` stays at its baseline; no push or PR is authorized by this task.
- Rollback: use the backup checkout. Kernel sources, math, tile sizes, launch
  configuration and numerical policies are preserved during migration.

## Environment

Validated inside server container `ghd-sageattn`, project
`/data/ghd/pr/flagattention-sage-integration`:

| Component | Observed value |
| --- | --- |
| Python | `/usr/local/python3.11.15/bin/python` |
| PyTorch | `2.9.0+cpu` (NPU extension available) |
| torch_npu | `2.9.0.post2` |
| Triton | `3.5.1` (Ascend/TLE build) |
| Chip | `Ascend910B4-1` |
| Device selection | `ASCEND_RT_VISIBLE_DEVICES=4` |
| CANN | `latest` resolves to `/usr/local/Ascend/cann-9.0.0` |

The locked upstream CI specifies runner `910b-flaggems-weekly`, container
`harbor.baai.ac.cn/flagos-app/vllm0.24.0-ascend-cann9.0.0:2.2.0-0.3.0`.
Local validation does not prove that GitHub self-hosted CI is accessible.

## P0 audit and file ownership

Both the original local branch and server worktree were clean at baseline.
The origin branch was fetched and confirmed at `cc675a5`; the upstream object
was fetched by its locked SHA, without advancing the target to a newer main.

| Classification | Paths | Action |
| --- | --- | --- |
| Inherit upstream | Public namespaces, `op_registry.py`, `testing/backend.py`, runtime, inventory, runner, packaging, CI and generic tests/benchmarks | Keep the locked upstream framework |
| Preserve fork | Ascend GLA, KDA, NSA, MiniMax and common primitives listed below | Keep implementations; register only after checking contracts |
| Manual integration | `_ascend/__init__.py`, GDN2 exports and validation, both Sage sources, `ops/__init__.py`, `.gitignore` | Keep fork fixes and adopt registry entry points |
| Rename-aware integration | `benchmark/parallel_nsa_benchmark.py` -> upstream `benchmark/test_parallel_nsa_benchmark.py` | Retain upstream rename and review lost fork content before accepting |

The 16 fork-only paths in a no-rename two-tree comparison are:

1. `benchmark/parallel_nsa_benchmark.py` (upstream renamed this benchmark).
2. `src/flag_attn/runtime/backend/_ascend/gla/__init__.py`.
3. `src/flag_attn/runtime/backend/_ascend/gla/chunk.py`.
4. `src/flag_attn/runtime/backend/_ascend/kda/__init__.py`.
5. `src/flag_attn/runtime/backend/_ascend/kda/chunk.py`.
6. `src/flag_attn/runtime/backend/_ascend/kda/forward.py`.
7. `src/flag_attn/runtime/backend/_ascend/minimax_sparse_attention/__init__.py`.
8. `src/flag_attn/runtime/backend/_ascend/minimax_sparse_attention/index_topk.py`.
9. `src/flag_attn/runtime/backend/_ascend/minimax_sparse_attention/sparse_attn.py`.
10. `src/flag_attn/runtime/backend/_ascend/nsa/__init__.py`.
11. `src/flag_attn/runtime/backend/_ascend/nsa/compression.py`.
12. `src/flag_attn/runtime/backend/_ascend/nsa/forward.py`.
13. `src/flag_attn/runtime/backend/_ascend/nsa/kernels.py`.
14. `src/flag_attn/runtime/backend/_ascend/ops/bwd_preprocess.py`.
15. `src/flag_attn/runtime/backend/_ascend/ops/cumsum.py`.
16. `src/flag_attn/runtime/backend/_ascend/ops/mean_pooling.py`.

## Migration-before validation

Logs remain local to the server in ignored
`BaselineBenchmark/migration-20261016/`. They must not enter Git.

```sh
ASCEND_RT_VISIBLE_DEVICES=4 PYTHONPATH=src:. /usr/local/python3.11.15/bin/python \
  -m pytest BaselineBenchmark/validation-current/test_gdn2.py \
  src/flag_attn/runtime/backend/_ascend/test_sage_attention.py -q --disable-warnings
```

Result: **10 passed, 1 warning in 13.58s**, exit 0; `pre-accuracy.log`.
The suite checks GDN2 lengths 16/33 against recurrence and Sage layouts,
grouped heads, masks and partial blocks. It does not prove all supported shapes.

Performance protocol: fixed seed 42, same physical device, synchronized wall
time including host dispatch; 5 warmups and 30 repetitions, median/P10/P90.
GDN2 `[B=1,T=128,H=1,K=64,V=64]` BF16; Sage `[B=1,H=2,T=128/256,D=64]`
FP16, HND, unmasked, quantization excluded. Source and log are
`protocol.py` and `pre-performance.log` in the same ignored directory.

Before-migration latency (ms):

| Operator | P10 | Median | P90 |
| --- | ---: | ---: | ---: |
| GDN2 T128 | 0.637028 | 0.640668 | 0.661935 |
| Sage T128 | 0.296561 | 0.299299 | 0.312621 |
| Sage T256 | 0.297467 | 0.300845 | 0.312095 |

Protocol exit 0. The old NSA benchmark is byte-identical to the common
ancestor; accepting its upstream rename does not drop fork-specific changes.

## Registration audit (P2)

| Operator | Contract boundary | Initial decision |
| --- | --- | --- |
| GDN2 | Inference only, BT=16 | Registered through `gdn2/api.py`, exact public parameter order; differentiable calls fail explicitly; original internal entry preserved |
| Sage | Public forward aliases and shared quantization; existing masked framework path | Registered with upstream aliases, fork mask/maxnreg behavior preserved |
| GLA | Same parameters/defaults, output/final state, variable length and backward | Registered to `gla.chunk_gla`; NPU regressions in P3 |
| KDA | Registry key `chunk_kda_fwd_infer`; Ascend wrapper has distinct defaults/options | Registered through matching inference signature, preserves `safe_gate=False` default; rejects precomputed chunk_indices explicitly |
| NSA selected | Returns output/LSE, lacks full backward API | Keep internal; do not register as full `parallel_nsa` |
| NSA compression | Same q/k/v/block_size/scale/cu_seqlens and output/LSE; backward supported | Registered to `nsa.compression`; NPU regressions in P3 |
| MiniMax | Prefill and four index primitives match parameter/default/return contracts | Registered prefill and index primitives; decode attention intentionally not registered because fork simply reexports Generic; no independent Ascend decode implementation |
| ops | Internal shared primitives without public generic contracts | Keep internal |

## Stage gates

- P0: PASS; audit, backup and migration branches exist; pre-migration accuracy
  and fixed-shape timing captured on 910B; original `npu-ghd` unchanged.
- P1: PASS. Merged the locked upstream, resolving `.gitignore` by retaining
  local MLA exclusion and Sage forward by preserving fork mask/maxnreg fixes
  plus upstream's named grid function. All 15 unique Ascend source files remain.
  `python -m compileall -q src` passed locally and on the server.
  Runtime/runner pytest initially had 36 pass / 1 failure: upstream #99 added
  SDK metadata without updating the environment dictionary assertion. Added
  a deterministic SDK probe mock/expectation; rerun: **37 passed**, exit 0.
  Logs: `/tmp/migration-p1-tests{,-rerun}.log` in the container.
  Local Windows lacks torch/Triton; real imports/tests ran in the existing
  Ascend container using the isolated migration directory.
- P2: dispatch/contracts PASS, NPU correctness pending P3. Added adapters and
  the single `_ascend._OPERATOR_EXPORTS` table; no second dispatcher. Registry
  override/fallback/dependency propagation, exact adapter parameter order,
  inference guards and internal primitive isolation tested with CPU tensors/mocks.
  Runtime + runner + registry: **52 passed**, exit 0, `/tmp/migration-p2-tests.log`.
  These results do not establish NPU mathematical correctness.
- P3-P4: not yet performed.
