# Ascend forward acceptance — 2026-10-17

## Scope and audit

- Baseline: `75e0c09a881d589f4dfc939ca757638b72008428`.
- Locked upstream: `17a2f54463e61f1906f3a684dc1da7413938fffd`, already an ancestor.
- Work branch: `fix/ascend-forward-acceptance-20261017`.
- Actual GitHub `npu-ghd` was checked with `ls-remote` and matches the baseline.
  The local `origin/npu-ghd` ref is stale (`7c88f51`), because this shared Git
  store has no configured origin fetch refspec; `fetch origin npu-ghd` updates
  FETCH_HEAD, not that tracking ref. It is not a remote divergence.
- Worktree was clean before creating this branch. Original local/server
  `npu-ghd` refs are not changed; no merge, push, PR or global Git changes.
- Server original checkout is still `cc675a5`; validation will use a new,
  isolated checkout transferred from this work branch.
- Only forward/inference is in scope. Backward is OUT OF SCOPE, never PASS.
- Raw device logs and benchmark data stay outside tracked files.
- Source plan: [ascend-post-sync-plan.md](ascend-post-sync-plan.md).
- Engineering references: [PR #63](https://github.com/flagos-ai/FlagAttention/pull/63)
  and [PR #91](https://github.com/flagos-ai/FlagAttention/pull/91); actual source
  contracts govern adapters, registry and tests.

## Environment (fresh detection)

Container `ghd-sageattn` is running. Its existing device mask is **4,5**;
it is not overwritten. `npu-smi` reports Ascend910B4-1. Tests select logical
device 0 within that existing mask. Fresh imports report Python 3.11.15,
PyTorch 2.9.0+cpu, torch_npu 2.9.0.post2, Triton 3.5.1; NPU available=True,
device_count=2 and logical device 0 is Ascend910B4-1. CANN environment is loaded
from the existing toolkit without replacing its compiler PATH.

## Stage gates

| Stage | State | Evidence |
| --- | --- | --- |
| P0 audit | PASS | Clean baseline, isolated branch, fresh NPU detection |
| P1 GLA forward | PASS | 17 passed, exit 0, real 910B |
| P2 NSA forward | PASS | Final combined run: 39 passed, exit 0 |
| P3 strict runner | PASS | 30 CPU/CLI tests passed locally and in container |
| P4 NPU regression | PASS | 83 CPU/contract tests, 54 NPU forward/guard cases; strict runner exit 0 |
| P5 performance/CI | NOT RUN | Historical timings do not count |

## Fork ownership review

`git diff --name-only 17a2f54...75e0c09` identifies existing fork changes in
Ascend implementations/exports, configuration inventory, migration docs,
Ascend tests and the runner. P1 will add an Ascend YAML and split its own GLA
test; P2 will change only a new adapter/exports/tests; P3 only runner/tests.
Existing JIT source is outside the implementation boundary.

## P1 — GLA configuration and separate forward gate

The new CPU-capable configuration test was executed before adding YAML:
**1 failed**, exit 1, because the loader returned `[]` (log `p1-red.log`).
The original projects and Git history contain no Ascend YAML for this kernel.
The single candidate is copied from the locked NVIDIA example: BK=32, BV=32,
num_warps=4, num_stages=3. It is a correctness restoration candidate, not a
performance tuning result. No backward configuration is invented.

`test_gla_forward` covers lengths 17/32, FP16/BF16, both state layouts, with
and without a nonzero initial state, using the existing 0.01 relative-RMS
threshold and an independent CPU recurrence. Existing gradient assertions
remain in `test_gla_backward`, outside the forward gate. The inventory node
now points to the new forward test; official cross-vendor tests remain intact.

Command (from the isolated checkout, existing CANN environment and device mask):

```sh
python -m pytest -q \
  tests/flag_attn/test_ascend_attention.py::test_gla_forward \
  tests/flag_attn/test_ascend_registry.py::test_ascend_forward_state_config_is_loadable \
  --disable-warnings
```

Hardware result: **17 passed, 2 warnings**, exit **0**, 1682.24 seconds on
910B (16 numerical cases plus the configuration check). The lengthy first run
compiled the existing autotune candidates unchanged. The candidate is now
validated for this matrix, not claimed optimal. Backward: OUT OF SCOPE.
Logs: `/tmp/flagattention-ascend-acceptance-20261017/p1-green.log`.
Pre-commit on the new YAML passed all applicable hooks; unmatched hooks skipped.
Standalone Ruff on both modified test files passed.
Static `compileall` and `git diff --check` pass. AST comparison with the baseline
covers **45 JIT functions** (43 Ascend plus 2 shared state functions): zero
changed function ASTs. CPU/mock checks are not hardware correctness evidence.

## P2 — NSA forward-only public contract

P1 commit: `96977eb`. Added only a lazy `nsa/api.py` adapter and the single
compression registry key, preserving exact public parameter names/defaults.
Training with any q/k/v requiring gradients is explicitly rejected while
autograd is enabled; no detach, no no-grad decorator, no silent fallback.
Explicit no-grad/inference contexts work. Full `parallel_nsa` is not registered.

Tests-first: 8 newly selected NSA contract cases failed before implementation
(exit 1, `p2-red.log`). The first combined implementation run had **12 failed,
27 passed, 2 teardown errors**: the new test incorrectly monkeypatched a frozen
device dataclass field, preventing mock cleanup and returning mock strings in
subsequent compression tests. This is a test-isolation bug, not NPU precision
evidence. Fixed the test by replacing the lightweight runtime device object
within monkeypatch instead; no production device metadata or kernel changes.
The failed log `p2-green.log` is retained, not counted as a pass.

Fresh isolated reruns: registry **23 passed**, exit 0 (`p2-contract-rerun.log`);
NPU **16 passed**, exit 0 (`p2-forward-rerun.log`). The latter comprises 4 new
public alias/facade tests, 6 selected-only internal tests, and 6 original
compression forward/no-grad tests. FP16/BF16 compression covers T64/128 and
GQA, output plus natural-log LSE; selected-only covers grouped/masked branches,
negative/future indices, empty selections, T32/33 and packed sequence lengths.
The existing numerical tolerances and backward tests are unchanged.

```sh
python -m pytest -q tests/flag_attn/test_ascend_registry.py --disable-warnings
python -m pytest -q tests/flag_attn/test_ascend_nsa_forward.py \
  tests/flag_attn/test_parallel_nsa_compression.py::test_parallel_nsa_compression_fwd_accuracy \
  tests/flag_attn/test_parallel_nsa_compression.py::test_parallel_nsa_compression_no_grad \
  --disable-warnings
```

Final combined run confirms mock isolation: **39 passed**, exit **0**, 13.04s
(`p2-final.log`). Applicable pre-commit hooks passed (`p2-hooks.log`, exit 0).
Backward remains OUT OF SCOPE; selected-only is explicitly INTERNAL API.

## P3 — Opt-in strict result gate

P2 commit: `f7059e4`. Added `--strict-results`, evaluated after writing the
unchanged summary JSON. It rejects missing/invalid worker summaries, crashed
workers, missing requested operators/phases, non-Passed accuracy, nonzero
subprocess exits, zero executed passes and non-Passed performance. Explicit
`--skip-benchmarks` permits a genuinely Skipped performance entry, never a
performance PASS. Partial Skip plus positive Pass is accepted with counts
retained. Default non-strict exit behavior is preserved and regression-tested.

Tests-first CLI run: **17 failed, 1 passed**, exit 1 (unknown strict flag before
implementation). Final local suite: **30 passed**, exit 0; same suite in the
container: **30 passed**, exit 0 (`p3-regression.log`). Includes 18 real
subprocess exit-code scenarios, Ctrl-C cleanup and actual pytest node collection
from both worker cwd layouts while an unrelated optional-backend file would
raise if accidentally imported. Pinned Ruff 0.1.14 passes runner and tests.

```sh
# CPU-only runner tests; confcutdir avoids unrelated hardware conftest locally.
python -m pytest -q --confcutdir=tests/flag_attn \
  tests/flag_attn/test_test_runner.py --disable-warnings
```

No NPU mathematical correctness is inferred from these CPU/mock tests.

## P4 — Fresh integrated forward gate

Production source at P3 commit `1a097ec`, plus only the P4 test additions,
was transferred to the isolated container checkout. Before tests, assertions
verified vendor `ascend`, device `npu`, actual NPU availability and that the
imported package resides in the current checkout (not an installed old copy).

The first integrated command passed 83 CPU/contract and 53 forward/guard tests,
and the strict runner, but **exited 2** at Linux `git diff --check`: source
transport inherited the Windows host's system `core.autocrlf=true`. Committed
Git blobs were confirmed LF. Even `git archive` applies working-tree conversion,
so the first archive retry still failed. Fixed transport with command-local
`git -c core.autocrlf=false -c core.eol=lf archive`; no global config changes,
kernel rewriting or ignored whitespace checks. Failed logs are retained as
`p4-acceptance.log` and `p4-static.log`; successful static retry is
`p4-static-lf.log`, exit 0.

Added one explicit Sage top-level/namespace forward alias test with uniform
attention output/LSE and invalid quantization block size; its separate run
passed 1 case. Final complete rerun (`p4-final.log`) exited **0**:

| Check | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Runtime / registry / runner CPU and mock contracts | 83 | 0 | 0 |
| Explicit 910B forward and error-contract matrix | 54 | 0 | 0 |
| Strict inventory runner: GDN2 accuracy | 9 | 0 | 83 |
| Strict inventory runner: Sage accuracy | 7 | 0 | 0 |

Runner tests repeat some matrix cases; do not sum these rows as unique tests.
Its performance status is **SKIPPED**, not PASS. Those 83 GDN2 skips are CUDA
native/TLE resource comparisons, not skipped Ascend recurrence cases.

| Forward matrix component | Cases | Coverage boundary |
| --- | ---: | --- |
| GDN2 | 9 | FP16/BF16, T16/33, nonzero initial/final state, both layouts, invalid chunk size |
| GLA | 16 | FP16/BF16, T17/32, both layouts, zero/nonzero initial state, output/final state |
| KDA | 3 | BF16 T16/33 recurrence/final state, public rejection of precomputed indices |
| MiniMax | 2 | Prefill + score/top-k composite reference; invalid stage |
| Sage | 8 | HND/NHD, GQA, bool/additive mask, tails, quantization/LSE, aliases, error guards |
| NSA compression | 10 | Public facade/namespace, FP16/BF16 T64/128, output/LSE, no-grad |
| NSA selected | 6 | INTERNAL API only; grouped/masked, packed/boundary/empty/invalid indices, output/LSE |

All backward: **OUT OF SCOPE**, original code/tests retained. MiniMax decode,
full public NSA and independent all-shape primitive validation: **NOT RUN**;
no claim of independent optimized Ascend decode/full NSA support.

Reproduction: use the explicit nodes in the execution plan, replacing the
Sage file with its three `test_forward_*` nodes, and adding:

```sh
python -m pytest -q \
  tests/flag_attn/test_ascend_nsa_forward.py::test_ascend_nsa_compression_public_alias \
  tests/flag_attn/test_ascend_nsa_forward.py::test_ascend_nsa_selected_forward \
  tests/flag_attn/test_ascend_attention.py::test_kda_public_rejects_precomputed_chunk_indices \
  tests/flag_attn/test_ascend_attention.py::test_minimax_rejects_unknown_stage \
  tests/flag_attn/test_ascend_attention.py::test_sage_public_forward_alias_and_quantization_error \
  --disable-warnings
python tools/run_tests.py --ops chunk_gdn2,sage_attention \
  --skip-benchmarks --strict-results --gpus 4 --dump-output --color never \
  --output-dir /tmp/flagattention-ascend-acceptance-20261017/runner
```

The existing runner interprets `--gpus` as child physical masks; `--gpus 4`
keeps its workers on physical 4, a subset of the existing parent mask `4,5`.
Direct tests use logical device 0 in that unchanged parent mask. Global mask
and runner device-selection implementation are not changed.

Final `compileall`, `git diff --check`, pinned Ruff on changed Python files,
and applicable source pre-commit hooks pass (`p4-hooks.log`); unmatched hooks
are Skipped. AST audit still reports **45 unchanged JIT functions**.
