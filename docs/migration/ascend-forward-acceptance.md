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
| P1 GLA forward | NOT RUN | No current test results |
| P2 NSA forward | NOT RUN | No current test results |
| P3 strict runner | NOT RUN | No current test results |
| P4 NPU regression | NOT RUN | Historical logs do not count |
| P5 performance/CI | NOT RUN | Historical timings do not count |

## Fork ownership review

`git diff --name-only 17a2f54...75e0c09` identifies existing fork changes in
Ascend implementations/exports, configuration inventory, migration docs,
Ascend tests and the runner. P1 will add an Ascend YAML and split its own GLA
test; P2 will change only a new adapter/exports/tests; P3 only runner/tests.
Existing JIT source is outside the implementation boundary.
