# NVIDIA GDN2 native Autotune seeds: H100 / SM90 / BF16

These are **native Triton/FlagTree `.autotune.json` records**, not a new runtime
configuration format. Keep their hash directories. The native Autotuner reads
candidate timings and selects the fastest recorded candidate for a matching key.
The package contains 65 records, including eight K1 records; it contains no JIT
binaries. The records cover the Native comparison and TLE paths used by the full
12-shape BF16 Benchmark, not every possible GDN2 workload.

Treat this directory as read-only seed data. Never point `TRITON_CACHE_DIR` here.
JIT artifacts belong in the user's or CI container's writable runtime cache.

## Provenance and compatibility

- GPU: NVIDIA H100 80GB HBM3, SM90, 132 SMs.
- Python 3.12.13; Torch 2.10.0+cu128; CUDA 12.8; driver 535.183.06.
- Real FlagTree 3.6.0 build, with Triton TLE support.
- `triton._C.libtriton` SHA256:
  `4bc12af82ee43f25dec3921317030d0c181c903704124eb0458a7e3aa0e72c82`.
- GDN2 source at HEAD `5365e182c89212c9ecadc6d34e9740ae2a98268d`, canonical
  `src/flag_attn/gdn2/chunk.py` unchanged.
- Generated on 2026-10-23 from the full original BF16 Benchmark, with
  `--warmup 20 --rep 100`, `TRITON_CACHE_AUTOTUNING=1` and an empty cache.

The same GPU architecture **alone does not guarantee a hit**. Native keys also
depend on the compiler/build, kernel/dependency source, target, candidate list,
workload key and relevant compilation environment. Standard Triton 3.6.0 is not
necessarily interchangeable with this FlagTree build. Native invalidation stays
authoritative; unmatched records fall back to native search. No custom reader,
fingerprint system, Direct Launch or kernel modification is required.

## Prepare a writable cache, then run

From the repository root, using the target Python/accelerator environment:

```bash
# Activate a compatible FlagTree environment first; retain its import path.
export PYTHONPATH="src:.:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.cache/flag_attn/gdn2}"
export TRITON_CACHE_AUTOTUNING=1

python tools/prepare_gdn2_cache.py \
  --profile h100-sm90-bf16 --cache-dir "$TRITON_CACHE_DIR"

python benchmark/test_gdn2_benchmark.py \
  --full --dtype bfloat16 --warmup 20 --rep 100
```

Set the environment before starting the workload process. Preparation is
explicit, outside the operator hot path; it does not set a parent shell's
environment. Existing cache files are preserved. `TRITON_CACHE_DIR` is a process
cache root used by both native Autotune and JIT, not a GDN2-only config filename.

CI should run preparation inside its matching NVIDIA test container before the
normal tests/Benchmark. Do not apply this H100 profile indiscriminately to other
GPUs or compiler images. No CI workflow is changed by this profile.

## Generate results on another environment

Use a compatible accelerator runtime and representative workload. Create the
experiment directory outside the repository:

```bash
WORK=$(mktemp -d "${TMPDIR:-/tmp}/gdn2-local-tuning.XXXXXXXX")

python tools/tune_gdn2_cache.py \
  --cache-dir "$WORK/cache" --output-dir "$WORK/config"
```

The default workload is the unchanged full BF16 Benchmark with warmup 20 and
rep 100. The tool sets `TRITON_CACHE_AUTOTUNING=1` before importing Triton in a
child process. Existing kernel `cache_results=True` support is preserved.
Starting with a new cache performs the normal candidate search and JIT compile;
using an existing cache permits native reuse. Logs are retained outside the
repository, including failures, and JSON is exported only after Benchmark success.

To choose another workload, delegate original Benchmark flags after `--`:

```bash
python tools/tune_gdn2_cache.py \
  --cache-dir "$WORK/another-cache" --output-dir "$WORK/another-config" \
  -- --full --dtype float16 --warmup 20 --rep 100
```

Exported configs can be copied into a new writable cache without JIT artifacts:

```bash
export TRITON_CACHE_DIR="$WORK/runtime-cache"
export TRITON_CACHE_AUTOTUNING=1
python tools/prepare_gdn2_cache.py \
  --source-dir "$WORK/config" --cache-dir "$TRITON_CACHE_DIR"
python benchmark/test_gdn2_benchmark.py \
  --full --dtype bfloat16 --warmup 20 --rep 100
```

The full 12-shape Benchmark measures both Native BT64 and public TLE BT16 paths.
Its preparation results include both paths. A public-operator-only workload may
need fewer records. The tool does not reduce the 216 K1 candidates or alter timing.

## Measured cold-JIT benefit

Same H100, separate Python processes and fresh JIT directories:

| Initial state | Wall seconds | Candidate measurements | Successful JIT compilations |
|---|---:|---:|---:|
| Empty Autotune + empty JIT | 2737.395 | 2348 | 2351 |
| These 65 JSON records + empty JIT | 77.915 | 0 | 68 |

This is a 35.13x reduction in full Benchmark process time (97.15% less time),
not a kernel-execution speedup. All 65 disk lookups hit in the JSON-only run.
The one-time offline search cost is excluded from the cached run, not eliminated.
Each state was measured once; transfer to a second physical machine has not been
validated. Imports, device setup, cache reading, module loading and JIT still cost
time. Full numerical correctness is not implied by successful Benchmark timing.
