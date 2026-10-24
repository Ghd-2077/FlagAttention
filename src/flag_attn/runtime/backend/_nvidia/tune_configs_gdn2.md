# Exact NVIDIA GDN2 workload configurations

`tune_configs_gdn2.yaml` contains one measured winner for each complete native
Autotune key, not a pool of candidates. There are eight K1 and eight K2 records
from the H100 80GB SM90 BF16 full Benchmark measurements.

Each entry has `WORKLOAD`, `META`, `num_warps`, `num_stages`, `num_ctas` and
`maxnreg`. `WORKLOAD` preserves the native key exactly: H, K, V, BT, followed by
every tensor dtype in kernel-argument order. Optional tensor/dtype changes must
match another measured entry; an unknown workload raises an error, not a search.
The native key does not contain B/T: these are eight native tuning-key results,
not twelve independently re-tuned B/T shapes or a guarantee of a global optimum.

The project's existing `TunedConfigLoader` selects one entry and creates a
single-config native Autotuner around the same JIT and outer heuristics. The
native single-config path never calls `_bench`. JIT compilation, imports, device
initialization and configuration reading still cost time. There is no GDN2 JSON
Reader, custom fingerprint, mode dispatch, Direct Launch or custom table cache.
The shared loader only caches native launchers and immutable tuning keys.

## Run the existing Benchmark

Activate a compatible NVIDIA FlagTree/TLE environment, then:

```bash
export PYTHONPATH="src:.:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.cache/flag_attn/gdn2}"
export TRITON_CACHE_AUTOTUNING=1
python benchmark/test_gdn2_benchmark.py \
  --full --dtype bfloat16 --warmup 20 --rep 100
```

No preparation/copy command is needed. Public GDN2 calls and tests automatically
use the shipped YAML. The Native comparison inside the existing Benchmark has
its own unchanged Autotuners; its timing/search is not the optimized TLE path.

These measurements cover the current H100 BF16 Benchmark API options (initial
and final state enabled). FP16, other hardware, or different optional-tensor
combinations need their own offline results. Configurations are not selected by
a custom compiler fingerprint; use the YAML validated for your runtime/device.

## Offline tuning only

```bash
WORK=$(mktemp -d "${TMPDIR:-/tmp}/gdn2-tuning.XXXXXXXX")
python tools/tune_gdn2_cache.py \
  --cache-dir "$WORK/cache" --output-file "$WORK/tune_configs_gdn2.yaml"

FLAG_ATTN_GDN2_TUNE_CONFIG="$WORK/tune_configs_gdn2.yaml" \
python benchmark/test_gdn2_benchmark.py \
  --full --dtype bfloat16 --warmup 20 --rep 100
```

The offline tool uses an explicit external empty YAML to enable the unchanged
216/3-candidate native search spaces in its child process. This is the only
intentional search path; it never overwrites the shipped file. Missing files
raise an error. A normal nonempty YAML never falls back to a candidate search.
Use a new cache for a new search; existing native disk results can be reused.
Logs and original measurements stay outside Git. The exporter preserves every
key, rejects conflicting winners for a key, and refuses to overwrite output.

Append original Benchmark flags after `--` for another dtype/shape. For API
features not exposed by the Benchmark CLI, run that workload offline with
`FLAG_ATTN_GDN2_TUNE_CONFIG` pointing to an external `{}` YAML and native disk
Autotune caching enabled. Then export its native cache with `--export-only`.
