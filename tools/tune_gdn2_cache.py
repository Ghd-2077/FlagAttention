#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare native GDN2 Autotune results using the unchanged project Benchmark."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from prepare_gdn2_cache import copy_configs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARGUMENTS = ["--full", "--dtype", "bfloat16", "--warmup", "20", "--rep", "100"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Native writable cache outside the repository",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Empty external directory for JSON-only export",
    )
    parser.add_argument(
        "benchmark_args",
        nargs=argparse.REMAINDER,
        help="Optional Benchmark arguments after --",
    )
    args = parser.parse_args(argv)
    cache, output = args.cache_dir.resolve(), args.output_dir.resolve()
    project = PROJECT_ROOT.resolve()
    if any(path == project or project in path.parents for path in (cache, output)):
        raise ValueError("Cache and output directories must be outside the repository")
    if cache == output or cache in output.parents or output in cache.parents:
        raise ValueError(
            "Cache and JSON output directories must be separate, non-nested directories"
        )
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "JSON output directory must be empty; existing results are never overwritten"
        )
    arguments = args.benchmark_args
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    arguments = arguments or DEFAULT_ARGUMENTS
    cache.mkdir(parents=True, exist_ok=True)
    environment = dict(
        os.environ,
        TRITON_CACHE_DIR=str(cache),
        TRITON_CACHE_AUTOTUNING="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(project / "src"), str(project), environment.get("PYTHONPATH", "")]
    )
    command = [
        sys.executable,
        "-c",
        "import sys; from benchmark.test_gdn2_benchmark import run_benchmark; run_benchmark(sys.argv[1:])",
        *arguments,
    ]
    with tempfile.NamedTemporaryFile(
        mode="w", prefix="gdn2-offline-", suffix=".log", dir=cache.parent, delete=False
    ) as log:
        print(
            f"cache_dir={cache}\nbenchmark_args={arguments}\nlog={log.name}", flush=True
        )
        result = subprocess.run(
            command, cwd=project, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
    if result.returncode:
        print(
            f"Benchmark failed (exit {result.returncode}); log retained at {log.name}",
            file=sys.stderr,
        )
        return result.returncode
    copied, skipped = copy_configs(cache, output)
    print(f"exported={copied} skipped={skipped} output_dir={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
