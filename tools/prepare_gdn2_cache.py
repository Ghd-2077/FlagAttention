#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Copy native Autotune JSON seeds without importing the accelerator runtime."""

import argparse
import json
import os
from pathlib import Path
import tempfile

PRE_TUNED_ROOT = Path(__file__).resolve().parents[1] / "src/flag_attn/gdn2/pre_tuned"


def copy_configs(source: Path, destination: Path) -> tuple[int, int]:
    """Preserve relative paths/bytes; atomically add missing files, never replace."""
    source, destination = source.resolve(), destination.resolve()
    if destination == source or source in destination.parents:
        raise ValueError(
            "The writable cache must be outside the source configuration directory"
        )
    paths = list(source.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Configuration source must not contain symlinks")
    files = sorted(
        path
        for path in paths
        if path.is_file() and path.name.endswith(".autotune.json")
    )
    if not files:
        raise ValueError(f"No native .autotune.json files found in {source}")
    copied = skipped = 0
    for path in files:
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.parent.resolve().is_relative_to(destination):
            raise ValueError(
                "Cache subdirectory resolves outside the requested destination"
            )
        if target.exists() or target.is_symlink():
            skipped += 1
            continue
        payload = path.read_bytes()
        json.loads(payload)  # Validate JSON syntax, not a new configuration schema.
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".tmp") as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            try:
                os.link(
                    temporary.name, target
                )  # Same filesystem; atomic and no overwrite.
                copied += 1
            except FileExistsError:
                skipped += 1
    return copied, skipped


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--profile", help="Bundled profile (default: h100-sm90-bf16)")
    source.add_argument(
        "--source-dir", type=Path, help="Native JSON exported on another environment"
    )
    parser.add_argument(
        "--cache-dir", type=Path, required=True, help="Writable TRITON_CACHE_DIR"
    )
    args = parser.parse_args(argv)
    try:
        if args.source_dir is not None:
            directory = args.source_dir
        else:
            profile = args.profile or "h100-sm90-bf16"
            directory = PRE_TUNED_ROOT / profile
            if (
                directory.resolve().parent != PRE_TUNED_ROOT.resolve()
                or not directory.is_dir()
            ):
                raise ValueError(f"Unknown bundled profile: {profile}")
        destination = args.cache_dir.resolve()
        if (
            destination == PRE_TUNED_ROOT.resolve()
            or PRE_TUNED_ROOT.resolve() in destination.parents
        ):
            raise ValueError(
                "Do not use the packaged pre_tuned directory as a writable cache"
            )
        copied, skipped = copy_configs(directory, destination)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"copied={copied} skipped={skipped} cache_dir={destination}")
    print(
        "Set TRITON_CACHE_DIR and TRITON_CACHE_AUTOTUNING=1 before starting the workload process."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
