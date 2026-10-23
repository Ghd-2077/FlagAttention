"""Contracts for native cache seeding; no GPU is required."""

import concurrent.futures
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
PREPARE = ROOT / "tools/prepare_gdn2_cache.py"
TUNE = ROOT / "tools/tune_gdn2_cache.py"


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_json(
    root, name="HASH/kernel.autotune.json", payload=b'{"configs_timings": []}'
):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_prepare_tools_exist():
    assert PREPARE.is_file(), "native cache preparation tool is missing"
    assert TUNE.is_file(), "native offline tuning tool is missing"


def test_copy_only_json_preserves_tree_and_existing_cache(tmp_path):
    module = load(PREPARE)
    source, destination = tmp_path / "source", tmp_path / "cache"
    first = native_json(source)
    native_json(source, "OTHER/second.autotune.json")
    (source / "HASH/kernel.cubin").write_bytes(b"not a config")
    native_json(destination, payload=b'{"local": "keep"}')
    assert module.copy_configs(source, destination) == (1, 1)
    assert (
        destination / "HASH/kernel.autotune.json"
    ).read_bytes() == b'{"local": "keep"}'
    assert (
        destination / "OTHER/second.autotune.json"
    ).read_bytes() == b'{"configs_timings": []}'
    assert not list(destination.rglob("*.cubin"))
    assert first.read_bytes() == b'{"configs_timings": []}'


def test_empty_source_is_an_error(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="autotune"):
        load(PREPARE).copy_configs(source, tmp_path / "cache")


def test_readonly_source_and_repeat_are_safe(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "cache"
    path = native_json(source)
    path.chmod(0o444)
    original = path.read_bytes()
    module = load(PREPARE)
    assert module.copy_configs(source, destination) == (1, 0)
    assert module.copy_configs(source, destination) == (0, 1)
    assert path.read_bytes() == original


def test_source_subdirectory_cannot_be_destination(tmp_path):
    source = tmp_path / "source"
    native_json(source)
    with pytest.raises(ValueError):
        load(PREPARE).copy_configs(source, source / "runtime")


def test_source_symlinks_are_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "alias.autotune.json").symlink_to(tmp_path / "unknown")
    with pytest.raises(ValueError, match="symlink"):
        load(PREPARE).copy_configs(source, tmp_path / "cache")


def test_concurrent_copy_is_atomic_and_non_overwriting(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "cache"
    path = native_json(source, payload=b'{"values": [1, 2, 3]}')
    module = load(PREPARE)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda _: module.copy_configs(source, destination), range(2))
        )
    assert sum(r[0] for r in results) == 1
    assert sum(r[1] for r in results) == 1
    assert (destination / path.relative_to(source)).read_bytes() == path.read_bytes()
    assert len(list(destination.rglob("*autotune.json"))) == 1
    assert not list(destination.rglob("*.tmp"))


def test_shipped_profile_has_65_jsons_and_cli_can_seed_it(tmp_path):
    source = ROOT / "src/flag_attn/gdn2/pre_tuned/h100-sm90-bf16"
    files = list(source.rglob("*.autotune.json"))
    assert len(files) == 65
    result = subprocess.run(
        [
            sys.executable,
            str(PREPARE),
            "--profile",
            "h100-sm90-bf16",
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "copied=65" in result.stdout
    for path in files:
        assert (
            path.read_bytes()
            == (tmp_path / "cache" / path.relative_to(source)).read_bytes()
        )


def test_cli_rejects_writing_to_packaged_seed(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(PREPARE),
            "--cache-dir",
            str(ROOT / "src/flag_attn/gdn2/pre_tuned/forbidden"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not (ROOT / "src/flag_attn/gdn2/pre_tuned/forbidden").exists()


def fixture_project(tmp_path, failure=False):
    project = tmp_path / "project"
    (project / "benchmark").mkdir(parents=True)
    (project / "benchmark/__init__.py").write_text("")
    code = """import json, os
from pathlib import Path
def run_benchmark(argv):
    assert os.environ['TRITON_CACHE_AUTOTUNING']=='1'
    cache=Path(os.environ['TRITON_CACHE_DIR'])
    cache.mkdir(parents=True,exist_ok=True)
    (cache/'argv.json').write_text(json.dumps(argv))
    p=cache/'HASH/kernel.autotune.json';p.parent.mkdir(exist_ok=True)
    p.write_text('{"configs_timings": []}')
    (cache/'HASH/kernel.cubin').write_bytes(b'JIT')
    raise SystemExit(7 if FAILURE else 0)
""".replace("FAILURE", repr(failure))
    (project / "benchmark/test_gdn2_benchmark.py").write_text(code)
    return project


def test_offline_launcher_sets_env_and_exports_only_json(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        module = load(TUNE)
    finally:
        sys.path.pop(0)
    monkeypatch.setattr(module, "PROJECT_ROOT", fixture_project(tmp_path))
    cache, output = tmp_path / "cache", tmp_path / "export"
    assert module.main(["--cache-dir", str(cache), "--output-dir", str(output)]) == 0
    assert json.loads((cache / "argv.json").read_text()) == [
        "--full",
        "--dtype",
        "bfloat16",
        "--warmup",
        "20",
        "--rep",
        "100",
    ]
    assert sorted(
        str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()
    ) == ["HASH/kernel.autotune.json"]


def test_offline_failure_preserves_exit_and_does_not_publish(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        module = load(TUNE)
    finally:
        sys.path.pop(0)
    monkeypatch.setattr(module, "PROJECT_ROOT", fixture_project(tmp_path, failure=True))
    cache, output = tmp_path / "cache", tmp_path / "export"
    assert module.main(["--cache-dir", str(cache), "--output-dir", str(output)]) == 7
    assert not list(output.rglob("*.autotune.json"))
    assert list(tmp_path.glob("gdn2-offline-*.log"))


def test_offline_rejects_nonempty_output_before_running(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        module = load(TUNE)
    finally:
        sys.path.pop(0)
    monkeypatch.setattr(module, "PROJECT_ROOT", fixture_project(tmp_path))
    output = tmp_path / "export"
    output.mkdir()
    (output / "keep").write_text("unchanged")
    with pytest.raises(ValueError, match="empty"):
        module.main(
            ["--cache-dir", str(tmp_path / "cache"), "--output-dir", str(output)]
        )
    assert not (tmp_path / "cache/argv.json").exists()


def test_offline_delegates_user_workload_arguments(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        module = load(TUNE)
    finally:
        sys.path.pop(0)
    monkeypatch.setattr(module, "PROJECT_ROOT", fixture_project(tmp_path))
    cache, output = tmp_path / "cache", tmp_path / "export"
    arguments = ["--shape", "1,2048,8,64,64", "--dtype", "float16"]
    assert (
        module.main(
            ["--cache-dir", str(cache), "--output-dir", str(output), "--", *arguments]
        )
        == 0
    )
    assert json.loads((cache / "argv.json").read_text()) == arguments
