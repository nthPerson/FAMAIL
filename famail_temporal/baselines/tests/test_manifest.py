import json
from pathlib import Path
from famail_temporal.baselines import _manifest


def test_write_run_manifest_captures_core_fields(tmp_path):
    out = _manifest.write_run_manifest(
        tmp_path, argv=["prog", "--seeds", "0,1"], seeds=[0, 1],
        edit_dir="results/edit_X", extra={"discriminator_sha256": "abc", "gate_matched": 0.84},
        now="2026-06-26T00:00:00Z",
    )
    assert out == tmp_path / "manifest.json"
    m = json.loads(out.read_text())
    assert m["argv"] == ["prog", "--seeds", "0,1"]
    assert m["seeds"] == [0, 1]
    assert m["edit_dir"] == "results/edit_X"
    assert m["timestamp_utc"] == "2026-06-26T00:00:00Z"
    assert m["discriminator_sha256"] == "abc"        # extra merged
    assert "git_sha" in m and "git_dirty" in m
    assert "env" in m and "python" in m["env"]
    assert "hostname" in m


def test_capture_env_never_raises_and_has_keys():
    env = _manifest.capture_env()
    for k in ("python", "torch", "cuda", "gpu_name", "numpy", "pandas",
              "cudnn_deterministic", "cudnn_benchmark"):
        assert k in env  # value may be "unknown"
