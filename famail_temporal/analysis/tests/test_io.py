import json
from famail_temporal.analysis import _io


def test_read_json_and_editor_metrics(tmp_path):
    d = tmp_path / "run"; d.mkdir()
    (d / "metrics.json").write_text(json.dumps({"deltas": {"f_causal": 0.012}}))
    assert _io.read_json(d / "metrics.json")["deltas"]["f_causal"] == 0.012
    assert _io.editor_metrics(d)["deltas"]["f_causal"] == 0.012
