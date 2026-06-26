import json
from famail_temporal.baselines import _bundle_index as B


def test_register_figure_upserts_by_id(tmp_path):
    mf = tmp_path / "FIGURES_MANIFEST.json"
    B.register_figure(mf, figure_id="fig5", caption="dose-response",
                      backing_files=["results/wbc/sweep.json"], producing_command="python ...")
    B.register_figure(mf, figure_id="fig5", caption="dose-response v2",
                      backing_files=["results/wbc/sweep.json"], producing_command="python ...")
    rows = json.loads(mf.read_text())
    assert len(rows) == 1 and rows[0]["caption"] == "dose-response v2"
    assert "git_sha" in rows[0]


def test_write_rerun_readme(tmp_path):
    p = tmp_path / "RERUN_README.md"
    B.write_rerun_readme(p, stages=[{"stage": "L1-v2", "out_dir": "results/l1", "gate_passed": True}])
    txt = p.read_text()
    assert "L1-v2" in txt and "results/l1" in txt
