import numpy as np
from famail_temporal.visualization.gradient_heatmap import app
from famail_temporal.visualization.gradient_heatmap.geometry import load_district_geometry
from famail_temporal.visualization.gradient_heatmap.loader import VizBundle


def _bundle():
    rng = np.random.default_rng(3)
    mask = np.zeros((48, 90, 24), dtype=bool)
    mask[0:10, 0:10, :] = True
    def f():
        a = np.full((48, 90, 24), np.nan, dtype=np.float32)
        a[mask] = rng.standard_normal(mask.sum()).astype(np.float32)
        return a
    return VizBundle(grad_spatial=f(), grad_causal=f(), attr_spatial=f(), attr_causal=f(),
                     pickup=rng.random((48, 90, 24)).astype(np.float32), active_mask=mask,
                     geometry=load_district_geometry(), meta={})


def _state(**over):
    s = dict(quantity="Gradient", term="F_causal", hour=8,
             alpha_spatial=0.33, alpha_causal=0.33, alpha_fidelity=0.34,
             magnitude=False, shared_scale=True, clip_pct=99.0,
             show_boundaries=True, contour_overlay=False,
             show_concentration_panel=False)
    s.update(over)
    return s


def test_build_views_returns_main_figure():
    out = app.build_views(_bundle(), _state())
    assert "main" in out and out["main"].data[0].type == "heatmap"


def test_concentration_panel_toggle():
    out = app.build_views(_bundle(), _state(show_concentration_panel=True))
    assert "concentration" in out and out["concentration"].data[0].type == "heatmap"
    out2 = app.build_views(_bundle(), _state(show_concentration_panel=False))
    assert "concentration" not in out2


def test_contour_overlay_adds_trace_to_main():
    out = app.build_views(_bundle(), _state(contour_overlay=True))
    assert any(t.type == "contour" for t in out["main"].data)


def test_magnitude_mode_makes_scale_nonnegative():
    out = app.build_views(_bundle(), _state(magnitude=True))
    assert out["main"].data[0].zmin == 0.0


def test_main_export_matches_main_figure():
    out = app.build_views(_bundle(), _state())
    exp = out["main_export"]
    hm = out["main"].data[0]
    assert exp["vmin"] == hm.zmin and exp["vmax"] == hm.zmax
    assert exp["slice2d"].shape == (48, 90)
    assert exp["cmap"] in ("RdBu_r", "viridis")


def test_app_top_level_imports_run_as_script(tmp_path):
    """`streamlit run <path>` executes app.py as a top-level script with no parent
    package, adding only the script's directory to sys.path (NOT the repo root).
    Its top-level imports must still resolve. Regression for the relative-import
    ImportError. We mimic that exactly: neutral cwd, no PYTHONPATH, only the script
    dir on sys.path, and runpy with a non-__main__ run_name so we exercise the
    module-level imports without launching the Streamlit UI.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    script_dir = app_path.parent
    code = (
        "import runpy, sys; "
        f"sys.path.insert(0, r'{script_dir}'); "
        f"runpy.run_path(r'{app_path}', run_name='probe')"
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)  # ensure the repo root is reachable ONLY via app.py's bootstrap
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(tmp_path), env=env,
        capture_output=True, text=True, timeout=120,
    )
    output = proc.stdout + proc.stderr
    assert "attempted relative import with no known parent package" not in output, output
    assert proc.returncode == 0, (
        f"app.py top-level imports failed under script execution (rc={proc.returncode}):\n{output}"
    )
