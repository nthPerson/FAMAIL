from famail_temporal.analysis import experiment_cleanup_delta as E


def test_l1v2_summary_reads_scalar_sources():
    m = {"sources": {"raw": {"f_causal": 0.807, "f_spatial": 0.103},
                     "edited": {"f_causal": 0.819, "f_spatial": 0.102}}}
    s = E.l1v2_summary(m)
    assert s["edited"]["f_causal"] == 0.819 and s["raw"]["f_spatial"] == 0.103
    assert s["bc"]["f_causal"] is None  # missing source -> None, no crash


def test_l2_summary_reads_mean_and_paired():
    m = {"per_source": {"edited": {"f_causal": {"mean": 0.807}, "f_spatial": {"mean": 0.105}}},
         "paired": {"f_causal": {"raw": {"mean": -0.0009, "wilcoxon_p": 0.44}}}}
    s = E.l2_summary(m)
    assert s["per_source"]["edited"]["f_causal"] == 0.807
    assert s["edited_vs_raw"]["delta"] == -0.0009 and s["edited_vs_raw"]["wilcoxon_p"] == 0.44


def test_wbc_summary_reads_arms_and_paired():
    sweep = {"per_arm": {"edited_w30": {"f_causal": {"mean": 0.834}}},
             "paired_vs_raw": {"f_causal": {"edited_w30": {"mean": 0.026, "wilcoxon_p": 0.03125}}}}
    s = E.wbc_summary(sweep)
    assert s["per_arm"]["edited_w30"]["f_causal"] == 0.834
    assert s["paired_vs_raw"]["edited_w30"]["delta"] == 0.026


def test_build_comparison_and_markdown_smoke():
    dirty = {"l1v2": {"sources": {"edited": {"f_causal": 0.818, "f_spatial": 0.082}}}}
    clean = {"l1v2": {"sources": {"edited": {"f_causal": 0.819, "f_spatial": 0.103}}}}
    cmp = E.build_comparison(dirty, clean)
    assert cmp["l1v2"]["dirty"]["edited"]["f_causal"] == 0.818
    md = E.render_markdown(cmp)
    assert "L1-v2 data quality" in md and "0.8180" in md
