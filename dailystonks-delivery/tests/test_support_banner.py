from app.delivery.runner import inject_support_banner


def test_support_banner_injected_after_header():
    html = "<html><body><header><h1>DailyStonks</h1></header><p>Report</p></body></html>"
    out = inject_support_banner(
        html=html,
        note="DB restored OK",
        support_ref="sch=123 run=2026-02-28T00:00:00Z",
    )

    assert "Support note:" in out
    assert "DB restored OK" in out
    assert "sch=123" in out
    assert 'data-ds-support-banner="1"' in out

    # ensure it went after </header> (the intended insertion point)
    assert out.index("</header>") < out.index("Support note:")


def test_support_banner_replaces_existing_single_banner():
    html = "<html><body><header>H</header><p>Report</p></body></html>"
    out1 = inject_support_banner(html, "first", "ref1")
    out2 = inject_support_banner(out1, "second", "ref2")

    assert "first" not in out2
    assert "second" in out2
    # only one banner should exist
    assert out2.count('data-ds-support-banner="1"') == 1