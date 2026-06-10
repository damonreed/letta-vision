from letta.services.migration.enrich_pending_images import EnrichPendingReport


def test_enrich_pending_report_dry_run_lines():
    report = EnrichPendingReport(
        generated_at="2026-01-01T00:00:00Z",
        organization_id="org-test",
        pending_count=118,
        processed=0,
        succeeded=0,
        failed=0,
        dry_run=True,
    )
    text = "\n".join(report.summary_lines())
    assert "118" in text
    assert "dry run" in text.lower()
