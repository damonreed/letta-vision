"""Historic embedding uplift management jobs (FR v0.6.0 GA)."""

__all__ = ["build_inventory_report", "run_conversion_dry_run", "run_conversion_live", "run_enrich_pending_live"]


def __getattr__(name: str):
    if name == "build_inventory_report":
        from letta.services.migration.uplift_inventory import build_inventory_report

        return build_inventory_report
    if name == "run_conversion_dry_run":
        from letta.services.migration.image_base64_conversion import run_conversion_dry_run

        return run_conversion_dry_run
    if name == "run_conversion_live":
        from letta.services.migration.image_base64_conversion import run_conversion_live

        return run_conversion_live
    if name == "run_enrich_pending_live":
        from letta.services.migration.enrich_pending_images import run_enrich_pending_live

        return run_enrich_pending_live
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
