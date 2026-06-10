"""Historic embedding uplift management jobs (FR v0.6.0 GA)."""

__all__ = ["build_inventory_report", "run_conversion_dry_run"]


def __getattr__(name: str):
    if name == "build_inventory_report":
        from letta.services.migration.uplift_inventory import build_inventory_report

        return build_inventory_report
    if name == "run_conversion_dry_run":
        from letta.services.migration.image_base64_conversion import run_conversion_dry_run

        return run_conversion_dry_run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
