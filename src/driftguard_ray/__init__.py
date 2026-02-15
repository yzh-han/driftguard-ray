"""Package root for driftguard-ray."""

def main() -> None:
    """CLI entrypoint exposed for script launcher."""
    from driftguard_ray.cli import main as _cli_main

    _cli_main()
