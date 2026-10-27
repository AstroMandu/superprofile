#!/usr/bin/env python3
"""Command-line entry point for HI superprofile experiments (Python 3.13)."""
if __package__ in (None, ''):
    # Direct execution adds pipeline/, so expose its parent for package imports.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.runtime import configure_runtime


def main(argv=None):
    configure_runtime()
    from pipeline.config import parse_args
    config = parse_args(argv)
    from pipeline.runner import run_pipeline
    return run_pipeline(config)


if __name__ == '__main__':
    main()
