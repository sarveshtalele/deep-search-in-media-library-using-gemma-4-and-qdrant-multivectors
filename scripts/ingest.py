#!/usr/bin/env python
"""Thin wrapper: `python scripts/ingest.py <path> [--category X]`."""

from deepsearch.cli import app

if __name__ == "__main__":
    import sys

    sys.argv = [sys.argv[0], "ingest", *sys.argv[1:]]
    app()
