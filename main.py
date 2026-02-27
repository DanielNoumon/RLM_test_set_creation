"""
Entry point — delegates to pipeline.main.

Usage:
    python -m pipeline.main
    # or
    python main.py
"""
import runpy
runpy.run_module("pipeline.main", run_name="__main__")
