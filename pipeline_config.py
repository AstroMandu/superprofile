"""Public configuration imports; implementation lives in pipeline.config.

Use immutable PipelineConfig / FitConfig / MaskConfig instances instead of
editing module globals. See README.md for CLI and Python examples.
"""
from pipeline.config import FitConfig, MaskConfig, PipelineConfig, parse_args, preset

__all__ = ['FitConfig', 'MaskConfig', 'PipelineConfig', 'parse_args', 'preset']
