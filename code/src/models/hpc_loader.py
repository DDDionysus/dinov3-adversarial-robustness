"""Compatibility import for legacy HPC experiments.

Use the same model loader on workstations and clusters so checkpoint
resolution and parameter-freezing behaviour cannot silently diverge.
"""

from src.models.loader import load_model

__all__ = ["load_model"]
