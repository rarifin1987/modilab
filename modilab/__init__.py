"""MoDiLab: a bilingual virtual laboratory for teaching molecular dynamics."""

from .core import Parameters, Result, compute_forces, run

__version__ = "0.2.0"
__all__ = ["Parameters", "Result", "compute_forces", "run"]
