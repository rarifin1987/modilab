"""MoDiLab: a bilingual virtual laboratory for teaching molecular dynamics."""

from .core import Parameters, Result, compute_forces, dimer, run

__version__ = "0.3.1"
__all__ = ["Parameters", "Result", "compute_forces", "dimer", "run"]
