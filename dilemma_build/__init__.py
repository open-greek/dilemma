"""Source-checkout build helpers, separate from the PyPA ``build`` package.

Keep script paths and generated artifacts under ``build/`` for existing commands
and recorded provenance. Python imports use ``dilemma_build`` so installing the
packaging frontend cannot hide these helpers.
"""
from pathlib import Path

__path__[:] = [str(Path(__file__).resolve().parents[1] / "build")]
