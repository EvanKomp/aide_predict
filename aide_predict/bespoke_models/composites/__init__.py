# aide_predict/bespoke_models/composites/__init__.py
'''
* Author: Evan Komp
* Created: 2026-05-26

Composite ProteinModelWrappers — wrappers that take another wrapper as input
and add additional behavior. The composition pattern lets us keep
ProteinSequences (and the AA-identity information they carry) in scope
across multi-stage scoring pipelines without resorting to sklearn metadata
routing or DataFrame contracts.
'''
from .zscore import ZScoreRescaledScorer

__all__ = ["ZScoreRescaledScorer"]
