"""Research-only Triple-Barrier labeling and OOF Meta-Label filtering."""

from services.meta_labeling.meta_model import (
    META_MODEL_SCHEMA_VERSION,
    MetaLabelModel,
    OOFFold,
    generate_oof_primary_predictions,
)
from services.meta_labeling.triple_barrier import (
    TRIPLE_BARRIER_SCHEMA_VERSION,
    TripleBarrierConfig,
    apply_triple_barrier,
)
from services.meta_labeling.walkforward import evaluate_meta_oof

__all__ = [
    "META_MODEL_SCHEMA_VERSION",
    "TRIPLE_BARRIER_SCHEMA_VERSION",
    "MetaLabelModel",
    "OOFFold",
    "TripleBarrierConfig",
    "apply_triple_barrier",
    "evaluate_meta_oof",
    "generate_oof_primary_predictions",
]
