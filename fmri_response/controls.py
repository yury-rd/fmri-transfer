"""Leave-one-participant-out prediction with seven control models.

Training participants supply target responses for fitting. The held participant's
target response is used only to score predictions. A memorization oracle records
zero loss for reference and is excluded from the control comparison.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._utils import _indices, _responses, _ridge_predict


FloatArray = NDArray[np.float64]
ParticipantScope = Literal["held", "training", "population", "design"]
TemporalScope = Literal["early", "future", "mixed", "none"]


@dataclass(frozen=True)
class PredictorFeatureLineage:
    predictor: str
    source: str
    participant_scope: ParticipantScope
    temporal_scope: TemporalScope
    operator_paths: tuple[str, ...] = ()
    clinical_fields: bool = False
    fitted_with_held_participant: bool = False


@dataclass(frozen=True)
class LineageAudit:
    predictor: str
    admissible: bool
    reasons: tuple[str, ...]
    lineage: PredictorFeatureLineage


@dataclass(frozen=True)
class CompleteAtlasFold:
    held_participant: int
    atlas_loss: float
    admissible_baseline_losses: dict[str, float]
    best_admissible_baseline: str
    best_admissible_baseline_loss: float
    memorization_oracle_loss: float


@dataclass(frozen=True)
class CompleteAtlasResult:
    early_visits: tuple[int, ...]
    future_visits: tuple[int, ...]
    source_nodes: tuple[int, ...]
    target_node: int
    target_path: str
    ridge: float
    folds: tuple[CompleteAtlasFold, ...]
    atlas_loss: float
    admissible_baseline_losses: dict[str, float]
    best_admissible_baseline_loss: float
    gain_over_best_admissible_baseline: float
    memorization_oracle_loss: float
    lineage_audits: tuple[LineageAudit, ...]

    def lineage_receipt(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "predictor": audit.predictor,
                "admissible": audit.admissible,
                "reasons": list(audit.reasons),
                "lineage": asdict(audit.lineage),
            }
            for audit in self.lineage_audits
        )


def audit_predictor_feature_lineage(
    lineage: PredictorFeatureLineage,
    *,
    target_path: str,
    permit_nontransportable_oracle: bool = False,
) -> LineageAudit:
    """Check for target leakage, clinical labels, and held-participant fitting."""
    if not target_path:
        raise ValueError("target_path must be nonempty")
    reasons: list[str] = []
    if lineage.clinical_fields:
        reasons.append("clinical fields are sealed")
    if lineage.fitted_with_held_participant:
        reasons.append("predictor was fitted with the held participant")
    held_target = (
        lineage.participant_scope == "held"
        and target_path in lineage.operator_paths
    )
    if held_target:
        reasons.append("held participant target-path response entered the predictor")
    if lineage.predictor == "memorization_oracle" and not permit_nontransportable_oracle:
        reasons.append("memorization oracle is nontransportable")
    return LineageAudit(
        predictor=lineage.predictor,
        admissible=not reasons,
        reasons=tuple(reasons),
        lineage=lineage,
    )


def frozen_complete_lineages(target_path: str = "FA") -> tuple[PredictorFeatureLineage, ...]:
    """Describe the data used by each predictor."""
    return (
        PredictorFeatureLineage(
            "atlas", "future base FC plus early operator responses", "held", "mixed",
            ("BASE", "A", "F", "AF"),
        ),
        PredictorFeatureLineage("universal", "training population target response", "training", "future", (target_path,)),
        PredictorFeatureLineage("additive", "early A plus F response", "held", "early", ("A", "F")),
        PredictorFeatureLineage("seen_path", "early AF carry-forward", "held", "early", ("AF",)),
        PredictorFeatureLineage("static_identity", "early base FC", "held", "early", ("BASE",)),
        PredictorFeatureLineage("future_base", "future base FC", "held", "future", ("BASE",)),
        PredictorFeatureLineage("reliability", "early repeat reliability summary", "held", "early", ("BASE",)),
        PredictorFeatureLineage("design_summary", "operator matrix and support summaries", "design", "none", ()),
        PredictorFeatureLineage("memorization_oracle", "held future target response", "held", "future", (target_path,)),
    )


def _participant_features(values: ArrayLike, participants: int, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] != participants or array.shape[1] < 1:
        raise ValueError(f"{name} must be participant-by-feature")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _shared_coordinate_predict(
    predictors: FloatArray,
    targets: FloatArray,
    query: FloatArray,
    ridge: float,
) -> FloatArray:
    """Fit coefficients shared across connectivity edges.

    ``predictors`` is training-participant by feature-coordinate by covariate.
    Pooling coordinates estimates one coefficient per response source.
    """
    if predictors.ndim != 3 or targets.shape != predictors.shape[:2]:
        raise ValueError("structured predictors and targets have incompatible topology")
    if query.shape != predictors.shape[1:]:
        raise ValueError("structured query has incompatible topology")
    x = predictors.reshape(-1, predictors.shape[-1])
    y = targets.reshape(-1)
    coefficients = np.linalg.solve(
        x.T @ x + ridge * np.eye(x.shape[1]),
        x.T @ y,
    )
    return query @ coefficients


def qualify_complete_participant_operator_atlas(
    responses: ArrayLike,
    *,
    early_visits: tuple[int, ...],
    future_visits: tuple[int, ...],
    base_node: int,
    source_nodes: tuple[int, ...],
    target_node: int,
    additive_nodes: tuple[int, int],
    seen_path_node: int,
    reliability_features: ArrayLike,
    design_summary_features: ArrayLike,
    ridge: float,
    target_path: str = "FA",
) -> CompleteAtlasResult:
    """Compare the response model with all seven controls under participant holdout."""
    array = _responses(responses)
    participants, visits, nodes, _ = array.shape
    if participants < 5:
        raise ValueError("at least five participants are required for participant holdout")
    early = _indices(early_visits, limit=visits, name="early_visits")
    future = _indices(future_visits, limit=visits, name="future_visits")
    if set(early) & set(future):
        raise ValueError("early_visits and future_visits must be disjoint")
    source = _indices(source_nodes, limit=nodes, name="source_nodes")
    if len(source) < 2 or target_node in source:
        raise ValueError("source nodes must exclude the unseen target and contain at least two nodes")
    node_values = (base_node, target_node, seen_path_node, *additive_nodes)
    if any(not isinstance(value, (int, np.integer)) or not 0 <= int(value) < nodes for value in node_values):
        raise ValueError("all declared nodes must be valid")
    if not np.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("ridge must be finite and positive")

    reliability = _participant_features(reliability_features, participants, "reliability_features")
    design = _participant_features(design_summary_features, participants, "design_summary_features")
    lineages = frozen_complete_lineages(target_path)
    audits = tuple(
        audit_predictor_feature_lineage(lineage, target_path=target_path)
        for lineage in lineages
    )
    admissible_names = {audit.predictor for audit in audits if audit.admissible}
    required = {
        "atlas", "universal", "additive", "seen_path", "static_identity",
        "future_base", "reliability", "design_summary",
    }
    if admissible_names != required:
        raise RuntimeError("frozen lineage audit did not yield the exact admissible predictor set")
    oracle_audit = next(audit for audit in audits if audit.predictor == "memorization_oracle")
    if oracle_audit.admissible:
        raise RuntimeError("memorization oracle must remain inadmissible")

    base_early = array[:, early, int(base_node)].mean(axis=1)
    base_future = array[:, future, int(base_node)].mean(axis=1)
    source_responses = [
        array[:, early, node].mean(axis=1) - base_early
        for node in source
    ]
    structured_signature = np.stack([base_future, *source_responses], axis=-1)
    truth = array[:, future, int(target_node)].mean(axis=1) - base_future
    direct = {
        "additive": sum(
            array[:, early, int(node)].mean(axis=1) - base_early
            for node in additive_nodes
        ),
        "seen_path": array[:, early, int(seen_path_node)].mean(axis=1) - base_early,
    }
    mapped = {
        "static_identity": base_early,
        "future_base": base_future,
        "reliability": reliability,
        "design_summary": design,
    }

    folds: list[CompleteAtlasFold] = []
    for held in range(participants):
        training = np.arange(participants) != held
        predictions = {
            "universal": truth[training].mean(axis=0),
            "additive": direct["additive"][held],
            "seen_path": direct["seen_path"][held],
        }
        for name, feature in mapped.items():
            predictions[name] = _ridge_predict(
                feature[training], truth[training], feature[held], float(ridge)
            )
        atlas_prediction = _shared_coordinate_predict(
            structured_signature[training], truth[training],
            structured_signature[held], float(ridge)
        )
        losses = {
            name: float(np.mean(np.square(truth[held] - prediction)))
            for name, prediction in predictions.items()
        }
        best_name = min(losses, key=losses.__getitem__)
        folds.append(CompleteAtlasFold(
            held_participant=held,
            atlas_loss=float(np.mean(np.square(truth[held] - atlas_prediction))),
            admissible_baseline_losses=losses,
            best_admissible_baseline=best_name,
            best_admissible_baseline_loss=losses[best_name],
            memorization_oracle_loss=0.0,
        ))

    baseline_names = tuple(sorted(folds[0].admissible_baseline_losses))
    aggregate_baselines = {
        name: float(np.mean([fold.admissible_baseline_losses[name] for fold in folds]))
        for name in baseline_names
    }
    atlas_loss = float(np.mean([fold.atlas_loss for fold in folds]))
    best_loss = float(np.mean([fold.best_admissible_baseline_loss for fold in folds]))
    scale = max(aggregate_baselines["universal"], np.finfo(np.float64).tiny)
    return CompleteAtlasResult(
        early_visits=early,
        future_visits=future,
        source_nodes=source,
        target_node=int(target_node),
        target_path=target_path,
        ridge=float(ridge),
        folds=tuple(folds),
        atlas_loss=atlas_loss,
        admissible_baseline_losses=aggregate_baselines,
        best_admissible_baseline_loss=best_loss,
        gain_over_best_admissible_baseline=float((best_loss - atlas_loss) / scale),
        memorization_oracle_loss=0.0,
        lineage_audits=audits,
    )
