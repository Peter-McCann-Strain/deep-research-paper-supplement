"""Judge-result scoring helpers for current public API reruns."""

from __future__ import annotations

from typing import Any

from deep_research.repro_compare import _coerce_float


def _criterion_metadata(query_record: dict[str, Any], fallback_criteria: list[str]) -> list[dict[str, Any]]:
    rubric = query_record.get("rubric")
    if isinstance(rubric, dict) and isinstance(rubric.get("criteria"), list):
        metadata = []
        for index, item in enumerate(rubric["criteria"]):
            if isinstance(item, dict):
                text = item.get("text") or item.get("criterion") or item.get("description")
                metadata.append(
                    {
                        "criterion_index": index,
                        "text": str(text or ""),
                        "dimension": str(item.get("dimension") or "overall"),
                        "weight": float(item.get("weight") or 1.0),
                    }
                )
            else:
                metadata.append(
                    {
                        "criterion_index": index,
                        "text": str(item),
                        "dimension": "overall",
                        "weight": 1.0,
                    }
                )
        return metadata
    return [
        {
            "criterion_index": index,
            "text": text,
            "dimension": "overall",
            "weight": 1.0,
        }
        for index, text in enumerate(fallback_criteria)
    ]


def _dimension_weight_map(query_record: dict[str, Any]) -> dict[str, float]:
    rubric = query_record.get("rubric")
    if not isinstance(rubric, dict) or not isinstance(rubric.get("dimension_weights"), dict):
        return {}
    weights = {}
    for dimension, value in rubric["dimension_weights"].items():
        number = _coerce_float(value)
        if number is not None and number > 0:
            weights[str(dimension)] = number
    return weights


def _score_provider_judgment(
    provider_result: dict[str, Any],
    criteria_metadata: list[dict[str, Any]],
    dimension_weights: dict[str, float],
) -> dict[str, Any] | None:
    if provider_result.get("status") != "success":
        return None
    parsed = provider_result.get("parsed")
    evaluations = parsed.get("evaluations") if isinstance(parsed, dict) else None
    if not isinstance(evaluations, list):
        return None
    verdicts = {item.get("criterion_index"): item.get("verdict") for item in evaluations}
    total_weight = 0.0
    satisfied_weight = 0.0
    dimensions: dict[str, dict[str, float]] = {}
    for criterion in criteria_metadata:
        index = int(criterion["criterion_index"])
        weight = abs(float(criterion.get("weight") or 1.0))
        dimension = str(criterion.get("dimension") or "overall")
        bucket = dimensions.setdefault(dimension, {"satisfied_weight": 0.0, "total_weight": 0.0})
        total_weight += weight
        bucket["total_weight"] += weight
        if verdicts.get(index) == "SATISFIED":
            satisfied_weight += weight
            bucket["satisfied_weight"] += weight
    dimension_scores = {
        dimension: values["satisfied_weight"] / values["total_weight"]
        for dimension, values in sorted(dimensions.items())
        if values["total_weight"]
    }
    criterion_weighted_score = satisfied_weight / total_weight if total_weight else 0.0
    weighted_dimensions = {
        dimension: weight
        for dimension, weight in dimension_weights.items()
        if dimension in dimension_scores and weight > 0
    }
    if weighted_dimensions:
        total_dimension_weight = sum(weighted_dimensions.values())
        overall_score = sum(
            dimension_scores[dimension] * weight
            for dimension, weight in weighted_dimensions.items()
        ) / total_dimension_weight
        scoring_method = "dimension_weighted"
    else:
        overall_score = criterion_weighted_score
        scoring_method = "criterion_weighted"

    return {
        "provider": provider_result.get("provider"),
        "provider_mode": provider_result.get("provider_mode"),
        "model": provider_result.get("model"),
        "call_model_or_deployment": provider_result.get("call_model_or_deployment"),
        "criteria_count": len(criteria_metadata),
        "evaluations_count": len(evaluations),
        "criterion_coverage": round(len(evaluations) / len(criteria_metadata), 4)
        if criteria_metadata
        else 0.0,
        "weighted_score": round(overall_score, 4),
        "criterion_weighted_score": round(criterion_weighted_score, 4),
        "scoring_method": scoring_method,
        "dimension_weights_applied": {
            key: round(value, 4) for key, value in sorted(weighted_dimensions.items())
        },
        "dimension_scores": {
            key: round(value, 4) for key, value in sorted(dimension_scores.items())
        },
    }


def _score_judge_report(
    judge_payload: dict[str, Any],
    query_record: dict[str, Any],
    fallback_criteria: list[str],
) -> dict[str, Any]:
    criteria_metadata = _criterion_metadata(query_record, fallback_criteria)
    dimension_weights = _dimension_weight_map(query_record)
    provider_scores = [
        score
        for result in judge_payload.get("results", [])
        if (score := _score_provider_judgment(result, criteria_metadata, dimension_weights))
        is not None
    ]
    mean_panel_score = (
        sum(score["weighted_score"] for score in provider_scores) / len(provider_scores)
        if provider_scores
        else 0.0
    )
    return {
        "source_query_id": str(query_record.get("id") or ""),
        "criteria_count": len(criteria_metadata),
        "dimension_weights": {key: round(value, 4) for key, value in sorted(dimension_weights.items())},
        "successful_provider_scores": len(provider_scores),
        "mean_panel_score": round(mean_panel_score, 4),
        "mean_3judge_current_api": round(mean_panel_score, 4) if len(provider_scores) == 3 else None,
        "provider_scores": provider_scores,
    }
