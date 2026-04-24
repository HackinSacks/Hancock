import asyncio

import pytest

from inference.optimized_inference import (
    HancockInferenceEngine,
    RecursiveLearningBudget,
    RecursiveSelfImprover,
    self_improve_hancock,
)


def run(coro):
    return asyncio.run(coro)


def test_recursive_learning_budget_serializes():
    budget = RecursiveLearningBudget(
        requested_iterations=100,
        materialized_iterations=10,
        virtualized_iterations=90,
        max_materialized_iterations=10,
        virtualized=True,
    )

    assert budget.to_dict() == {
        "requested_iterations": 100,
        "materialized_iterations": 10,
        "virtualized_iterations": 90,
        "max_materialized_iterations": 10,
        "virtualized": True,
    }


def test_recursive_self_improver_runs_small_iteration_count():
    improver = RecursiveSelfImprover(HancockInferenceEngine(), max_materialized_iterations=64)

    result = run(improver.recursive_self_improve(iterations=3))

    assert result["status"] == "recursive_improvement_complete"
    assert result["iterations"] == 3
    assert result["materialized_iterations"] == 3
    assert result["virtualized_iterations"] == 0
    assert result["budget"]["virtualized"] is False
    assert len(result["history"]) == 3
    assert len(improver.improvement_history) == 3
    assert "authorized-scope only" in result["safety_verified"]


def test_recursive_self_improver_virtualizes_absurd_iteration_count():
    requested = 100_000_000_000_000
    improver = RecursiveSelfImprover(HancockInferenceEngine(), max_materialized_iterations=5)

    result = run(improver.recursive_self_improve(iterations=requested))

    assert result["iterations"] == requested
    assert result["materialized_iterations"] == 5
    assert result["virtualized_iterations"] == requested - 5
    assert result["budget"]["virtualized"] is True
    assert result["budget"]["max_materialized_iterations"] == 5
    assert len(result["history"]) == 5
    assert len(improver.improvement_history) == 5
    assert result["final_estimated_ctx"] == 1_000_000


def test_recursive_self_improver_zero_iterations_is_safe_noop():
    improver = RecursiveSelfImprover(HancockInferenceEngine())

    result = run(improver.recursive_self_improve(iterations=0))

    assert result["iterations"] == 0
    assert result["materialized_iterations"] == 0
    assert result["virtualized_iterations"] == 0
    assert result["history"] == []
    assert result["final_estimated_ctx"] == 32768


def test_recursive_self_improver_rejects_negative_iterations():
    improver = RecursiveSelfImprover(HancockInferenceEngine())

    with pytest.raises(ValueError, match="iterations must be >= 0"):
        run(improver.recursive_self_improve(iterations=-1))


def test_recursive_self_improver_rejects_invalid_materialized_budget():
    with pytest.raises(ValueError, match="max_materialized_iterations must be >= 1"):
        RecursiveSelfImprover(HancockInferenceEngine(), max_materialized_iterations=0)


def test_self_improve_hancock_default_path_preserves_contract():
    result = run(self_improve_hancock(iterations=1))

    assert result["status"] == "recursive_improvement_complete"
    assert result["iterations"] == 1
    assert result["materialized_iterations"] == 1
    assert "recommendation-only" in result["safety_verified"]
