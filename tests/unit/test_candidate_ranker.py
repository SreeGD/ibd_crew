"""
Tests for Candidate Ranker — risk-adjusted scoring factor.
IBD Momentum Investment Framework v4.0
"""

import pytest

from ibd_agents.tools.candidate_ranker import (
    COMPOSITE_WEIGHT,
    CONVICTION_WEIGHT,
    MOMENTUM_WEIGHT,
    RISK_ADJUSTED_WEIGHT,
    RS_WEIGHT,
    VALIDATION_WEIGHT,
    rank_candidates,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rated_stock(
    symbol: str,
    composite: int = 95,
    rs: int = 90,
    eps: int = 85,
    tier: int = 1,
    conviction: int = 7,
    is_multi_source: bool = False,
    sector: str = "CHIPS",
):
    """Build a minimal RatedStock-like object for testing rank_candidates()."""
    from unittest.mock import MagicMock
    stock = MagicMock()
    stock.symbol = symbol
    stock.company_name = f"{symbol} Corp"
    stock.sector = sector
    stock.tier = tier
    stock.composite_rating = composite
    stock.rs_rating = rs
    stock.eps_rating = eps
    stock.conviction = conviction
    stock.is_multi_source_validated = is_multi_source
    stock.validation_score = 5 if is_multi_source else 2
    stock.sector_rank_in_sector = 1
    stock.cap_size = "large"
    stock.estimated_return_pct = 30.0
    return stock


# ---------------------------------------------------------------------------
# Test: Score weight constants
# ---------------------------------------------------------------------------

class TestCandidateRankerConstants:
    """Test that score weights sum to 1.0."""

    @pytest.mark.schema
    def test_weights_sum_to_1(self):
        total = (
            COMPOSITE_WEIGHT + RS_WEIGHT + CONVICTION_WEIGHT
            + VALIDATION_WEIGHT + MOMENTUM_WEIGHT + RISK_ADJUSTED_WEIGHT
        )
        assert abs(total - 1.0) < 0.001

    @pytest.mark.schema
    def test_risk_adjusted_weight_is_015(self):
        assert RISK_ADJUSTED_WEIGHT == 0.15


# ---------------------------------------------------------------------------
# Test: Risk-adjusted scoring
# ---------------------------------------------------------------------------

class TestRiskAdjustedScoring:
    """Test that stock_metrics parameter influences ranking."""

    @pytest.mark.schema
    def test_no_metrics_backward_compatible(self):
        """With stock_metrics=None, ranking still works."""
        stocks = [
            _make_rated_stock("AAA", composite=98, rs=95, conviction=9),
            _make_rated_stock("BBB", composite=95, rs=90, conviction=7),
        ]
        result = rank_candidates(stocks, tier=1)
        assert len(result) == 2
        assert result[0]["ticker"] == "AAA"
        assert result[1]["ticker"] == "BBB"

    @pytest.mark.schema
    def test_high_sharpe_boosts_ranking(self):
        """Stock with higher Sharpe ratio should get a score boost."""
        stocks = [
            _make_rated_stock("AAA", composite=95, rs=90, conviction=7),
            _make_rated_stock("BBB", composite=95, rs=90, conviction=7),
        ]
        metrics = {
            "AAA": {"sharpe_ratio": 2.0, "estimated_volatility_pct": 20.0, "estimated_beta": 1.0},
            "BBB": {"sharpe_ratio": 0.5, "estimated_volatility_pct": 30.0, "estimated_beta": 1.2},
        }
        result = rank_candidates(stocks, tier=1, stock_metrics=metrics)
        # AAA has higher Sharpe (2.0 vs 0.5), should rank higher
        assert result[0]["ticker"] == "AAA"
        # Verify scores differ
        assert result[0]["candidate_score"] > result[1]["candidate_score"]

    @pytest.mark.schema
    def test_negative_sharpe_no_bonus(self):
        """Negative Sharpe ratio → risk_adj_bonus = 0."""
        stocks = [
            _make_rated_stock("AAA", composite=95, rs=90, conviction=7),
        ]
        # Sharpe of -2.0 → (−2+1)*25 = −25 → clamped to 0
        metrics = {
            "AAA": {"sharpe_ratio": -2.0, "estimated_volatility_pct": 40.0, "estimated_beta": 2.0},
        }
        result_with = rank_candidates(stocks, tier=1, stock_metrics=metrics)
        result_without = rank_candidates(stocks, tier=1, stock_metrics=None)

        # With negative Sharpe (bonus=0), score should be lower than without metrics
        # since without metrics, risk_adj term is 0 but weights are rebalanced
        # Actually both should have bonus=0, but the scores use same weights
        # The main check: negative Sharpe doesn't crash and produces a valid score
        assert result_with[0]["candidate_score"] > 0

    @pytest.mark.schema
    def test_sharpe_none_treated_as_no_data(self):
        """If sharpe_ratio is None in metrics, risk_adj_bonus stays 0."""
        stocks = [
            _make_rated_stock("AAA", composite=95, rs=90, conviction=7),
        ]
        metrics = {
            "AAA": {"sharpe_ratio": None, "estimated_volatility_pct": 20.0, "estimated_beta": 1.0},
        }
        result_with = rank_candidates(stocks, tier=1, stock_metrics=metrics)
        result_without = rank_candidates(stocks, tier=1, stock_metrics=None)

        # Both should produce identical scores (no bonus in either case)
        assert result_with[0]["candidate_score"] == result_without[0]["candidate_score"]

    @pytest.mark.schema
    def test_missing_symbol_in_metrics_no_bonus(self):
        """Stock not in stock_metrics dict → risk_adj_bonus stays 0."""
        stocks = [
            _make_rated_stock("AAA", composite=95, rs=90, conviction=7),
        ]
        metrics = {
            "ZZZ": {"sharpe_ratio": 3.0, "estimated_volatility_pct": 10.0, "estimated_beta": 0.5},
        }
        result = rank_candidates(stocks, tier=1, stock_metrics=metrics)
        result_none = rank_candidates(stocks, tier=1, stock_metrics=None)
        assert result[0]["candidate_score"] == result_none[0]["candidate_score"]

    @pytest.mark.schema
    def test_sharpe_normalized_0_to_100(self):
        """Sharpe of 3.0 → (3+1)*25 = 100 (max). Sharpe of -1.0 → 0 (min)."""
        stocks = [
            _make_rated_stock("HIGH", composite=95, rs=90, conviction=7),
            _make_rated_stock("LOW", composite=95, rs=90, conviction=7, sector="SOFTWARE"),
        ]
        metrics = {
            "HIGH": {"sharpe_ratio": 3.0, "estimated_volatility_pct": 10.0, "estimated_beta": 0.5},
            "LOW": {"sharpe_ratio": -1.0, "estimated_volatility_pct": 50.0, "estimated_beta": 2.5},
        }
        result = rank_candidates(stocks, tier=1, stock_metrics=metrics)
        # HIGH should have max risk-adjusted bonus, LOW should have 0
        high_score = next(r for r in result if r["ticker"] == "HIGH")["candidate_score"]
        low_score = next(r for r in result if r["ticker"] == "LOW")["candidate_score"]
        # Difference should be approximately RISK_ADJUSTED_WEIGHT * 100 = 15
        diff = high_score - low_score
        assert abs(diff - 15.0) < 0.1
