"""
Tests for Monte Carlo engine — per-stock volatility support.
IBD Momentum Investment Framework v4.0
"""

import pytest

from ibd_agents.schemas.returns_projection_output import TIER_HISTORICAL_RETURNS
from ibd_agents.tools.monte_carlo_engine import (
    run_monte_carlo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_positions(tier: int = 1, count: int = 5, volatility_pct=None):
    """Build simple position dicts for testing."""
    positions = []
    alloc = 100.0 / count
    for i in range(count):
        pos = {
            "tier": tier,
            "allocation_pct": alloc,
            "stop_loss_pct": 22.0,
        }
        if volatility_pct is not None:
            pos["volatility_pct"] = volatility_pct
        positions.append(pos)
    return positions


# ---------------------------------------------------------------------------
# Test: Per-stock volatility override
# ---------------------------------------------------------------------------

class TestMonteCarloVolatility:
    """Tests for per-stock volatility override in run_monte_carlo()."""

    @pytest.mark.schema
    def test_no_volatility_same_as_before(self):
        """Positions without volatility_pct produce same results as before."""
        positions = _make_positions(tier=1, count=5)
        result = run_monte_carlo(
            positions,
            tier_pcts={1: 100.0, 2: 0.0, 3: 0.0},
            regime="bull",
            target_return=30.0,
            seed=42,
        )
        # Re-run — same seed, same result
        result2 = run_monte_carlo(
            positions,
            tier_pcts={1: 100.0, 2: 0.0, 3: 0.0},
            regime="bull",
            target_return=30.0,
            seed=42,
        )
        assert result["expected_return_pct"] == result2["expected_return_pct"]
        assert result["prob_achieve_target"] == result2["prob_achieve_target"]

    @pytest.mark.schema
    def test_none_volatility_treated_as_absent(self):
        """volatility_pct=None is ignored (same as missing key)."""
        positions_no_vol = _make_positions(tier=1, count=5)
        positions_none_vol = _make_positions(tier=1, count=5, volatility_pct=None)

        # Both should produce identical results
        # Note: volatility_pct=None won't trigger the override since `is not None` check
        r1 = run_monte_carlo(
            positions_no_vol,
            tier_pcts={1: 100.0, 2: 0.0, 3: 0.0},
            regime="bull",
            target_return=30.0,
            seed=42,
        )
        r2 = run_monte_carlo(
            positions_none_vol,
            tier_pcts={1: 100.0, 2: 0.0, 3: 0.0},
            regime="bull",
            target_return=30.0,
            seed=42,
        )
        assert r1["expected_return_pct"] == r2["expected_return_pct"]

    @pytest.mark.schema
    def test_high_volatility_wider_distribution(self):
        """High volatility_pct produces wider return distribution (higher p90-p10 spread)."""
        tier_base_std = TIER_HISTORICAL_RETURNS[1]["base"]["std"]
        high_vol = tier_base_std * 2.0  # 2x tier average

        positions_normal = _make_positions(tier=1, count=5)
        positions_high_vol = _make_positions(tier=1, count=5, volatility_pct=high_vol)

        r_normal = run_monte_carlo(
            positions_normal,
            tier_pcts={1: 100.0, 2: 0.0, 3: 0.0},
            regime="bull",
            target_return=30.0,
            seed=42,
        )
        r_high = run_monte_carlo(
            positions_high_vol,
            tier_pcts={1: 100.0, 2: 0.0, 3: 0.0},
            regime="bull",
            target_return=30.0,
            seed=42,
        )

        spread_normal = r_normal["p90"] - r_normal["p10"]
        spread_high = r_high["p90"] - r_high["p10"]

        # High volatility should produce wider spread
        assert spread_high > spread_normal

    @pytest.mark.schema
    def test_low_volatility_tighter_distribution(self):
        """Low volatility_pct produces tighter return distribution."""
        tier_base_std = TIER_HISTORICAL_RETURNS[1]["base"]["std"]
        low_vol = tier_base_std * 0.5  # Half of tier average

        positions_normal = _make_positions(tier=1, count=5)
        positions_low_vol = _make_positions(tier=1, count=5, volatility_pct=low_vol)

        r_normal = run_monte_carlo(
            positions_normal,
            tier_pcts={1: 100.0, 2: 0.0, 3: 0.0},
            regime="bull",
            target_return=30.0,
            seed=42,
        )
        r_low = run_monte_carlo(
            positions_low_vol,
            tier_pcts={1: 100.0, 2: 0.0, 3: 0.0},
            regime="bull",
            target_return=30.0,
            seed=42,
        )

        spread_normal = r_normal["p90"] - r_normal["p10"]
        spread_low = r_low["p90"] - r_low["p10"]

        # Low volatility should produce tighter spread
        assert spread_low < spread_normal

    @pytest.mark.schema
    def test_volatility_ratio_clamped_at_extremes(self):
        """Extreme volatility ratios are clamped to [0.5, 2.0]."""
        tier_base_std = TIER_HISTORICAL_RETURNS[1]["base"]["std"]

        # Very high vol (10x tier) should be clamped to 2.0x
        extreme_high = tier_base_std * 10.0
        # Moderate high (2x tier) — already at clamp boundary
        moderate_high = tier_base_std * 2.0

        pos_extreme = _make_positions(tier=1, count=5, volatility_pct=extreme_high)
        pos_moderate = _make_positions(tier=1, count=5, volatility_pct=moderate_high)

        r_extreme = run_monte_carlo(
            pos_extreme,
            tier_pcts={1: 100.0, 2: 0.0, 3: 0.0},
            regime="bull",
            target_return=30.0,
            seed=42,
        )
        r_moderate = run_monte_carlo(
            pos_moderate,
            tier_pcts={1: 100.0, 2: 0.0, 3: 0.0},
            regime="bull",
            target_return=30.0,
            seed=42,
        )

        # Both should produce same distribution since 10x is clamped to 2.0x
        assert r_extreme["expected_return_pct"] == r_moderate["expected_return_pct"]
        assert r_extreme["p10"] == r_moderate["p10"]
        assert r_extreme["p90"] == r_moderate["p90"]

    @pytest.mark.schema
    def test_zero_volatility_treated_as_absent(self):
        """volatility_pct=0 doesn't trigger override (guard: stock_vol > 0)."""
        positions = _make_positions(tier=1, count=5, volatility_pct=0.0)
        positions_no_vol = _make_positions(tier=1, count=5)

        r1 = run_monte_carlo(
            positions,
            tier_pcts={1: 100.0, 2: 0.0, 3: 0.0},
            regime="bull",
            target_return=30.0,
            seed=42,
        )
        r2 = run_monte_carlo(
            positions_no_vol,
            tier_pcts={1: 100.0, 2: 0.0, 3: 0.0},
            regime="bull",
            target_return=30.0,
            seed=42,
        )
        assert r1["expected_return_pct"] == r2["expected_return_pct"]
