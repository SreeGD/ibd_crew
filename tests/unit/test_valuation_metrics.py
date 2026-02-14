"""
Valuation & Risk Metrics — Pure Function Tests
Tests all estimation functions in valuation_metrics.py.
"""

from __future__ import annotations

import pytest

from ibd_agents.tools.valuation_metrics import (
    RISK_FREE_RATE,
    MARKET_RETURN,
    SP500_FORWARD_PE,
    SP500_10Y_AVG_PE,
    SECTOR_PE_BASELINES,
    SECTOR_BETA_BASELINES,
    SECTOR_VOLATILITY_BASELINES,
    build_market_context,
    calculate_alpha,
    calculate_peg,
    calculate_risk_rating,
    calculate_sharpe,
    classify_alpha_category,
    classify_pe_category,
    classify_peg_category,
    classify_return_category,
    classify_sharpe_category,
    compute_all_valuation_metrics,
    estimate_beta,
    estimate_pe,
    estimate_return_pct,
    estimate_volatility,
)


# ---------------------------------------------------------------------------
# TestEstimatePE
# ---------------------------------------------------------------------------

class TestEstimatePE:

    @pytest.mark.schema
    def test_pe_increases_with_growth_ratings(self):
        """Higher Composite/EPS/RS -> higher P/E."""
        pe_low = estimate_pe("CHIPS", 70, 70, 70)
        pe_high = estimate_pe("CHIPS", 99, 99, 99)
        assert pe_high > pe_low

    @pytest.mark.schema
    def test_pe_respects_sector_range(self):
        """SOFTWARE P/E should be higher than BANKS P/E for same ratings."""
        pe_software = estimate_pe("SOFTWARE", 90, 90, 90)
        pe_banks = estimate_pe("BANKS", 90, 90, 90)
        assert pe_software > pe_banks

    @pytest.mark.schema
    def test_pe_clamped_to_bounds(self):
        """P/E never negative, never absurdly high."""
        pe = estimate_pe("BANKS", 99, 99, 99)
        assert pe >= 1.0
        assert pe <= 200.0
        # Low-rated stock
        pe_low = estimate_pe("UTILITIES", 50, 50, 50)
        assert pe_low >= 1.0

    @pytest.mark.schema
    def test_pe_known_inputs_mu(self):
        """MU (CHIPS, 99/99/81) — should be in reasonable range for CHIPS."""
        pe = estimate_pe("CHIPS", 99, 81, 99)
        assert 15.0 <= pe <= 60.0  # CHIPS range


# ---------------------------------------------------------------------------
# TestClassifyPECategory
# ---------------------------------------------------------------------------

class TestClassifyPECategory:

    @pytest.mark.schema
    def test_deep_value(self):
        """PE below sector low -> Deep Value."""
        assert classify_pe_category(7.0, "BANKS") == "Deep Value"

    @pytest.mark.schema
    def test_value(self):
        """PE between low and mid -> Value."""
        assert classify_pe_category(9.5, "BANKS") == "Value"

    @pytest.mark.schema
    def test_reasonable(self):
        """PE near midpoint -> Reasonable."""
        # BANKS mid=11.0, 10% above = 12.1
        assert classify_pe_category(11.0, "BANKS") == "Reasonable"

    @pytest.mark.schema
    def test_growth_premium(self):
        """PE above midpoint up to 1.5x high -> Growth Premium."""
        assert classify_pe_category(18.0, "BANKS") == "Growth Premium"

    @pytest.mark.schema
    def test_speculative(self):
        """PE > 1.5x sector high -> Speculative."""
        # BANKS high=16, 1.5x = 24
        assert classify_pe_category(25.0, "BANKS") == "Speculative"


# ---------------------------------------------------------------------------
# TestCalculatePEG
# ---------------------------------------------------------------------------

class TestCalculatePEG:

    @pytest.mark.schema
    def test_peg_formula_correct(self):
        """PEG = PE / (EPS * 0.5)."""
        result = calculate_peg(25.0, 90)
        assert result == pytest.approx(25.0 / 45.0, abs=0.01)

    @pytest.mark.schema
    def test_peg_zero_eps_returns_none(self):
        assert calculate_peg(25.0, 0) is None

    @pytest.mark.schema
    def test_peg_none_eps_returns_none(self):
        assert calculate_peg(25.0, None) is None

    @pytest.mark.schema
    def test_peg_category_undervalued(self):
        assert classify_peg_category(0.8) == "Undervalued"

    @pytest.mark.schema
    def test_peg_category_fair(self):
        assert classify_peg_category(1.5) == "Fair Value"

    @pytest.mark.schema
    def test_peg_category_expensive(self):
        assert classify_peg_category(2.5) == "Expensive"

    @pytest.mark.schema
    def test_peg_category_none(self):
        assert classify_peg_category(None) is None


# ---------------------------------------------------------------------------
# TestEstimateBeta
# ---------------------------------------------------------------------------

class TestEstimateBeta:

    @pytest.mark.schema
    def test_beta_increases_with_momentum(self):
        """RS 99 should have higher beta than RS 50."""
        beta_high = estimate_beta("CHIPS", 99, 99)
        beta_low = estimate_beta("CHIPS", 50, 50)
        assert beta_high > beta_low

    @pytest.mark.schema
    def test_beta_respects_sector(self):
        """MINING baseline > UTILITIES baseline."""
        beta_mining = estimate_beta("MINING", 85, 85)
        beta_util = estimate_beta("UTILITIES", 85, 85)
        assert beta_mining > beta_util

    @pytest.mark.schema
    def test_beta_clamped(self):
        """Beta in [0.2, 2.5]."""
        beta = estimate_beta("MINING", 99, 99)
        assert 0.2 <= beta <= 2.5
        beta_low = estimate_beta("UTILITIES", 50, 50)
        assert 0.2 <= beta_low <= 2.5

    @pytest.mark.schema
    def test_beta_unknown_sector(self):
        """Unknown sector uses default baseline."""
        beta = estimate_beta("UNKNOWN", 85, 85)
        assert 0.2 <= beta <= 2.5


# ---------------------------------------------------------------------------
# TestEstimateReturn
# ---------------------------------------------------------------------------

class TestEstimateReturn:

    @pytest.mark.schema
    def test_return_rs99(self):
        assert estimate_return_pct(99) == 60.0

    @pytest.mark.schema
    def test_return_rs90(self):
        assert estimate_return_pct(90) == 50.0

    @pytest.mark.schema
    def test_return_rs50(self):
        assert estimate_return_pct(50) == 10.0

    @pytest.mark.schema
    def test_return_category_strong(self):
        assert classify_return_category(40.0) == "Strong"

    @pytest.mark.schema
    def test_return_category_good(self):
        assert classify_return_category(25.0) == "Good"

    @pytest.mark.schema
    def test_return_category_moderate(self):
        assert classify_return_category(5.0) == "Moderate"


# ---------------------------------------------------------------------------
# TestEstimateVolatility
# ---------------------------------------------------------------------------

class TestEstimateVolatility:

    @pytest.mark.schema
    def test_volatility_higher_for_high_rs(self):
        """Higher RS -> more momentum -> more volatility."""
        vol_high = estimate_volatility("CHIPS", 99, 99)
        vol_low = estimate_volatility("CHIPS", 50, 50)
        assert vol_high > vol_low

    @pytest.mark.schema
    def test_volatility_sector_baseline(self):
        """MINING vol > UTILITIES vol."""
        vol_mining = estimate_volatility("MINING", 80, 80)
        vol_util = estimate_volatility("UTILITIES", 80, 80)
        assert vol_mining > vol_util

    @pytest.mark.schema
    def test_volatility_clamped(self):
        """Volatility in [10, 60]."""
        vol = estimate_volatility("MINING", 99, 99)
        assert 10.0 <= vol <= 60.0


# ---------------------------------------------------------------------------
# TestSharpeRatio
# ---------------------------------------------------------------------------

class TestSharpeRatio:

    @pytest.mark.schema
    def test_sharpe_formula(self):
        """Sharpe = (Return - RiskFree) / Volatility."""
        result = calculate_sharpe(50.0, 30.0)
        expected = (50.0 - RISK_FREE_RATE) / 30.0
        assert result == pytest.approx(expected, abs=0.01)

    @pytest.mark.schema
    def test_sharpe_category_excellent(self):
        assert classify_sharpe_category(1.6) == "Excellent"

    @pytest.mark.schema
    def test_sharpe_category_below_average(self):
        assert classify_sharpe_category(0.5) == "Below Average"

    @pytest.mark.schema
    def test_sharpe_zero_volatility(self):
        assert calculate_sharpe(50.0, 0) is None


# ---------------------------------------------------------------------------
# TestAlpha
# ---------------------------------------------------------------------------

class TestAlpha:

    @pytest.mark.schema
    def test_alpha_formula(self):
        """Alpha = Return - (RiskFree + Beta * (MarketReturn - RiskFree))."""
        result = calculate_alpha(50.0, 1.2)
        expected = 50.0 - (RISK_FREE_RATE + 1.2 * (MARKET_RETURN - RISK_FREE_RATE))
        assert result == pytest.approx(expected, abs=0.01)

    @pytest.mark.schema
    def test_alpha_category_strong_outperformer(self):
        assert classify_alpha_category(15.0) == "Strong Outperformer"

    @pytest.mark.schema
    def test_alpha_category_underperformer(self):
        assert classify_alpha_category(-8.0) == "Underperformer"


# ---------------------------------------------------------------------------
# TestRiskRating
# ---------------------------------------------------------------------------

class TestRiskRating:

    @pytest.mark.schema
    def test_excellent(self):
        assert calculate_risk_rating(1.6, 1.0) == "Excellent"

    @pytest.mark.schema
    def test_good(self):
        assert calculate_risk_rating(1.2, 1.3) == "Good"

    @pytest.mark.schema
    def test_moderate(self):
        assert calculate_risk_rating(0.8, 1.6) == "Moderate"

    @pytest.mark.schema
    def test_below_average(self):
        assert calculate_risk_rating(0.5, 1.8) == "Below Average"

    @pytest.mark.schema
    def test_poor(self):
        assert calculate_risk_rating(0.3, 2.0) == "Poor"

    @pytest.mark.schema
    def test_none_sharpe(self):
        assert calculate_risk_rating(None, 1.0) == "Moderate"


# ---------------------------------------------------------------------------
# TestComputeAll
# ---------------------------------------------------------------------------

class TestComputeAll:

    @pytest.mark.schema
    def test_returns_all_keys(self):
        """compute_all returns all 13 expected keys."""
        result = compute_all_valuation_metrics("CHIPS", 99, 99, 81)
        expected_keys = {
            "estimated_pe", "pe_category", "peg_ratio", "peg_category",
            "estimated_beta", "estimated_return_pct", "return_category",
            "estimated_volatility_pct", "sharpe_ratio", "sharpe_category",
            "alpha_pct", "alpha_category", "risk_rating",
        }
        assert set(result.keys()) == expected_keys

    @pytest.mark.schema
    def test_known_stock_mu(self):
        """MU (CHIPS, Comp 99, RS 99, EPS 81) — reasonable ranges."""
        m = compute_all_valuation_metrics("CHIPS", 99, 99, 81)
        assert 15.0 <= m["estimated_pe"] <= 60.0
        assert m["pe_category"] in ("Value", "Reasonable", "Growth Premium", "Speculative")
        assert m["peg_ratio"] is not None and m["peg_ratio"] > 0
        assert m["estimated_beta"] > 1.0  # CHIPS is high-beta
        assert m["estimated_return_pct"] == 60.0  # RS 99
        assert m["sharpe_ratio"] is not None and m["sharpe_ratio"] > 0
        assert m["risk_rating"] in ("Excellent", "Good", "Moderate", "Below Average", "Poor")

    @pytest.mark.schema
    def test_unknown_sector_fallback(self):
        """UNKNOWN sector uses defaults, doesn't crash."""
        result = compute_all_valuation_metrics("UNKNOWN", 85, 85, 85)
        assert result["estimated_pe"] > 0
        assert result["estimated_beta"] > 0
        assert result["risk_rating"] is not None


# ---------------------------------------------------------------------------
# TestMarketContext
# ---------------------------------------------------------------------------

class TestMarketContext:

    @pytest.mark.schema
    def test_market_context_values(self):
        ctx = build_market_context()
        assert ctx["sp500_forward_pe"] == SP500_FORWARD_PE
        assert ctx["sp500_10y_avg_pe"] == SP500_10Y_AVG_PE
        assert ctx["risk_free_rate"] == RISK_FREE_RATE
        assert ctx["current_premium_pct"] == pytest.approx(18.9, abs=0.5)

    @pytest.mark.schema
    def test_sector_baselines_complete(self):
        """All 28 IBD sectors have PE, Beta, and Volatility baselines."""
        from ibd_agents.schemas.research_output import IBD_SECTORS
        for sector in IBD_SECTORS:
            assert sector in SECTOR_PE_BASELINES, f"Missing PE baseline for {sector}"
            assert sector in SECTOR_BETA_BASELINES, f"Missing Beta baseline for {sector}"
            assert sector in SECTOR_VOLATILITY_BASELINES, f"Missing Vol baseline for {sector}"
