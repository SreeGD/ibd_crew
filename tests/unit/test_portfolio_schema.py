"""
Agent 05: Portfolio Manager — Schema Tests
Level 1: Pure Pydantic validation, no LLM, no file I/O.
"""

from __future__ import annotations

import pytest

from ibd_agents.schemas.portfolio_output import (
    ALL_KEEPS,
    CASH_RANGE,
    ETF_LIMITS,
    FUNDAMENTAL_KEEPS,
    IBD_KEEPS,
    KEEP_METADATA,
    MAX_LOSS,
    STOCK_ETF_SPLIT,
    STOCK_LIMITS,
    STOP_TIGHTENING,
    TIER_RANGES,
    TRAILING_STOPS,
    VALID_ORDER_ACTIONS,
    WEEK_FOCUSES,
    ImplementationWeek,
    KeepDetail,
    KeepsPlacement,
    OrderAction,
    PortfolioOutput,
    PortfolioPosition,
    TierPortfolio,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_position(
    symbol: str = "AAPL",
    tier: int = 1,
    asset_type: str = "stock",
    target_pct: float = 2.5,
    keep_category: str | None = None,
    trailing_stop_pct: float | None = None,
    conviction: int = 7,
) -> PortfolioPosition:
    """Build a valid PortfolioPosition for testing."""
    stop = trailing_stop_pct if trailing_stop_pct is not None else TRAILING_STOPS[tier]
    limit = ETF_LIMITS[tier] if asset_type == "etf" else STOCK_LIMITS[tier]
    max_loss = MAX_LOSS["etf" if asset_type == "etf" else "stock"][tier]
    return PortfolioPosition(
        symbol=symbol,
        company_name=f"{symbol} Inc",
        sector="CHIPS",
        tier=tier,
        asset_type=asset_type,
        target_pct=min(target_pct, limit),
        trailing_stop_pct=stop,
        max_loss_pct=max_loss,
        keep_category=keep_category,
        conviction=conviction,
        reasoning=f"Test position for {symbol} in tier {tier}",
    )


def _make_tier(
    tier: int,
    n_stocks: int = 5,
    n_etfs: int = 5,
    actual_pct: float = 39.0,
) -> TierPortfolio:
    """Build a valid TierPortfolio."""
    labels = {1: "Momentum", 2: "Quality Growth", 3: "Defensive"}
    pct_per = actual_pct / (n_stocks + n_etfs) if (n_stocks + n_etfs) > 0 else 0

    positions = []
    for i in range(n_stocks):
        positions.append(_make_position(
            symbol=f"S{tier}{i:02d}",
            tier=tier,
            asset_type="stock",
            target_pct=pct_per,
        ))
    for i in range(n_etfs):
        positions.append(_make_position(
            symbol=f"E{tier}{i:02d}",
            tier=tier,
            asset_type="etf",
            target_pct=pct_per,
        ))

    return TierPortfolio(
        tier=tier,
        label=labels[tier],
        target_pct=actual_pct,
        actual_pct=actual_pct,
        positions=positions,
        stock_count=n_stocks,
        etf_count=n_etfs,
    )


def _make_keeps() -> KeepsPlacement:
    """Build a valid KeepsPlacement with all 14 keeps."""
    def _details(symbols: list[str], category: str) -> list[KeepDetail]:
        return [
            KeepDetail(
                symbol=s,
                category=category,
                tier_placed=KEEP_METADATA[s]["default_tier"],
                target_pct=2.0,
                reason=KEEP_METADATA[s]["reason"],
            )
            for s in symbols
        ]

    fund = _details(FUNDAMENTAL_KEEPS, "fundamental")
    ibd = _details(IBD_KEEPS, "ibd")
    total_pct = sum(k.target_pct for k in fund + ibd)

    return KeepsPlacement(
        fundamental_keeps=fund,
        ibd_keeps=ibd,
        total_keeps=14,
        keeps_pct=total_pct,
    )


def _make_portfolio_output(**overrides) -> PortfolioOutput:
    """Build a valid PortfolioOutput. Override any field via kwargs."""
    defaults = dict(
        tier_1=_make_tier(1, n_stocks=5, n_etfs=5, actual_pct=39.0),
        tier_2=_make_tier(2, n_stocks=5, n_etfs=5, actual_pct=37.0),
        tier_3=_make_tier(3, n_stocks=5, n_etfs=5, actual_pct=22.0),
        cash_pct=2.0,
        keeps_placement=_make_keeps(),
        orders=[],
        implementation_plan=[
            ImplementationWeek(week=1, focus="Liquidation", actions=["Sell non-recommended"]),
            ImplementationWeek(week=2, focus="T1 Momentum", actions=["Establish T1 positions"]),
            ImplementationWeek(week=3, focus="T2 Quality Growth", actions=["Establish T2 positions"]),
            ImplementationWeek(week=4, focus="T3 Defensive", actions=["Establish T3 positions"]),
        ],
        sector_exposure={
            "CHIPS": 12.0, "SOFTWARE": 10.0, "MEDICAL": 8.0, "MINING": 8.0,
            "FINANCE": 7.0, "AEROSPACE": 7.0, "INTERNET": 7.0, "CONSUMER": 7.0,
            "BUILDING": 5.0, "ENERGY": 5.0,
        },
        total_positions=30,
        stock_count=15,
        etf_count=15,
        construction_methodology="3-tier portfolio construction with IBD momentum framework rules applied",
        deviation_notes=[],
        analysis_date="2025-01-15",
        summary="Portfolio constructed with 30 positions across 10 sectors, 14 keeps placed, 50/50 stock/ETF ratio",
    )
    defaults.update(overrides)
    return PortfolioOutput(**defaults)


# ---------------------------------------------------------------------------
# Constant Tests
# ---------------------------------------------------------------------------

class TestConstants:

    @pytest.mark.schema
    def test_keeps_total_14(self):
        """ALL_KEEPS has exactly 14 symbols."""
        assert len(ALL_KEEPS) == 14

    @pytest.mark.schema
    def test_keep_categories_sum(self):
        """5 fundamental + 9 IBD = 14."""
        assert len(FUNDAMENTAL_KEEPS) == 5
        assert len(IBD_KEEPS) == 9

    @pytest.mark.schema
    def test_keeps_no_duplicates(self):
        """No duplicate symbols across keep categories."""
        assert len(ALL_KEEPS) == len(set(ALL_KEEPS))

    @pytest.mark.schema
    def test_keep_metadata_covers_all(self):
        """KEEP_METADATA has entry for every keep symbol."""
        for s in ALL_KEEPS:
            assert s in KEEP_METADATA, f"Missing metadata for {s}"

    @pytest.mark.schema
    def test_stock_limits_tiers(self):
        """Stock limits defined for tiers 1-3, decreasing."""
        assert STOCK_LIMITS[1] > STOCK_LIMITS[2] > STOCK_LIMITS[3]

    @pytest.mark.schema
    def test_etf_limits_tiers(self):
        """ETF limits defined for tiers 1-3."""
        assert all(t in ETF_LIMITS for t in (1, 2, 3))

    @pytest.mark.schema
    def test_trailing_stops_decreasing(self):
        """Trailing stops: T1 > T2 > T3."""
        assert TRAILING_STOPS[1] > TRAILING_STOPS[2] > TRAILING_STOPS[3]

    @pytest.mark.schema
    def test_max_loss_stock_decreasing(self):
        """Max loss for stocks: T1 > T2 > T3."""
        ml = MAX_LOSS["stock"]
        assert ml[1] > ml[2] > ml[3]

    @pytest.mark.schema
    def test_max_loss_etf_decreasing(self):
        """Max loss for ETFs: T1 > T2 > T3."""
        ml = MAX_LOSS["etf"]
        assert ml[1] > ml[2] > ml[3]

    @pytest.mark.schema
    def test_four_order_actions(self):
        """Four valid order actions."""
        assert set(VALID_ORDER_ACTIONS) == {"BUY", "SELL", "ADD", "TRIM"}

    @pytest.mark.schema
    def test_four_implementation_weeks(self):
        """Implementation focuses defined for weeks 1-4."""
        assert len(WEEK_FOCUSES) == 4
        assert WEEK_FOCUSES[1] == "Liquidation"


# ---------------------------------------------------------------------------
# Position Tests
# ---------------------------------------------------------------------------

class TestPortfolioPosition:

    @pytest.mark.schema
    def test_valid_stock_position(self):
        """Valid T1 stock position passes validation."""
        pos = _make_position("NVDA", tier=1, asset_type="stock", target_pct=3.0)
        assert pos.symbol == "NVDA"
        assert pos.tier == 1

    @pytest.mark.schema
    def test_valid_etf_position(self):
        """Valid T2 ETF position passes validation."""
        pos = _make_position("QQQ", tier=2, asset_type="etf", target_pct=5.0)
        assert pos.symbol == "QQQ"

    @pytest.mark.schema
    def test_stock_exceeds_t1_limit(self):
        """T1 stock > 5% fails."""
        with pytest.raises(ValueError, match="exceeds max"):
            PortfolioPosition(
                symbol="AAPL", company_name="Apple", sector="CHIPS",
                tier=1, asset_type="stock", target_pct=6.0,
                trailing_stop_pct=22, max_loss_pct=1.1,
                keep_category=None, conviction=7,
                reasoning="Test position exceeding T1 stock limit",
            )

    @pytest.mark.schema
    def test_stock_exceeds_t2_limit(self):
        """T2 stock > 4% fails."""
        with pytest.raises(ValueError, match="exceeds max"):
            PortfolioPosition(
                symbol="AAPL", company_name="Apple", sector="CHIPS",
                tier=2, asset_type="stock", target_pct=4.5,
                trailing_stop_pct=18, max_loss_pct=0.72,
                keep_category=None, conviction=7,
                reasoning="Test position exceeding T2 stock limit",
            )

    @pytest.mark.schema
    def test_stock_exceeds_t3_limit(self):
        """T3 stock > 3% fails."""
        with pytest.raises(ValueError, match="exceeds max"):
            PortfolioPosition(
                symbol="AAPL", company_name="Apple", sector="CHIPS",
                tier=3, asset_type="stock", target_pct=3.5,
                trailing_stop_pct=12, max_loss_pct=0.36,
                keep_category=None, conviction=7,
                reasoning="Test position exceeding T3 stock limit",
            )

    @pytest.mark.schema
    def test_etf_exceeds_t3_limit(self):
        """T3 ETF > 6% fails."""
        with pytest.raises(ValueError, match="exceeds max"):
            PortfolioPosition(
                symbol="XYZ", company_name="XYZ ETF", sector="ENERGY",
                tier=3, asset_type="etf", target_pct=7.0,
                trailing_stop_pct=12, max_loss_pct=0.72,
                keep_category=None, conviction=5,
                reasoning="Test ETF exceeding T3 ETF limit",
            )

    @pytest.mark.schema
    def test_stop_mismatch_fails(self):
        """Trailing stop outside valid range for tier fails."""
        with pytest.raises(ValueError, match="stop"):
            PortfolioPosition(
                symbol="AAPL", company_name="Apple", sector="CHIPS",
                tier=1, asset_type="stock", target_pct=3.0,
                trailing_stop_pct=30.0, max_loss_pct=1.1,
                keep_category=None, conviction=7,
                reasoning="Test position with invalid stop percentage",
            )

    @pytest.mark.schema
    def test_invalid_keep_category(self):
        """Invalid keep_category fails."""
        with pytest.raises(ValueError, match="keep_category"):
            _make_position("X", keep_category="invalid")

    @pytest.mark.schema
    def test_symbol_uppercased(self):
        """Symbol is uppercased."""
        pos = _make_position("aapl")
        assert pos.symbol == "AAPL"


# ---------------------------------------------------------------------------
# TierPortfolio Tests
# ---------------------------------------------------------------------------

class TestTierPortfolio:

    @pytest.mark.schema
    def test_valid_tier(self):
        """Valid tier portfolio passes."""
        tier = _make_tier(1, n_stocks=3, n_etfs=3, actual_pct=39.0)
        assert tier.tier == 1
        assert tier.stock_count == 3
        assert tier.etf_count == 3

    @pytest.mark.schema
    def test_count_mismatch_fails(self):
        """Mismatched stock_count vs actual positions fails."""
        with pytest.raises(ValueError, match="stock_count"):
            TierPortfolio(
                tier=1, label="Momentum", target_pct=39, actual_pct=39,
                positions=[_make_position("A", tier=1)],
                stock_count=5, etf_count=0,
            )


# ---------------------------------------------------------------------------
# KeepsPlacement Tests
# ---------------------------------------------------------------------------

class TestKeepsPlacement:

    @pytest.mark.schema
    def test_valid_keeps(self):
        """Valid 14 keeps passes."""
        kp = _make_keeps()
        assert kp.total_keeps == 14

    @pytest.mark.schema
    def test_wrong_total_fails(self):
        """total_keeps != actual count fails."""
        with pytest.raises(ValueError, match="total_keeps"):
            kp = _make_keeps()
            # Reconstruct with wrong total
            KeepsPlacement(
                fundamental_keeps=kp.fundamental_keeps,
                ibd_keeps=kp.ibd_keeps,
                total_keeps=20,
                keeps_pct=kp.keeps_pct,
            )

    @pytest.mark.schema
    def test_missing_fundamental_fails(self):
        """Wrong number of fundamental keeps fails."""
        with pytest.raises(ValueError, match="keeps"):
            kp = _make_keeps()
            KeepsPlacement(
                fundamental_keeps=[],
                ibd_keeps=kp.ibd_keeps,
                total_keeps=9,
                keeps_pct=18.0,
            )


# ---------------------------------------------------------------------------
# OrderAction Tests
# ---------------------------------------------------------------------------

class TestOrderAction:

    @pytest.mark.schema
    def test_valid_order(self):
        """Valid order passes."""
        o = OrderAction(
            symbol="NVDA", action="BUY", tier=1, target_pct=3.0,
            rationale="New momentum position from analyst top 50",
        )
        assert o.action == "BUY"

    @pytest.mark.schema
    def test_invalid_action_fails(self):
        """Invalid action type fails."""
        with pytest.raises(ValueError, match="action"):
            OrderAction(
                symbol="NVDA", action="HOLD", tier=1, target_pct=3.0,
                rationale="Invalid action type for order testing",
            )


# ---------------------------------------------------------------------------
# PortfolioOutput Tests
# ---------------------------------------------------------------------------

class TestPortfolioOutput:

    @pytest.mark.schema
    def test_valid_output(self):
        """Valid PortfolioOutput passes."""
        out = _make_portfolio_output()
        assert out.total_positions == 30

    @pytest.mark.schema
    def test_tier_sum_exceeds_105_fails(self):
        """Tier allocations summing > 105% fails."""
        with pytest.raises(ValueError, match="sum to"):
            _make_portfolio_output(
                tier_1=_make_tier(1, actual_pct=45.0),
                tier_2=_make_tier(2, actual_pct=40.0),
                tier_3=_make_tier(3, actual_pct=25.0),
                cash_pct=5.0,
            )

    @pytest.mark.schema
    def test_fewer_than_8_sectors_fails(self):
        """Fewer than 8 sectors fails."""
        with pytest.raises(ValueError, match="sectors"):
            _make_portfolio_output(
                sector_exposure={"CHIPS": 30.0, "SOFTWARE": 30.0, "MEDICAL": 20.0,
                                 "MINING": 10.0, "FINANCE": 10.0},
            )

    @pytest.mark.schema
    def test_sector_over_40_fails(self):
        """Sector > 40% fails."""
        with pytest.raises(ValueError, match="exceeds max 40"):
            _make_portfolio_output(
                sector_exposure={
                    "CHIPS": 42.0, "SOFTWARE": 8.0, "MEDICAL": 8.0, "MINING": 7.0,
                    "FINANCE": 7.0, "AEROSPACE": 7.0, "INTERNET": 7.0, "CONSUMER": 7.0,
                    "BUILDING": 4.0, "ENERGY": 3.0,
                },
            )

    @pytest.mark.schema
    def test_stock_etf_ratio_out_of_range_fails(self):
        """Stock/ETF ratio outside 35-65% fails."""
        with pytest.raises(ValueError, match="Stock ratio"):
            _make_portfolio_output(
                tier_1=_make_tier(1, n_stocks=8, n_etfs=2, actual_pct=39.0),
                tier_2=_make_tier(2, n_stocks=8, n_etfs=2, actual_pct=37.0),
                tier_3=_make_tier(3, n_stocks=8, n_etfs=2, actual_pct=22.0),
                total_positions=30,
                stock_count=24,
                etf_count=6,
            )

    @pytest.mark.schema
    def test_implementation_weeks_wrong_fails(self):
        """Missing week in implementation plan fails."""
        with pytest.raises(ValueError):
            _make_portfolio_output(
                implementation_plan=[
                    ImplementationWeek(week=1, focus="Liquidation", actions=["sell"]),
                    ImplementationWeek(week=2, focus="T1", actions=["buy"]),
                    ImplementationWeek(week=3, focus="T2", actions=["buy"]),
                    ImplementationWeek(week=3, focus="T3", actions=["buy"]),  # dup week 3
                ],
            )

    @pytest.mark.schema
    def test_summary_too_short_fails(self):
        """Summary < 50 chars fails."""
        with pytest.raises(ValueError):
            _make_portfolio_output(summary="Too short")

    @pytest.mark.schema
    def test_position_count_mismatch_fails(self):
        """total_positions doesn't match actual positions fails."""
        with pytest.raises(ValueError, match="total_positions"):
            _make_portfolio_output(total_positions=99)


# ---------------------------------------------------------------------------
# Dynamic Sizing Field Tests
# ---------------------------------------------------------------------------

class TestDynamicSizingFields:

    @pytest.mark.schema
    def test_volatility_adjustment_defaults_to_zero(self):
        """volatility_adjustment defaults to 0.0."""
        pos = _make_position("NVDA", tier=1)
        assert pos.volatility_adjustment == 0.0

    @pytest.mark.schema
    def test_sizing_source_defaults_to_none(self):
        """sizing_source defaults to None."""
        pos = _make_position("NVDA", tier=1)
        assert pos.sizing_source is None

    @pytest.mark.schema
    def test_sizing_source_llm_accepted(self):
        """sizing_source='llm' is valid."""
        pos = PortfolioPosition(
            symbol="NVDA", company_name="NVIDIA", sector="CHIPS",
            tier=1, asset_type="stock", target_pct=3.0,
            trailing_stop_pct=22, max_loss_pct=1.1,
            keep_category=None, conviction=7,
            reasoning="Test position with LLM sizing source",
            sizing_source="llm",
        )
        assert pos.sizing_source == "llm"

    @pytest.mark.schema
    def test_sizing_source_deterministic_accepted(self):
        """sizing_source='deterministic' is valid."""
        pos = PortfolioPosition(
            symbol="NVDA", company_name="NVIDIA", sector="CHIPS",
            tier=1, asset_type="stock", target_pct=3.0,
            trailing_stop_pct=22, max_loss_pct=1.1,
            keep_category=None, conviction=7,
            reasoning="Test position with deterministic sizing source",
            sizing_source="deterministic",
        )
        assert pos.sizing_source == "deterministic"

    @pytest.mark.schema
    def test_sizing_source_invalid_rejected(self):
        """Invalid sizing_source is rejected."""
        with pytest.raises(ValueError, match="sizing_source"):
            PortfolioPosition(
                symbol="NVDA", company_name="NVIDIA", sector="CHIPS",
                tier=1, asset_type="stock", target_pct=3.0,
                trailing_stop_pct=22, max_loss_pct=1.1,
                keep_category=None, conviction=7,
                reasoning="Test position with invalid sizing source",
                sizing_source="other",
            )

    @pytest.mark.schema
    def test_volatility_adjustment_positive(self):
        """Positive volatility adjustment within range."""
        pos = PortfolioPosition(
            symbol="AAPL", company_name="Apple", sector="COMPUTER",
            tier=2, asset_type="stock", target_pct=3.0,
            trailing_stop_pct=18, max_loss_pct=0.72,
            keep_category=None, conviction=7,
            reasoning="Test position with positive vol adjustment",
            volatility_adjustment=1.5,
        )
        assert pos.volatility_adjustment == 1.5

    @pytest.mark.schema
    def test_volatility_adjustment_negative(self):
        """Negative volatility adjustment within range."""
        pos = PortfolioPosition(
            symbol="AAPL", company_name="Apple", sector="COMPUTER",
            tier=2, asset_type="stock", target_pct=3.0,
            trailing_stop_pct=18, max_loss_pct=0.72,
            keep_category=None, conviction=7,
            reasoning="Test position with negative vol adjustment",
            volatility_adjustment=-1.5,
        )
        assert pos.volatility_adjustment == -1.5

    @pytest.mark.schema
    def test_volatility_adjustment_below_min_fails(self):
        """volatility_adjustment < -2.0 fails."""
        with pytest.raises(ValueError):
            PortfolioPosition(
                symbol="AAPL", company_name="Apple", sector="COMPUTER",
                tier=2, asset_type="stock", target_pct=3.0,
                trailing_stop_pct=18, max_loss_pct=0.72,
                keep_category=None, conviction=7,
                reasoning="Test position with vol adjustment below minimum",
                volatility_adjustment=-3.0,
            )

    @pytest.mark.schema
    def test_volatility_adjustment_above_max_fails(self):
        """volatility_adjustment > 2.0 fails."""
        with pytest.raises(ValueError):
            PortfolioPosition(
                symbol="AAPL", company_name="Apple", sector="COMPUTER",
                tier=2, asset_type="stock", target_pct=3.0,
                trailing_stop_pct=18, max_loss_pct=0.72,
                keep_category=None, conviction=7,
                reasoning="Test position with vol adjustment above maximum",
                volatility_adjustment=3.0,
            )


# ---------------------------------------------------------------------------
# Cap Size Field Tests
# ---------------------------------------------------------------------------

class TestCapSizeField:

    @pytest.mark.schema
    def test_position_accepts_cap_size(self):
        """PortfolioPosition accepts cap_size."""
        pos = PortfolioPosition(
            symbol="NVDA", company_name="NVIDIA", sector="CHIPS",
            tier=1, asset_type="stock", target_pct=3.0,
            trailing_stop_pct=22, max_loss_pct=1.1,
            keep_category=None, conviction=7,
            reasoning="Test position with cap_size field set",
            cap_size="mid",
        )
        assert pos.cap_size == "mid"

    @pytest.mark.schema
    def test_position_cap_size_default_none(self):
        """cap_size defaults to None."""
        pos = _make_position()
        assert pos.cap_size is None
