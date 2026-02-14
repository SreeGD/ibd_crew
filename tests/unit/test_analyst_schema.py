"""
Agent 02: Analyst Agent — Schema Tests
Level 1: Pure deterministic validation, no LLM calls.

Tests per Agent02Analyst.md §8 Level 1.
"""

import pytest
from pydantic import ValidationError

from ibd_agents.schemas.analyst_output import (
    CATALYST_TYPES,
    TIER_LABELS,
    AnalystOutput,
    ETFTierDistribution,
    EliteFilterSummary,
    IBDKeep,
    RatedETF,
    RatedStock,
    SectorRank,
    TierDistribution,
    UnratedStock,
)
from ibd_agents.schemas.research_output import compute_preliminary_tier, is_ibd_keep_candidate
from ibd_agents.tools.elite_screener import (
    calculate_conviction,
    calculate_etf_conviction,
    generate_catalyst,
    generate_strengths,
    generate_weaknesses,
    passes_all_elite,
    passes_elite_acc_dis,
    passes_elite_eps,
    passes_elite_rs,
    passes_elite_smr,
    passes_etf_acc_dis_screen,
    passes_etf_rs_screen,
    passes_etf_screen,
)
from ibd_agents.tools.sector_scorer import sector_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rated_stock(**overrides) -> dict:
    """Minimal valid RatedStock dict."""
    base = {
        "symbol": "TEST",
        "company_name": "Test Company",
        "sector": "CHIPS",
        "security_type": "stock",
        "tier": 1,
        "tier_label": "Momentum",
        "composite_rating": 99,
        "rs_rating": 95,
        "eps_rating": 90,
        "smr_rating": "A",
        "acc_dis_rating": "B",
        "passes_eps_filter": True,
        "passes_rs_filter": True,
        "passes_smr_filter": True,
        "passes_acc_dis_filter": True,
        "passes_all_elite": True,
        "is_ibd_keep": True,
        "is_multi_source_validated": False,
        "ibd_lists": ["IBD 50"],
        "schwab_themes": [],
        "validation_score": 3,
        "conviction": 8,
        "strengths": ["Strong composite rating"],
        "weaknesses": ["Limited cross-source validation"],
        "catalyst": "Key catalysts: sector momentum with strong price action",
        "reasoning": "T1 Momentum stock with elite composite 99, RS 95, on IBD 50 list — strong growth candidate",
        "sector_rank_in_sector": 1,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Tier Assignment Tests (spec §4.2)
# ---------------------------------------------------------------------------

class TestTierAssignment:
    """Deterministic tier assignment from Framework v4.0 thresholds."""

    @pytest.mark.schema
    def test_tier_1_threshold_exact(self):
        """Comp=95, RS=90, EPS=80 -> T1."""
        assert compute_preliminary_tier(95, 90, 80) == 1

    @pytest.mark.schema
    def test_tier_1_below_composite(self):
        """Comp=94, RS=90, EPS=80 -> NOT T1."""
        assert compute_preliminary_tier(94, 90, 80) != 1

    @pytest.mark.schema
    def test_tier_1_below_rs(self):
        """Comp=95, RS=89, EPS=80 -> NOT T1."""
        assert compute_preliminary_tier(95, 89, 80) != 1

    @pytest.mark.schema
    def test_tier_2_threshold_exact(self):
        """Comp=85, RS=80, EPS=75 -> T2."""
        assert compute_preliminary_tier(85, 80, 75) == 2

    @pytest.mark.schema
    def test_tier_3_threshold_exact(self):
        """Comp=80, RS=75, EPS=70 -> T3."""
        assert compute_preliminary_tier(80, 75, 70) == 3

    @pytest.mark.schema
    def test_below_all_thresholds(self):
        """Comp=79 -> None (below T3)."""
        assert compute_preliminary_tier(79, 75, 70) is None

    @pytest.mark.schema
    def test_tier_labels_correct(self):
        assert TIER_LABELS[1] == "Momentum"
        assert TIER_LABELS[2] == "Quality Growth"
        assert TIER_LABELS[3] == "Defensive"


# ---------------------------------------------------------------------------
# IBD Keep Tests (spec §4.3)
# ---------------------------------------------------------------------------

class TestIBDKeep:

    @pytest.mark.schema
    def test_ibd_keep_boundary(self):
        """Comp=93, RS=90 -> keep=True."""
        assert is_ibd_keep_candidate(93, 90) is True

    @pytest.mark.schema
    def test_ibd_keep_miss_by_one(self):
        """Comp=92, RS=90 -> keep=False."""
        assert is_ibd_keep_candidate(92, 90) is False

    @pytest.mark.schema
    def test_ibd_keep_rs_miss(self):
        """Comp=93, RS=89 -> keep=False."""
        assert is_ibd_keep_candidate(93, 89) is False


# ---------------------------------------------------------------------------
# Elite Filter Tests (spec §4.1)
# ---------------------------------------------------------------------------

class TestEliteFilters:

    @pytest.mark.schema
    def test_elite_eps_filter_pass(self):
        assert passes_elite_eps(85) is True

    @pytest.mark.schema
    def test_elite_eps_filter_fail(self):
        """EPS=84 -> fails."""
        assert passes_elite_eps(84) is False

    @pytest.mark.schema
    def test_elite_rs_filter_pass(self):
        assert passes_elite_rs(75) is True

    @pytest.mark.schema
    def test_elite_rs_filter_fail(self):
        assert passes_elite_rs(74) is False

    @pytest.mark.schema
    def test_elite_smr_filter_pass(self):
        assert passes_elite_smr("A") is True
        assert passes_elite_smr("B") is True
        assert passes_elite_smr("B-") is True

    @pytest.mark.schema
    def test_elite_smr_filter_fail(self):
        """SMR='C' -> fails."""
        assert passes_elite_smr("C") is False

    @pytest.mark.schema
    def test_elite_smr_filter_missing(self):
        assert passes_elite_smr(None) is None

    @pytest.mark.schema
    def test_elite_acc_dis_pass(self):
        assert passes_elite_acc_dis("A") is True
        assert passes_elite_acc_dis("B-") is True

    @pytest.mark.schema
    def test_elite_acc_dis_fail(self):
        assert passes_elite_acc_dis("D") is False

    @pytest.mark.schema
    def test_elite_all_pass(self):
        """All 4 above threshold -> passes_all=True."""
        assert passes_all_elite(90, 80, "A", "B") is True

    @pytest.mark.schema
    def test_elite_all_fail_one(self):
        """EPS fails -> passes_all=False."""
        assert passes_all_elite(84, 80, "A", "B") is False

    @pytest.mark.schema
    def test_elite_missing_smr(self):
        """Missing SMR -> passes_all=False."""
        assert passes_all_elite(90, 80, None, "B") is False


# ---------------------------------------------------------------------------
# Sector Score Formula (spec §4.4)
# ---------------------------------------------------------------------------

class TestSectorScore:

    @pytest.mark.schema
    def test_sector_score_formula(self):
        """Known inputs -> known result."""
        # From spec §6 Decision 4:
        # (98.83*0.25) + (93.50*0.25) + (83.3*0.30) + (66.7*0.20) = ~86.41
        result = sector_score(98.83, 93.50, 83.3, 66.7)
        assert abs(result - 86.4125) < 0.01

    @pytest.mark.schema
    def test_sector_score_zeros(self):
        result = sector_score(0, 0, 0, 0)
        assert result == 0.0

    @pytest.mark.schema
    def test_sector_ranking_order(self):
        """Higher sector_score should get lower rank number."""
        s1 = sector_score(95, 90, 80, 60)
        s2 = sector_score(80, 75, 40, 30)
        assert s1 > s2


# ---------------------------------------------------------------------------
# Conviction Tests
# ---------------------------------------------------------------------------

class TestConviction:

    @pytest.mark.schema
    def test_conviction_range(self):
        """Conviction must be 1-10."""
        # T1 base=7, +elite=8, +multi=9, +lists=10 -> capped at 10
        assert calculate_conviction(1, True, True, ["L1", "L2"], ["Theme1"]) == 10

    @pytest.mark.schema
    def test_conviction_min(self):
        """Untiered stock with no bonuses = 1."""
        assert calculate_conviction(None, False, False, [], []) == 1

    @pytest.mark.schema
    def test_conviction_tier_2_base(self):
        """T2 base=5."""
        assert calculate_conviction(2, False, False, [], []) == 5

    @pytest.mark.schema
    def test_conviction_tier_3_base(self):
        """T3 base=3."""
        assert calculate_conviction(3, False, False, [], []) == 3

    @pytest.mark.schema
    def test_conviction_bonuses_additive(self):
        """T1(7) + elite(1) + multi(1) = 9."""
        assert calculate_conviction(1, True, True, ["L1"], []) == 9

    @pytest.mark.schema
    def test_conviction_cap_at_10(self):
        """Cannot exceed 10."""
        result = calculate_conviction(1, True, True, ["L1", "L2"], ["T1"])
        assert result == 10

    @pytest.mark.schema
    def test_conviction_morningstar_5star_bonus(self):
        """Morningstar 5-star adds +1 conviction."""
        base = calculate_conviction(2, False, False, [], [])
        with_ms = calculate_conviction(2, False, False, [], [], morningstar_rating="5-star")
        assert with_ms == base + 1

    @pytest.mark.schema
    def test_conviction_morningstar_4star_no_bonus(self):
        """Morningstar 4-star does NOT add conviction bonus."""
        base = calculate_conviction(2, False, False, [], [])
        with_ms = calculate_conviction(2, False, False, [], [], morningstar_rating="4-star")
        assert with_ms == base

    @pytest.mark.schema
    def test_conviction_wide_moat_bonus(self):
        """Wide economic moat adds +1 conviction."""
        base = calculate_conviction(2, False, False, [], [])
        with_moat = calculate_conviction(2, False, False, [], [], economic_moat="Wide")
        assert with_moat == base + 1

    @pytest.mark.schema
    def test_conviction_narrow_moat_no_bonus(self):
        """Narrow moat does NOT add conviction bonus."""
        base = calculate_conviction(2, False, False, [], [])
        with_moat = calculate_conviction(2, False, False, [], [], economic_moat="Narrow")
        assert with_moat == base

    @pytest.mark.schema
    def test_conviction_morningstar_both_bonuses(self):
        """5-star + Wide moat = +2 conviction."""
        base = calculate_conviction(2, False, False, [], [])
        with_both = calculate_conviction(2, False, False, [], [],
                                         morningstar_rating="5-star", economic_moat="Wide")
        assert with_both == base + 2


# ---------------------------------------------------------------------------
# Real Data Tests (spec §5)
# ---------------------------------------------------------------------------

class TestRealData:

    @pytest.mark.schema
    def test_real_data_MU(self):
        """MU(99/99/81) -> T1, keep=True."""
        assert compute_preliminary_tier(99, 99, 81) == 1
        assert is_ibd_keep_candidate(99, 99) is True

    @pytest.mark.schema
    def test_real_data_SCCO(self):
        """SCCO(98/94/79) -> T2 (EPS 79 < 80 fails T1)."""
        assert compute_preliminary_tier(98, 94, 79) == 2

    @pytest.mark.schema
    def test_real_data_GS(self):
        """GS(93/91/76) -> T2, keep=True."""
        assert compute_preliminary_tier(93, 91, 76) == 2
        assert is_ibd_keep_candidate(93, 91) is True

    @pytest.mark.schema
    def test_real_data_APP(self):
        """APP(98/95/80) -> T1 (EPS 80 = T1 threshold)."""
        assert compute_preliminary_tier(98, 95, 80) == 1

    @pytest.mark.schema
    def test_real_data_RGLD(self):
        """RGLD(99/90/95) -> T1 by strict thresholds."""
        assert compute_preliminary_tier(99, 90, 95) == 1

    @pytest.mark.schema
    def test_real_data_ISRG(self):
        """ISRG(99/80/96) -> T2 (RS 80 < T1 threshold 90)."""
        assert compute_preliminary_tier(99, 80, 96) == 2


# ---------------------------------------------------------------------------
# Pydantic Model Validation
# ---------------------------------------------------------------------------

class TestRatedStockModel:

    @pytest.mark.schema
    def test_valid_rated_stock(self):
        stock = RatedStock(**_make_rated_stock())
        assert stock.symbol == "TEST"
        assert stock.tier == 1
        assert stock.tier_label == "Momentum"

    @pytest.mark.schema
    def test_tier_label_mismatch_rejects(self):
        with pytest.raises(ValidationError, match="tier_label"):
            RatedStock(**_make_rated_stock(tier=2, tier_label="Momentum"))

    @pytest.mark.schema
    def test_keep_logic_mismatch_rejects(self):
        with pytest.raises(ValidationError, match="is_ibd_keep"):
            RatedStock(**_make_rated_stock(
                composite_rating=80, rs_rating=70,
                is_ibd_keep=True,
                tier=3, tier_label="Defensive",
            ))

    @pytest.mark.schema
    def test_conviction_out_of_range_rejects(self):
        with pytest.raises(ValidationError):
            RatedStock(**_make_rated_stock(conviction=11))

    @pytest.mark.schema
    def test_reasoning_too_short_rejects(self):
        with pytest.raises(ValidationError, match="reasoning"):
            RatedStock(**_make_rated_stock(reasoning="Short"))

    @pytest.mark.schema
    def test_morningstar_fields_optional(self):
        """RatedStock without Morningstar fields is valid (all None by default)."""
        stock = RatedStock(**_make_rated_stock())
        assert stock.morningstar_rating is None
        assert stock.economic_moat is None
        assert stock.fair_value is None
        assert stock.price_to_fair_value is None

    @pytest.mark.schema
    def test_morningstar_fields_populated(self):
        """RatedStock with Morningstar fields is valid."""
        stock = RatedStock(**_make_rated_stock(
            morningstar_rating="5-star",
            economic_moat="Wide",
            fair_value=590.0,
            morningstar_price=420.0,
            price_to_fair_value=0.71,
            morningstar_uncertainty="Medium",
        ))
        assert stock.morningstar_rating == "5-star"
        assert stock.economic_moat == "Wide"
        assert stock.fair_value == 590.0
        assert stock.price_to_fair_value == 0.71


class TestUnratedStockModel:

    @pytest.mark.schema
    def test_valid_unrated_stock(self):
        stock = UnratedStock(
            symbol="AMZN",
            company_name="Amazon",
            sector="INTERNET",
            reason_unrated="No IBD ratings available",
            schwab_themes=["E-Commerce"],
            validation_score=2,
            sources=["schwab_ecommerce.pdf"],
            note="Fundamental Keep — requires manual assessment",
        )
        assert stock.symbol == "AMZN"


class TestSectorRankModel:

    @pytest.mark.schema
    def test_valid_sector_rank(self):
        sr = SectorRank(
            rank=1,
            sector="MINING",
            stock_count=6,
            avg_composite=98.83,
            avg_rs=93.50,
            elite_pct=83.3,
            multi_list_pct=66.7,
            sector_score=86.42,
            top_stocks=["NEM", "KGC", "AEM"],
            ibd_keep_count=5,
        )
        assert sr.rank == 1


class TestIBDKeepModel:

    @pytest.mark.schema
    def test_valid_ibd_keep(self):
        keep = IBDKeep(
            symbol="CLS",
            composite_rating=99,
            rs_rating=97,
            eps_rating=99,
            ibd_lists=["IBD Big Cap 20"],
            tier=1,
            keep_rationale="Comp 99, RS 97 — well above keep threshold",
            override_risk=None,
        )
        assert keep.symbol == "CLS"

    @pytest.mark.schema
    def test_keep_below_threshold_rejects(self):
        with pytest.raises(ValidationError):
            IBDKeep(
                symbol="FAIL",
                composite_rating=90,  # below 93
                rs_rating=90,
                eps_rating=80,
                ibd_lists=[],
                tier=2,
                keep_rationale="Should fail",
            )


class TestAnalystOutput:

    @pytest.mark.schema
    def test_valid_output(self):
        output = AnalystOutput(
            rated_stocks=[RatedStock(**_make_rated_stock())],
            unrated_stocks=[],
            sector_rankings=[
                SectorRank(
                    rank=1, sector="CHIPS", stock_count=1,
                    avg_composite=99.0, avg_rs=95.0,
                    elite_pct=100.0, multi_list_pct=100.0,
                    sector_score=86.0, top_stocks=["TEST"],
                    ibd_keep_count=1,
                )
            ],
            ibd_keeps=[],
            elite_filter_summary=EliteFilterSummary(
                total_screened=1, passed_all_four=1,
                failed_eps=0, failed_rs=0, failed_smr=0,
                failed_acc_dis=0, missing_ratings=0,
            ),
            tier_distribution=TierDistribution(
                tier_1_count=1, tier_2_count=0, tier_3_count=0,
                below_threshold_count=0, unrated_count=0,
            ),
            methodology_notes="Framework v4.0 Elite Screening applied",
            analysis_date="2026-02-10",
            summary="Analyzed 1 stock: 1 T1 Momentum. Top sector: CHIPS (score 86.0). 0 IBD keeps identified.",
        )
        assert len(output.rated_stocks) == 1

    @pytest.mark.schema
    def test_duplicate_sector_ranks_rejects(self):
        sr1 = SectorRank(
            rank=1, sector="CHIPS", stock_count=1,
            avg_composite=95.0, avg_rs=90.0, elite_pct=50.0,
            multi_list_pct=50.0, sector_score=70.0,
            top_stocks=["TEST"], ibd_keep_count=0,
        )
        sr2 = SectorRank(
            rank=1, sector="MINING", stock_count=1,
            avg_composite=90.0, avg_rs=85.0, elite_pct=40.0,
            multi_list_pct=40.0, sector_score=65.0,
            top_stocks=["NEM"], ibd_keep_count=0,
        )
        with pytest.raises(ValidationError, match="unique"):
            AnalystOutput(
                rated_stocks=[RatedStock(**_make_rated_stock())],
                sector_rankings=[sr1, sr2],
                elite_filter_summary=EliteFilterSummary(
                    total_screened=1, passed_all_four=1,
                    failed_eps=0, failed_rs=0, failed_smr=0,
                    failed_acc_dis=0, missing_ratings=0,
                ),
                tier_distribution=TierDistribution(
                    tier_1_count=1, tier_2_count=0, tier_3_count=0,
                    below_threshold_count=0, unrated_count=0,
                ),
                methodology_notes="Test",
                analysis_date="2026-02-10",
                summary="Test summary that is at least fifty characters long for validation to pass.",
            )


# ---------------------------------------------------------------------------
# RatedETF Tests
# ---------------------------------------------------------------------------

def _make_rated_etf(**overrides) -> dict:
    """Minimal valid RatedETF dict."""
    base = {
        "symbol": "SOXX",
        "name": "iShares Semiconductor ETF",
        "tier": 1,
        "rs_rating": 85,
        "acc_dis_rating": "B",
        "ytd_change": 12.5,
        "volume_pct_change": 15.0,
        "price_change": 3.20,
        "close_price": 245.30,
        "div_yield": 0.8,
        "etf_score": 72.5,
        "etf_rank": 1,
        "theme_tags": ["Artificial Intelligence"],
        "strengths": ["RS 85 — strong momentum"],
        "weaknesses": [],
    }
    base.update(overrides)
    return base


class TestRatedETFModel:

    @pytest.mark.schema
    def test_valid_rated_etf(self):
        etf = RatedETF(**_make_rated_etf())
        assert etf.symbol == "SOXX"
        assert etf.tier == 1
        assert etf.etf_score == 72.5

    @pytest.mark.schema
    def test_rated_etf_tier_range(self):
        for tier in [1, 2, 3]:
            etf = RatedETF(**_make_rated_etf(tier=tier))
            assert etf.tier == tier

    @pytest.mark.schema
    def test_rated_etf_tier_invalid(self):
        with pytest.raises(ValidationError):
            RatedETF(**_make_rated_etf(tier=0))
        with pytest.raises(ValidationError):
            RatedETF(**_make_rated_etf(tier=4))

    @pytest.mark.schema
    def test_rated_etf_rs_range(self):
        etf = RatedETF(**_make_rated_etf(rs_rating=99))
        assert etf.rs_rating == 99
        with pytest.raises(ValidationError):
            RatedETF(**_make_rated_etf(rs_rating=0))
        with pytest.raises(ValidationError):
            RatedETF(**_make_rated_etf(rs_rating=100))

    @pytest.mark.schema
    def test_rated_etf_acc_dis_valid(self):
        for val in ["A", "B", "B-", "C", "D", "E", None]:
            etf = RatedETF(**_make_rated_etf(acc_dis_rating=val))
            assert etf.acc_dis_rating == val

    @pytest.mark.schema
    def test_rated_etf_acc_dis_invalid(self):
        with pytest.raises(ValidationError):
            RatedETF(**_make_rated_etf(acc_dis_rating="F"))

    @pytest.mark.schema
    def test_rated_etf_rank_positive(self):
        with pytest.raises(ValidationError):
            RatedETF(**_make_rated_etf(etf_rank=0))

    @pytest.mark.schema
    def test_rated_etf_symbol_uppercased(self):
        etf = RatedETF(**_make_rated_etf(symbol="soxx"))
        assert etf.symbol == "SOXX"

    @pytest.mark.schema
    def test_rated_etf_optional_fields(self):
        etf = RatedETF(**_make_rated_etf(
            rs_rating=None, acc_dis_rating=None, ytd_change=None,
            volume_pct_change=None, price_change=None, close_price=None,
            div_yield=None,
        ))
        assert etf.rs_rating is None
        assert etf.div_yield is None

    @pytest.mark.schema
    def test_rated_etf_conviction_field(self):
        etf = RatedETF(**_make_rated_etf(conviction=8))
        assert etf.conviction == 8

    @pytest.mark.schema
    def test_rated_etf_conviction_range(self):
        with pytest.raises(ValidationError):
            RatedETF(**_make_rated_etf(conviction=0))
        with pytest.raises(ValidationError):
            RatedETF(**_make_rated_etf(conviction=11))

    @pytest.mark.schema
    def test_rated_etf_conviction_default(self):
        """Conviction defaults to 1 for backward compatibility."""
        etf = RatedETF(**_make_rated_etf())
        assert etf.conviction >= 1

    @pytest.mark.schema
    def test_rated_etf_focus_fields(self):
        etf = RatedETF(**_make_rated_etf(
            focus="Semiconductors", focus_group="Semiconductors", focus_rank=1,
        ))
        assert etf.focus == "Semiconductors"
        assert etf.focus_group == "Semiconductors"
        assert etf.focus_rank == 1

    @pytest.mark.schema
    def test_rated_etf_screening_fields(self):
        etf = RatedETF(**_make_rated_etf(
            passes_rs_screen=True, passes_acc_dis_screen=True, passes_etf_screen=True,
        ))
        assert etf.passes_rs_screen is True
        assert etf.passes_acc_dis_screen is True
        assert etf.passes_etf_screen is True


# ---------------------------------------------------------------------------
# ETF Screening Tests
# ---------------------------------------------------------------------------

class TestETFScreening:

    @pytest.mark.schema
    def test_etf_rs_screen_pass(self):
        assert passes_etf_rs_screen(70) is True
        assert passes_etf_rs_screen(99) is True

    @pytest.mark.schema
    def test_etf_rs_screen_fail(self):
        assert passes_etf_rs_screen(69) is False

    @pytest.mark.schema
    def test_etf_rs_screen_none(self):
        assert passes_etf_rs_screen(None) is None

    @pytest.mark.schema
    def test_etf_acc_dis_screen_pass(self):
        assert passes_etf_acc_dis_screen("A") is True
        assert passes_etf_acc_dis_screen("B") is True
        assert passes_etf_acc_dis_screen("B-") is True

    @pytest.mark.schema
    def test_etf_acc_dis_screen_fail(self):
        assert passes_etf_acc_dis_screen("C") is False
        assert passes_etf_acc_dis_screen("D") is False
        assert passes_etf_acc_dis_screen("E") is False

    @pytest.mark.schema
    def test_etf_acc_dis_screen_none(self):
        assert passes_etf_acc_dis_screen(None) is None

    @pytest.mark.schema
    def test_etf_screen_both_pass(self):
        assert passes_etf_screen(85, "A") is True

    @pytest.mark.schema
    def test_etf_screen_rs_fail(self):
        assert passes_etf_screen(50, "A") is False

    @pytest.mark.schema
    def test_etf_screen_acc_dis_fail(self):
        assert passes_etf_screen(85, "D") is False

    @pytest.mark.schema
    def test_etf_screen_missing_data(self):
        assert passes_etf_screen(None, "A") is False
        assert passes_etf_screen(85, None) is False


# ---------------------------------------------------------------------------
# ETF Conviction Tests
# ---------------------------------------------------------------------------

class TestETFConviction:

    @pytest.mark.schema
    def test_etf_conviction_t1_max(self):
        """T1(7) + RS>=85(1) + Acc/Dis A(1) + YTD>10(1) + themes(1) = 10."""
        result = calculate_etf_conviction(1, 90, "A", 15.0, ["AI"], 80.0)
        assert result == 10

    @pytest.mark.schema
    def test_etf_conviction_t1_base(self):
        """T1 base with no bonuses = 7."""
        result = calculate_etf_conviction(1, 60, "C", -5.0, [], 50.0)
        assert result == 7

    @pytest.mark.schema
    def test_etf_conviction_t2_base(self):
        """T2 base = 5."""
        result = calculate_etf_conviction(2, 60, "C", -5.0, [], 50.0)
        assert result == 5

    @pytest.mark.schema
    def test_etf_conviction_t3_base(self):
        """T3 base = 3."""
        result = calculate_etf_conviction(3, 60, "C", -5.0, [], 50.0)
        assert result == 3

    @pytest.mark.schema
    def test_etf_conviction_none_tier(self):
        """None tier defaults to 3."""
        result = calculate_etf_conviction(None, 60, "C", -5.0, [], 50.0)
        assert result == 3

    @pytest.mark.schema
    def test_etf_conviction_cap_10(self):
        """Cannot exceed 10."""
        result = calculate_etf_conviction(1, 90, "A", 15.0, ["AI", "Cloud"], 80.0)
        assert result == 10

    @pytest.mark.schema
    def test_etf_conviction_additive(self):
        """T2(5) + RS>=85(1) + themes(1) = 7."""
        result = calculate_etf_conviction(2, 90, "C", -5.0, ["AI"], 50.0)
        assert result == 7


# ---------------------------------------------------------------------------
# ETF Tier Distribution Tests
# ---------------------------------------------------------------------------

class TestETFTierDistribution:

    @pytest.mark.schema
    def test_valid_distribution(self):
        dist = ETFTierDistribution(
            tier_1_count=5, tier_2_count=8, tier_3_count=3, total_rated=16,
        )
        assert dist.total_rated == 16

    @pytest.mark.schema
    def test_defaults(self):
        dist = ETFTierDistribution()
        assert dist.tier_1_count == 0
        assert dist.total_rated == 0

    @pytest.mark.schema
    def test_negative_rejects(self):
        with pytest.raises(ValidationError):
            ETFTierDistribution(tier_1_count=-1)


# ---------------------------------------------------------------------------
# Catalyst Field Tests
# ---------------------------------------------------------------------------

class TestCatalystFields:

    @pytest.mark.schema
    def test_valid_catalyst_date_format(self):
        stock = RatedStock(**_make_rated_stock(catalyst_date="2026-03-15"))
        assert stock.catalyst_date == "2026-03-15"

    @pytest.mark.schema
    def test_invalid_catalyst_date_fails(self):
        with pytest.raises(ValidationError):
            RatedStock(**_make_rated_stock(catalyst_date="03-15-2026"))

    @pytest.mark.schema
    def test_invalid_catalyst_date_format_fails(self):
        with pytest.raises(ValidationError):
            RatedStock(**_make_rated_stock(catalyst_date="2026/03/15"))

    @pytest.mark.schema
    def test_null_catalyst_date_valid(self):
        stock = RatedStock(**_make_rated_stock(catalyst_date=None))
        assert stock.catalyst_date is None

    @pytest.mark.schema
    def test_valid_catalyst_types(self):
        for ct in CATALYST_TYPES:
            stock = RatedStock(**_make_rated_stock(catalyst_type=ct))
            assert stock.catalyst_type == ct

    @pytest.mark.schema
    def test_invalid_catalyst_type_fails(self):
        with pytest.raises(ValidationError):
            RatedStock(**_make_rated_stock(catalyst_type="ipo"))

    @pytest.mark.schema
    def test_catalyst_adjustment_range(self):
        for adj in (-1, 0, 1, 2):
            stock = RatedStock(**_make_rated_stock(catalyst_conviction_adjustment=adj))
            assert stock.catalyst_conviction_adjustment == adj

    @pytest.mark.schema
    def test_catalyst_adjustment_too_high_fails(self):
        with pytest.raises(ValidationError):
            RatedStock(**_make_rated_stock(catalyst_conviction_adjustment=3))

    @pytest.mark.schema
    def test_catalyst_adjustment_too_low_fails(self):
        with pytest.raises(ValidationError):
            RatedStock(**_make_rated_stock(catalyst_conviction_adjustment=-2))

    @pytest.mark.schema
    def test_catalyst_source_llm_valid(self):
        stock = RatedStock(**_make_rated_stock(catalyst_source="llm"))
        assert stock.catalyst_source == "llm"

    @pytest.mark.schema
    def test_catalyst_source_template_valid(self):
        stock = RatedStock(**_make_rated_stock(catalyst_source="template"))
        assert stock.catalyst_source == "template"

    @pytest.mark.schema
    def test_catalyst_source_invalid_fails(self):
        with pytest.raises(ValidationError):
            RatedStock(**_make_rated_stock(catalyst_source="gpt"))


# ---------------------------------------------------------------------------
# Cap Size Propagation Tests
# ---------------------------------------------------------------------------

class TestCapSizePropagation:

    @pytest.mark.schema
    @pytest.mark.parametrize("cap", ["large", "mid", "small", None])
    def test_rated_stock_accepts_valid_cap_sizes(self, cap):
        stock = RatedStock(**_make_rated_stock(cap_size=cap))
        assert stock.cap_size == cap

    @pytest.mark.schema
    def test_rated_stock_rejects_invalid_cap_size(self):
        with pytest.raises(ValidationError):
            RatedStock(**_make_rated_stock(cap_size="mega"))
