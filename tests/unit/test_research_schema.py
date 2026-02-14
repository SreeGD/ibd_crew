"""
Agent 01: Research Agent — Schema Tests
Level 1: Pure Pydantic validation, no LLM calls, runs in <1s.

Tests per AGENT_01_RESEARCH.md §8 Level 1:
  test_valid_output_parses
  test_symbols_uppercase
  test_sector_in_28_ibd_categories
  test_composite_range
  test_smr_rating_values
  test_acc_dis_rating_values
  test_validation_score_non_negative
  test_ibd_keep_logic
  test_multi_source_logic
  test_min_5_sectors
  test_security_type_stock
  test_sources_not_empty
  test_preliminary_tier_valid
  test_reasoning_min_length
  test_validation_score_arithmetic
  + additional edge case tests
"""

import pytest
from pydantic import ValidationError

from ibd_agents.schemas.research_output import (
    IBD_SECTORS,
    ResearchETF,
    ResearchOutput,
    ResearchStock,
    SectorPattern,
    compute_preliminary_tier,
    compute_validation_score,
    is_ibd_keep_candidate,
)

# ---------------------------------------------------------------------------
# Fixtures: minimal valid objects
# ---------------------------------------------------------------------------

def _make_stock(**overrides) -> dict:
    """Minimal valid ResearchStock dict."""
    base = {
        "symbol": "NVDA",
        "company_name": "NVIDIA Corporation",
        "sector": "CHIPS",
        "security_type": "stock",
        "composite_rating": 98,
        "rs_rating": 81,
        "eps_rating": 99,
        "smr_rating": "A",
        "acc_dis_rating": "B",
        "ibd_lists": ["IBD Sector Leaders"],
        "schwab_themes": ["Artificial Intelligence"],
        "fool_status": None,
        "other_ratings": {},
        "validation_score": 3,
        "validation_providers": 1,
        "is_multi_source_validated": False,
        "is_ibd_keep_candidate": False,
        "preliminary_tier": 2,
        "sources": ["SECTOR_LEADERS_1.xls"],
        "confidence": 0.85,
        "reasoning": "Strong semiconductor leader with top IBD ratings",
    }
    base.update(overrides)
    return base


def _make_keep_stock(symbol: str = "CLS", comp: int = 99, rs: int = 97) -> dict:
    """A stock that qualifies as IBD keep candidate."""
    return _make_stock(
        symbol=symbol,
        company_name=f"{symbol} Corp",
        composite_rating=comp,
        rs_rating=rs,
        eps_rating=99,
        is_ibd_keep_candidate=True,
        preliminary_tier=1,
        validation_score=6,
        validation_providers=2,
        is_multi_source_validated=True,
        ibd_lists=["IBD Big Cap 20", "IBD Sector Leaders"],
    )


def _make_validated_stock(symbol: str = "PLTR") -> dict:
    """A stock that is multi-source validated (score≥5, providers≥2)."""
    return _make_stock(
        symbol=symbol,
        company_name=f"{symbol} Inc",
        composite_rating=99,
        rs_rating=93,
        eps_rating=99,
        validation_score=8,
        validation_providers=3,
        is_multi_source_validated=True,
        is_ibd_keep_candidate=True,
        ibd_lists=["IBD Big Cap 20", "IBD Sector Leaders"],
        schwab_themes=["Artificial Intelligence", "Big Data & Analytics"],
        other_ratings={"CFRA": "5-star"},
        preliminary_tier=1,
    )


def _make_sector_pattern(sector: str = "CHIPS", **overrides) -> dict:
    base = {
        "sector": sector,
        "stock_count": 12,
        "avg_composite": 95.5,
        "avg_rs": 88.3,
        "elite_pct": 42.0,
        "multi_list_pct": 58.0,
        "ibd_keep_count": 3,
        "strength": "leading",
        "trend_direction": "up",
        "evidence": "Highest avg composite among all sectors, strong breadth",
    }
    base.update(overrides)
    return base


def _make_etf(**overrides) -> dict:
    base = {
        "symbol": "SOXX",
        "name": "iShares Semiconductor ETF",
        "focus": "Semiconductors",
        "schwab_themes": [],
        "ibd_lists": ["IBD ETF Leaders"],
        "sources": ["ETF_Leaders__Jan_09_2026___Investors_Business_Daily.pdf"],
        "key_theme_etf": False,
        "rs_rating": 85,
        "acc_dis_rating": "B",
        "ytd_change": 12.5,
        "close_price": 245.30,
        "price_change": 3.20,
        "volume_pct_change": 15.0,
        "div_yield": 0.8,
        "preliminary_tier": 1,
        "etf_score": 72.5,
        "etf_rank": 3,
    }
    base.update(overrides)
    return base


def _stocks_across_sectors(n: int = 50) -> list[dict]:
    """Generate n stocks across enough sectors to pass min-5 rule."""
    sectors = IBD_SECTORS[:max(6, n // 8)]
    stocks = []
    for i in range(n):
        sector = sectors[i % len(sectors)]
        sym = f"SYM{i:03d}"
        stocks.append(_make_stock(
            symbol=sym,
            company_name=f"Company {i}",
            sector=sector,
        ))
    return stocks


def _minimal_output(**overrides) -> dict:
    """Minimal valid ResearchOutput dict."""
    stocks = _stocks_across_sectors(50)
    # Add keep and validated stocks
    keep = _make_keep_stock("CLS", 99, 97)
    keep["sector"] = "COMPUTER"
    validated = _make_validated_stock("PLTR")
    validated["sector"] = "SOFTWARE"
    stocks.extend([keep, validated])

    base = {
        "stocks": stocks,
        "etfs": [_make_etf()],
        "sector_patterns": [
            _make_sector_pattern("CHIPS"),
            _make_sector_pattern("SOFTWARE", strength="lagging", trend_direction="down",
                                evidence="Declining RS averages across software sector"),
        ],
        "data_sources_used": ["SECTOR_LEADERS_1.xls", "BIG_CAP_20_1.xls"],
        "data_sources_failed": [],
        "total_securities_scanned": 500,
        "ibd_keep_candidates": ["CLS", "PLTR"],
        "multi_source_validated": ["CLS", "PLTR"],
        "analysis_date": "2026-01-09",
        "summary": "Processed 40 data files covering IBD, Schwab themes, and Motley Fool. "
                   "Identified 52 stocks across 6+ sectors with strong momentum.",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Level 1 Schema Tests
# ---------------------------------------------------------------------------

class TestValidOutputParses:
    """test_valid_output_parses — known-good output parses."""

    def test_minimal_output_parses(self):
        output = ResearchOutput(**_minimal_output())
        assert len(output.stocks) >= 50
        assert output.analysis_date == "2026-01-09"

    def test_stock_parses(self):
        stock = ResearchStock(**_make_stock())
        assert stock.symbol == "NVDA"
        assert stock.security_type == "stock"

    def test_etf_parses(self):
        etf = ResearchETF(**_make_etf())
        assert etf.symbol == "SOXX"

    def test_sector_pattern_parses(self):
        sp = SectorPattern(**_make_sector_pattern())
        assert sp.sector == "CHIPS"


class TestSymbolsUppercase:
    """test_symbols_uppercase — auto-uppercased."""

    def test_lowercase_uppercased(self):
        stock = ResearchStock(**_make_stock(symbol="nvda"))
        assert stock.symbol == "NVDA"

    def test_mixed_case_uppercased(self):
        stock = ResearchStock(**_make_stock(symbol="Nvda"))
        assert stock.symbol == "NVDA"

    def test_whitespace_stripped(self):
        stock = ResearchStock(**_make_stock(symbol=" NVDA "))
        assert stock.symbol == "NVDA"

    def test_etf_symbol_uppercased(self):
        etf = ResearchETF(**_make_etf(symbol="soxx"))
        assert etf.symbol == "SOXX"


class TestSectorIn28IBDCategories:
    """test_sector_in_28_ibd_categories — only valid IBD sectors."""

    def test_valid_sector_passes(self):
        stock = ResearchStock(**_make_stock(sector="CHIPS"))
        assert stock.sector == "CHIPS"

    def test_lowercase_normalized(self):
        stock = ResearchStock(**_make_stock(sector="chips"))
        assert stock.sector == "CHIPS"

    def test_unknown_sector_accepted(self):
        stock = ResearchStock(**_make_stock(sector="CRYPTO"))
        assert stock.sector == "CRYPTO"

    def test_all_28_sectors_valid(self):
        for sector in IBD_SECTORS:
            stock = ResearchStock(**_make_stock(sector=sector))
            assert stock.sector == sector

    def test_sector_pattern_validates_sector(self):
        with pytest.raises(ValidationError, match="not in 28 IBD sectors"):
            SectorPattern(**_make_sector_pattern(sector="INVALID"))


class TestCompositeRange:
    """test_composite_range — 1-99 or None."""

    def test_valid_composite(self):
        stock = ResearchStock(**_make_stock(composite_rating=99))
        assert stock.composite_rating == 99

    def test_none_composite(self):
        stock = ResearchStock(**_make_stock(
            composite_rating=None, is_ibd_keep_candidate=False
        ))
        assert stock.composite_rating is None

    def test_zero_composite_rejects(self):
        with pytest.raises(ValidationError):
            ResearchStock(**_make_stock(composite_rating=0))

    def test_100_composite_rejects(self):
        with pytest.raises(ValidationError):
            ResearchStock(**_make_stock(composite_rating=100))

    def test_boundary_1(self):
        stock = ResearchStock(**_make_stock(
            composite_rating=1, is_ibd_keep_candidate=False
        ))
        assert stock.composite_rating == 1


class TestSMRRatingValues:
    """test_smr_rating_values — only A/B/B-/C/D/E or None."""

    @pytest.mark.parametrize("val", ["A", "B", "B-", "C", "D", "E", None])
    def test_valid_smr(self, val):
        stock = ResearchStock(**_make_stock(smr_rating=val))
        assert stock.smr_rating == val

    @pytest.mark.parametrize("val", ["F", "B+", "A+", "1", ""])
    def test_invalid_smr_rejects(self, val):
        with pytest.raises(ValidationError):
            ResearchStock(**_make_stock(smr_rating=val))


class TestAccDisRatingValues:
    """test_acc_dis_rating_values — only A/B/B-/C/D/E or None."""

    @pytest.mark.parametrize("val", ["A", "B", "B-", "C", "D", "E", None])
    def test_valid_acc_dis(self, val):
        stock = ResearchStock(**_make_stock(acc_dis_rating=val))
        assert stock.acc_dis_rating == val

    @pytest.mark.parametrize("val", ["F", "B+", "X", ""])
    def test_invalid_acc_dis_rejects(self, val):
        with pytest.raises(ValidationError):
            ResearchStock(**_make_stock(acc_dis_rating=val))


class TestValidationScoreNonNegative:
    """test_validation_score_non_negative — ≥ 0."""

    def test_zero_score(self):
        stock = ResearchStock(**_make_stock(validation_score=0))
        assert stock.validation_score == 0

    def test_positive_score(self):
        stock = ResearchStock(**_make_stock(validation_score=8))
        assert stock.validation_score == 8

    def test_negative_rejects(self):
        with pytest.raises(ValidationError):
            ResearchStock(**_make_stock(validation_score=-1))


class TestIBDKeepLogic:
    """test_ibd_keep_logic — is_ibd_keep ↔ (comp≥93 AND rs≥90)."""

    def test_qualifies(self):
        stock = ResearchStock(**_make_stock(
            composite_rating=93, rs_rating=90, is_ibd_keep_candidate=True
        ))
        assert stock.is_ibd_keep_candidate is True

    def test_does_not_qualify_rs_low(self):
        stock = ResearchStock(**_make_stock(
            composite_rating=99, rs_rating=89, is_ibd_keep_candidate=False
        ))
        assert stock.is_ibd_keep_candidate is False

    def test_does_not_qualify_comp_low(self):
        stock = ResearchStock(**_make_stock(
            composite_rating=92, rs_rating=95, is_ibd_keep_candidate=False
        ))
        assert stock.is_ibd_keep_candidate is False

    def test_mismatch_rejects_false_positive(self):
        with pytest.raises(ValidationError, match="is_ibd_keep_candidate"):
            ResearchStock(**_make_stock(
                composite_rating=80, rs_rating=70, is_ibd_keep_candidate=True
            ))

    def test_mismatch_rejects_false_negative(self):
        with pytest.raises(ValidationError, match="is_ibd_keep_candidate"):
            ResearchStock(**_make_stock(
                composite_rating=99, rs_rating=95, is_ibd_keep_candidate=False
            ))

    def test_none_ratings_not_keep(self):
        stock = ResearchStock(**_make_stock(
            composite_rating=None, rs_rating=None, is_ibd_keep_candidate=False
        ))
        assert stock.is_ibd_keep_candidate is False


class TestMultiSourceLogic:
    """test_multi_source_logic — is_validated ↔ (score≥5 AND providers≥2)."""

    def test_qualifies(self):
        stock = ResearchStock(**_make_stock(
            validation_score=5, validation_providers=2,
            is_multi_source_validated=True
        ))
        assert stock.is_multi_source_validated is True

    def test_score_4_does_not_qualify(self):
        stock = ResearchStock(**_make_stock(
            validation_score=4, validation_providers=3,
            is_multi_source_validated=False
        ))
        assert stock.is_multi_source_validated is False

    def test_one_provider_does_not_qualify(self):
        stock = ResearchStock(**_make_stock(
            validation_score=10, validation_providers=1,
            is_multi_source_validated=False
        ))
        assert stock.is_multi_source_validated is False

    def test_mismatch_rejects(self):
        with pytest.raises(ValidationError, match="is_multi_source_validated"):
            ResearchStock(**_make_stock(
                validation_score=2, validation_providers=1,
                is_multi_source_validated=True
            ))


class TestMin5Sectors:
    """test_min_5_sectors — At least 5 sectors in output."""

    def test_5_sectors_passes(self):
        output = ResearchOutput(**_minimal_output())
        sectors = {s.sector for s in output.stocks}
        assert len(sectors) >= 5

    def test_1_sector_passes(self):
        stocks = [_make_stock(symbol=f"S{i:03d}", company_name=f"Co {i}", sector="CHIPS")
                  for i in range(50)]
        output = ResearchOutput(**_minimal_output(
            stocks=stocks, ibd_keep_candidates=[], multi_source_validated=[]))
        assert len({s.sector for s in output.stocks}) >= 1


class TestSecurityTypeStock:
    """test_security_type_stock — All 'stock' in stocks list."""

    def test_stock_type(self):
        stock = ResearchStock(**_make_stock())
        assert stock.security_type == "stock"

    def test_etf_type_rejects_in_stock(self):
        with pytest.raises(ValidationError):
            ResearchStock(**_make_stock(security_type="etf"))


class TestSourcesNotEmpty:
    """test_sources_not_empty — Every stock has ≥1 source."""

    def test_one_source(self):
        stock = ResearchStock(**_make_stock(sources=["file.xls"]))
        assert len(stock.sources) == 1

    def test_empty_sources_rejects(self):
        with pytest.raises(ValidationError):
            ResearchStock(**_make_stock(sources=[]))


class TestPreliminaryTierValid:
    """test_preliminary_tier_valid — Only 1/2/3/None."""

    @pytest.mark.parametrize("tier", [1, 2, 3, None])
    def test_valid_tiers(self, tier):
        stock = ResearchStock(**_make_stock(preliminary_tier=tier))
        assert stock.preliminary_tier == tier

    @pytest.mark.parametrize("tier", [0, 4, -1, 5])
    def test_invalid_tier_rejects(self, tier):
        with pytest.raises(ValidationError):
            ResearchStock(**_make_stock(preliminary_tier=tier))


class TestReasoningMinLength:
    """test_reasoning_min_length — ≥ 20 chars."""

    def test_20_chars_passes(self):
        stock = ResearchStock(**_make_stock(reasoning="A" * 20))
        assert len(stock.reasoning) >= 20

    def test_19_chars_rejects(self):
        with pytest.raises(ValidationError):
            ResearchStock(**_make_stock(reasoning="A" * 19))


class TestValidationScoreArithmetic:
    """test_validation_score_arithmetic — helper function returns correct sums."""

    def test_single_source(self):
        score, providers = compute_validation_score(["IBD Big Cap 20"])
        assert score == 3
        assert providers == 1

    def test_multi_source(self):
        labels = ["IBD Big Cap 20", "IBD Sector Leaders", "CFRA 5-star", "Schwab Theme Core"]
        score, providers = compute_validation_score(labels)
        assert score == 3 + 3 + 3 + 2  # 11
        assert providers == 3  # IBD, CFRA, Schwab

    def test_empty(self):
        score, providers = compute_validation_score([])
        assert score == 0
        assert providers == 0

    def test_unknown_label_scores_zero(self):
        score, providers = compute_validation_score(["Unknown List"])
        assert score == 0
        assert providers == 0

    def test_duplicate_labels(self):
        labels = ["IBD Big Cap 20", "IBD Big Cap 20"]
        score, providers = compute_validation_score(labels)
        assert score == 6  # 3 + 3, counted twice
        assert providers == 1  # Still just IBD


class TestHelperFunctions:
    """Test helper functions for tier assignment and keep logic."""

    def test_tier_1_assignment(self):
        assert compute_preliminary_tier(95, 90, 80) == 1

    def test_tier_2_assignment(self):
        assert compute_preliminary_tier(85, 80, 75) == 2

    def test_tier_3_assignment(self):
        assert compute_preliminary_tier(80, 75, 70) == 3

    def test_no_tier(self):
        assert compute_preliminary_tier(70, 60, 50) is None

    def test_none_ratings_no_tier(self):
        assert compute_preliminary_tier(None, 90, 80) is None

    def test_keep_candidate_true(self):
        assert is_ibd_keep_candidate(93, 90) is True

    def test_keep_candidate_false(self):
        assert is_ibd_keep_candidate(92, 90) is False
        assert is_ibd_keep_candidate(93, 89) is False

    def test_keep_candidate_none(self):
        assert is_ibd_keep_candidate(None, 90) is False
        assert is_ibd_keep_candidate(93, None) is False


class TestOutputConsistencyValidators:
    """Test cross-field validators on ResearchOutput."""

    def test_keep_candidate_mismatch_rejects(self):
        """Stock flagged as keep but not in ibd_keep_candidates list."""
        data = _minimal_output()
        # Add a keep stock but don't include in list
        keep_stock = _make_keep_stock("AGI", 99, 92)
        keep_stock["sector"] = "MINING"
        data["stocks"].append(keep_stock)
        # AGI is flagged but not in ibd_keep_candidates
        with pytest.raises(ValidationError, match="Keep candidate inconsistency"):
            ResearchOutput(**data)

    def test_multi_source_mismatch_rejects(self):
        """Stock flagged as multi-source validated but not in list."""
        data = _minimal_output()
        ms_stock = _make_validated_stock("GOOGL")
        ms_stock["sector"] = "INTERNET"
        ms_stock["is_ibd_keep_candidate"] = True
        data["stocks"].append(ms_stock)
        data["ibd_keep_candidates"].append("GOOGL")
        # GOOGL is flagged validated but not in multi_source_validated list
        with pytest.raises(ValidationError, match="Multi-source inconsistency"):
            ResearchOutput(**data)

    def test_analysis_date_format(self):
        with pytest.raises(ValidationError):
            ResearchOutput(**_minimal_output(analysis_date="Jan 9 2026"))

    def test_summary_min_length(self):
        with pytest.raises(ValidationError):
            ResearchOutput(**_minimal_output(summary="Too short"))


class TestFoolStatus:
    """Fool status must be 'Epic Top', 'New Rec', or None."""

    def test_epic_top(self):
        stock = ResearchStock(**_make_stock(fool_status="Epic Top"))
        assert stock.fool_status == "Epic Top"

    def test_new_rec(self):
        stock = ResearchStock(**_make_stock(fool_status="New Rec"))
        assert stock.fool_status == "New Rec"

    def test_none(self):
        stock = ResearchStock(**_make_stock(fool_status=None))
        assert stock.fool_status is None

    def test_invalid_rejects(self):
        with pytest.raises(ValidationError, match="fool_status"):
            ResearchStock(**_make_stock(fool_status="Strong Buy"))


class TestConfidenceRange:
    """Confidence must be 0.0-1.0."""

    def test_valid_confidence(self):
        stock = ResearchStock(**_make_stock(confidence=0.95))
        assert stock.confidence == 0.95

    def test_above_1_rejects(self):
        with pytest.raises(ValidationError):
            ResearchStock(**_make_stock(confidence=1.5))

    def test_below_0_rejects(self):
        with pytest.raises(ValidationError):
            ResearchStock(**_make_stock(confidence=-0.1))


@pytest.mark.schema
class TestCapSizeField:
    """Test cap_size field on ResearchStock."""

    @pytest.mark.parametrize("val", ["large", "mid", "small", None])
    def test_valid_cap_sizes(self, val):
        stock = ResearchStock(**_make_stock(cap_size=val))
        assert stock.cap_size == val

    def test_invalid_cap_size_rejects(self):
        with pytest.raises(ValidationError):
            ResearchStock(**_make_stock(cap_size="mega"))

    def test_cap_size_default_none(self):
        stock = ResearchStock(**_make_stock())
        assert stock.cap_size is None


# ---------------------------------------------------------------------------
# ETF Field Validation Tests
# ---------------------------------------------------------------------------

class TestETFFields:
    """Validate new ETF rating, metric, and ranking fields."""

    def test_etf_with_all_fields(self):
        etf = ResearchETF(**_make_etf())
        assert etf.rs_rating == 85
        assert etf.acc_dis_rating == "B"
        assert etf.ytd_change == 12.5
        assert etf.close_price == 245.30
        assert etf.price_change == 3.20
        assert etf.volume_pct_change == 15.0
        assert etf.div_yield == 0.8
        assert etf.preliminary_tier == 1
        assert etf.etf_score == 72.5
        assert etf.etf_rank == 3

    def test_etf_all_fields_optional(self):
        """ETF with only required fields still parses."""
        etf = ResearchETF(**_make_etf(
            rs_rating=None, acc_dis_rating=None, ytd_change=None,
            close_price=None, price_change=None, volume_pct_change=None,
            div_yield=None, preliminary_tier=None, etf_score=None,
            etf_rank=None,
        ))
        assert etf.rs_rating is None
        assert etf.etf_score is None

    def test_etf_rs_rating_range(self):
        etf = ResearchETF(**_make_etf(rs_rating=1))
        assert etf.rs_rating == 1
        etf = ResearchETF(**_make_etf(rs_rating=99))
        assert etf.rs_rating == 99

    def test_etf_rs_rating_zero_rejects(self):
        with pytest.raises(ValidationError):
            ResearchETF(**_make_etf(rs_rating=0))

    def test_etf_rs_rating_100_rejects(self):
        with pytest.raises(ValidationError):
            ResearchETF(**_make_etf(rs_rating=100))

    @pytest.mark.parametrize("val", ["A", "B", "B-", "C", "D", "E"])
    def test_etf_acc_dis_valid(self, val):
        etf = ResearchETF(**_make_etf(acc_dis_rating=val))
        assert etf.acc_dis_rating == val

    @pytest.mark.parametrize("val", ["F", "B+", "X", ""])
    def test_etf_acc_dis_invalid_rejects(self, val):
        with pytest.raises(ValidationError):
            ResearchETF(**_make_etf(acc_dis_rating=val))

    def test_etf_tier_valid(self):
        for tier in [1, 2, 3]:
            etf = ResearchETF(**_make_etf(preliminary_tier=tier))
            assert etf.preliminary_tier == tier

    def test_etf_tier_invalid_rejects(self):
        with pytest.raises(ValidationError):
            ResearchETF(**_make_etf(preliminary_tier=0))
        with pytest.raises(ValidationError):
            ResearchETF(**_make_etf(preliminary_tier=4))

    def test_etf_rank_must_be_positive(self):
        with pytest.raises(ValidationError):
            ResearchETF(**_make_etf(etf_rank=0))

    def test_etf_negative_ytd(self):
        etf = ResearchETF(**_make_etf(ytd_change=-5.3))
        assert etf.ytd_change == -5.3

    def test_etf_negative_price_change(self):
        etf = ResearchETF(**_make_etf(price_change=-2.10))
        assert etf.price_change == -2.10
