"""
Agent 11: Value Investor
Value-Oriented Analysis — IBD Momentum Framework v4.0

Screens and classifies stocks by value characteristics.
Computes composite Value Score (0-100), identifies GARP, Deep Value,
Quality Value, and Dividend Value opportunities. Detects value traps
and momentum-value mismatches.

Never makes buy/sell recommendations — only discovers, scores, and organizes.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import List, Optional

try:
    from crewai import Agent, Task
    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False
    Agent = None
    Task = None

from ibd_agents.schemas.analyst_output import AnalystOutput, RatedStock
from ibd_agents.schemas.value_investor_output import (
    VALUE_CATEGORIES,
    VALUE_CATEGORY_DESCRIPTIONS,
    MoatAnalysisSummary,
    SectorValueAnalysis,
    ValueCategorySummary,
    ValueInvestorOutput,
    ValueStock,
)
from ibd_agents.tools.value_screener import (
    ValueScreenerTool,
    assess_momentum_value_alignment,
    classify_value_category,
    compute_discount_score,
    compute_moat_quality_score,
    compute_morningstar_score,
    compute_pe_value_score,
    compute_peg_value_score,
    compute_value_score,
    detect_value_trap,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CrewAI Builders
# ---------------------------------------------------------------------------

def build_value_investor_agent() -> "Agent":
    """Build CrewAI Value Investor Agent."""
    if not HAS_CREWAI:
        raise ImportError("crewai is required for agentic mode")
    return Agent(
        role="Value Investment Analyst",
        goal=(
            "Analyze the rated stock universe through a value investing lens. "
            "Compute composite Value Scores, classify stocks into value categories "
            "(GARP, Deep Value, Quality Value, Dividend Value), detect value traps, "
            "and identify the best value opportunities. Never make buy/sell "
            "recommendations — only discover, score, and organize."
        ),
        backstory=(
            "You are a seasoned value investor in the tradition of Buffett, Munger, "
            "and Morningstar's methodology. You look for wide moat companies trading "
            "below intrinsic value, growth at reasonable prices (GARP), and statistically "
            "cheap stocks. You are disciplined about avoiding value traps — stocks "
            "that appear cheap but have deteriorating fundamentals."
        ),
        tools=[ValueScreenerTool()],
        verbose=True,
        allow_delegation=False,
        max_iter=15,
    )


def build_value_investor_task(agent: "Agent", analyst_output_json: str = "") -> "Task":
    """Build CrewAI Value Investor Task."""
    if not HAS_CREWAI:
        raise ImportError("crewai is required for agentic mode")
    return Task(
        description=f"""Analyze all rated stocks from the Value Investing perspective.

FOR EACH RATED STOCK:
1. Compute Morningstar Score (star + moat + discount + certainty)
2. Compute component value scores (PE, PEG, Moat, Discount)
3. Compute weighted composite Value Score
4. Classify into value category (GARP / Deep Value / Quality Value / Dividend Value)
5. Detect value trap risk signals
6. Assess momentum-value alignment

AGGREGATE ANALYSIS:
- Rank all stocks by value_score
- Sector-level value analysis
- Category-level summaries
- Top value picks list
- Moat analysis summary

Analyst data:
{analyst_output_json}
""",
        expected_output=(
            "JSON object with value_stocks, sector_value_analysis, "
            "value_category_summaries, moat_analysis, value_traps, "
            "momentum_value_mismatches, top_value_picks."
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Deterministic Pipeline
# ---------------------------------------------------------------------------

def run_value_investor_pipeline(analyst_output: AnalystOutput) -> ValueInvestorOutput:
    """
    Run the deterministic value investor pipeline.

    5 phases:
    1. Score each stock (component scores → composite → category → trap → alignment)
    2. Rank by value_score descending
    3. Build ValueStock objects
    4. Aggregate: sector, category, moat analysis
    5. Assemble ValueInvestorOutput
    """
    logger.info("[Agent 11] Running Value Investor pipeline ...")
    rated_stocks = analyst_output.rated_stocks

    # --- Phase 1: Score each stock ---
    scored_data: list[dict] = []
    for s in rated_stocks:
        ms = compute_morningstar_score(
            s.morningstar_rating, s.economic_moat,
            s.price_to_fair_value, s.morningstar_uncertainty,
        )
        pe_vs = compute_pe_value_score(s.estimated_pe, s.pe_category, s.sector)
        peg_vs = compute_peg_value_score(s.peg_ratio, s.peg_category)
        moat_qs = compute_moat_quality_score(
            s.economic_moat, s.morningstar_rating, s.smr_rating,
        )
        disc_s = compute_discount_score(s.price_to_fair_value)
        v_score = compute_value_score(
            ms["morningstar_score"], pe_vs, peg_vs, moat_qs, disc_s,
        )
        cat, cat_reason = classify_value_category(
            s.pe_category, s.peg_ratio, s.peg_category,
            s.rs_rating, s.eps_rating, s.economic_moat,
            s.morningstar_rating, s.price_to_fair_value,
            s.llm_dividend_yield,
        )
        trap_risk, trap_signals = detect_value_trap(
            s.pe_category, s.rs_rating, s.eps_rating,
            s.smr_rating, s.acc_dis_rating,
            s.price_to_fair_value, cat,
        )
        align, align_detail = assess_momentum_value_alignment(
            s.rs_rating, v_score, cat,
        )

        scored_data.append({
            "stock": s,
            "ms": ms,
            "pe_value_score": pe_vs,
            "peg_value_score": peg_vs,
            "moat_quality_score": moat_qs,
            "discount_score": disc_s,
            "value_score": v_score,
            "category": cat,
            "category_reason": cat_reason,
            "trap_risk": trap_risk,
            "trap_signals": trap_signals,
            "alignment": align,
            "alignment_detail": align_detail,
        })

    # --- Phase 2: Rank by value_score descending ---
    scored_data.sort(key=lambda x: (-x["value_score"], x["stock"].symbol))

    # --- Phase 3: Build ValueStock objects ---
    value_stocks: list[ValueStock] = []
    for rank, item in enumerate(scored_data, 1):
        s: RatedStock = item["stock"]
        ms = item["ms"]
        try:
            vs = ValueStock(
                symbol=s.symbol,
                company_name=s.company_name,
                sector=s.sector,
                tier=s.tier,
                tier_label=s.tier_label,
                composite_rating=s.composite_rating,
                rs_rating=s.rs_rating,
                eps_rating=s.eps_rating,
                smr_rating=s.smr_rating,
                acc_dis_rating=s.acc_dis_rating,
                morningstar_rating=s.morningstar_rating,
                economic_moat=s.economic_moat,
                fair_value=s.fair_value,
                morningstar_price=s.morningstar_price,
                price_to_fair_value=s.price_to_fair_value,
                morningstar_uncertainty=s.morningstar_uncertainty,
                estimated_pe=s.estimated_pe,
                pe_category=s.pe_category,
                peg_ratio=s.peg_ratio,
                peg_category=s.peg_category,
                llm_pe_ratio=s.llm_pe_ratio,
                llm_forward_pe=s.llm_forward_pe,
                llm_dividend_yield=s.llm_dividend_yield,
                llm_market_cap_b=s.llm_market_cap_b,
                llm_valuation_grade=s.llm_valuation_grade,
                morningstar_score=ms["morningstar_score"],
                star_points=ms["star_points"],
                moat_points=ms["moat_points"],
                discount_points=ms["discount_points"],
                certainty_points=ms["certainty_points"],
                pe_value_score=item["pe_value_score"],
                peg_value_score=item["peg_value_score"],
                moat_quality_score=item["moat_quality_score"],
                discount_score=item["discount_score"],
                value_score=item["value_score"],
                value_category=item["category"],
                value_category_reasoning=item["category_reason"],
                value_rank=rank,
                is_value_trap_risk=item["trap_risk"] != "None",
                value_trap_risk_level=item["trap_risk"],
                value_trap_signals=item["trap_signals"],
                momentum_value_alignment=item["alignment"],
                alignment_detail=item["alignment_detail"],
            )
            value_stocks.append(vs)
        except Exception as e:
            logger.warning(f"[Agent 11] Skipping {s.symbol}: {e}")

    logger.info(f"[Agent 11] Scored {len(value_stocks)} stocks")

    # --- Phase 4: Aggregations ---
    sector_analyses = _build_sector_analysis(value_stocks)
    category_summaries = _build_category_summaries(value_stocks)
    moat_summary = _build_moat_analysis(value_stocks)

    # Collect lists
    value_traps = [vs.symbol for vs in value_stocks if vs.is_value_trap_risk]
    mismatches = [
        vs.symbol for vs in value_stocks
        if vs.momentum_value_alignment == "Strong Mismatch"
    ]
    top_picks = [
        vs.symbol for vs in value_stocks
        if vs.value_category != "Not Value"
    ][:15]
    # Ensure at least 1 top pick
    if not top_picks:
        top_picks = [value_stocks[0].symbol] if value_stocks else []

    # Summary
    cat_counts = {}
    for vs in value_stocks:
        cat_counts[vs.value_category] = cat_counts.get(vs.value_category, 0) + 1
    cat_parts = [f"{c}: {n}" for c, n in sorted(cat_counts.items(), key=lambda x: -x[1])]

    summary = (
        f"Value analysis of {len(value_stocks)} stocks: "
        f"{', '.join(cat_parts)}. "
        f"{len(value_traps)} value trap risks identified, "
        f"{len(mismatches)} momentum-value mismatches. "
        f"Top value score: {value_stocks[0].value_score:.1f} ({value_stocks[0].symbol})."
    )

    methodology = (
        "Composite Value Score (0-100) = weighted average of 5 components: "
        "Morningstar Score (30%), P/E Value (20%), PEG Value (20%), "
        "Moat Quality (15%), Discount to Fair Value (15%). "
        "Categories: Quality Value (Wide moat + 4/5-star + P/FV<=1.0), "
        "Deep Value (Deep Value P/E or P/FV<0.80), "
        "GARP (PEG<=2.0 + Reasonable/Value P/E + RS>=80 + EPS>=75), "
        "Dividend Value (yield>=2.0% + reasonable P/E). "
        "Value trap detection: RS<50, EPS<60, SMR C/D/E, Acc/Dis D/E."
    )

    # --- Phase 5: Assemble output ---
    return ValueInvestorOutput(
        value_stocks=value_stocks,
        sector_value_analysis=sector_analyses,
        value_category_summaries=category_summaries,
        moat_analysis=moat_summary,
        value_traps=value_traps,
        momentum_value_mismatches=mismatches,
        top_value_picks=top_picks,
        methodology_notes=methodology,
        analysis_date=date.today().isoformat(),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _build_sector_analysis(value_stocks: List[ValueStock]) -> List[SectorValueAnalysis]:
    """Build per-sector value analysis."""
    sectors: dict[str, list[ValueStock]] = {}
    for vs in value_stocks:
        sectors.setdefault(vs.sector, []).append(vs)

    analyses = []
    for sector, stocks in sectors.items():
        pes = [s.estimated_pe for s in stocks if s.estimated_pe is not None]
        pegs = [s.peg_ratio for s in stocks if s.peg_ratio is not None]
        pfvs = [s.price_to_fair_value for s in stocks if s.price_to_fair_value is not None]

        analyses.append(SectorValueAnalysis(
            sector=sector,
            stock_count=len(stocks),
            avg_value_score=round(sum(s.value_score for s in stocks) / len(stocks), 1),
            avg_pe=round(sum(pes) / len(pes), 1) if pes else None,
            avg_peg=round(sum(pegs) / len(pegs), 2) if pegs else None,
            avg_pfv=round(sum(pfvs) / len(pfvs), 2) if pfvs else None,
            garp_count=sum(1 for s in stocks if s.value_category == "GARP"),
            deep_value_count=sum(1 for s in stocks if s.value_category == "Deep Value"),
            quality_value_count=sum(1 for s in stocks if s.value_category == "Quality Value"),
            dividend_value_count=sum(1 for s in stocks if s.value_category == "Dividend Value"),
            value_trap_count=sum(1 for s in stocks if s.is_value_trap_risk),
            moat_wide_count=sum(1 for s in stocks if s.economic_moat == "Wide"),
            moat_narrow_count=sum(1 for s in stocks if s.economic_moat == "Narrow"),
            top_value_stocks=[s.symbol for s in sorted(stocks, key=lambda x: -x.value_score)[:5]],
            sector_value_rank=1,  # placeholder, reassigned below after sorting
        ))

    # Rank sectors by avg_value_score
    analyses.sort(key=lambda a: -a.avg_value_score)
    for rank, a in enumerate(analyses, 1):
        a.sector_value_rank = rank

    return analyses


def _build_category_summaries(value_stocks: List[ValueStock]) -> List[ValueCategorySummary]:
    """Build per-category summary."""
    categories: dict[str, list[ValueStock]] = {}
    for vs in value_stocks:
        categories.setdefault(vs.value_category, []).append(vs)

    summaries = []
    for cat in VALUE_CATEGORIES:
        stocks = categories.get(cat, [])
        if not stocks:
            continue

        pes = [s.estimated_pe for s in stocks if s.estimated_pe is not None]
        pegs = [s.peg_ratio for s in stocks if s.peg_ratio is not None]
        pfvs = [s.price_to_fair_value for s in stocks if s.price_to_fair_value is not None]

        summaries.append(ValueCategorySummary(
            category=cat,
            stock_count=len(stocks),
            avg_value_score=round(sum(s.value_score for s in stocks) / len(stocks), 1),
            avg_pe=round(sum(pes) / len(pes), 1) if pes else None,
            avg_peg=round(sum(pegs) / len(pegs), 2) if pegs else None,
            avg_pfv=round(sum(pfvs) / len(pfvs), 2) if pfvs else None,
            top_stocks=[s.symbol for s in sorted(stocks, key=lambda x: -x.value_score)[:10]],
            description=VALUE_CATEGORY_DESCRIPTIONS.get(cat, f"{cat} stocks"),
        ))

    return summaries


def _build_moat_analysis(value_stocks: List[ValueStock]) -> MoatAnalysisSummary:
    """Build moat analysis summary."""
    wide = [s for s in value_stocks if s.economic_moat == "Wide"]
    narrow = [s for s in value_stocks if s.economic_moat == "Narrow"]
    none_ = [s for s in value_stocks if s.economic_moat == "None"]
    no_data = [s for s in value_stocks if s.economic_moat is None]

    wide_pfvs = [s.price_to_fair_value for s in wide if s.price_to_fair_value is not None]
    narrow_pfvs = [s.price_to_fair_value for s in narrow if s.price_to_fair_value is not None]

    return MoatAnalysisSummary(
        wide_moat_count=len(wide),
        narrow_moat_count=len(narrow),
        no_moat_count=len(none_),
        no_data_count=len(no_data),
        wide_moat_stocks=[s.symbol for s in sorted(wide, key=lambda x: -x.value_score)],
        avg_value_score_wide=round(sum(s.value_score for s in wide) / len(wide), 1) if wide else None,
        avg_value_score_narrow=round(sum(s.value_score for s in narrow) / len(narrow), 1) if narrow else None,
        avg_pfv_wide=round(sum(wide_pfvs) / len(wide_pfvs), 2) if wide_pfvs else None,
        avg_pfv_narrow=round(sum(narrow_pfvs) / len(narrow_pfvs), 2) if narrow_pfvs else None,
    )
