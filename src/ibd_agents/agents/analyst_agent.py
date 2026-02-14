"""
Agent 02: Analyst Agent 📊
Quantitative Investment Analyst — IBD Momentum Framework v4.0

Applies Elite Screening, Tier Assignment, Sector Ranking,
and per-stock analysis to Research Agent's output.
Produces top 50 stocks with conviction, strengths,
weaknesses, and catalysts.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

try:
    from crewai import Agent, Task
    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False
    Agent = None  # type: ignore
    Task = None  # type: ignore

from ibd_agents.schemas.analyst_output import (
    TIER_LABELS,
    AnalystOutput,
    EliteFilterSummary,
    IBDKeep,
    MarketContext,
    PEDistribution,
    PEGAnalysis,
    RatedETF,
    RatedStock,
    SectorPEEntry,
    SectorRank,
    TierDistribution,
    TierPEStats,
    UnratedStock,
    ValuationSummary,
)
from ibd_agents.schemas.research_output import (
    ResearchOutput,
    ResearchStock,
    compute_preliminary_tier,
    is_ibd_keep_candidate,
)
from ibd_agents.tools.elite_screener import (
    EliteScreenerTool,
    calculate_conviction,
    generate_catalyst,
    generate_strengths,
    generate_weaknesses,
    passes_all_elite,
    passes_elite_acc_dis,
    passes_elite_eps,
    passes_elite_rs,
    passes_elite_smr,
    rank_stocks_in_sector,
)
from ibd_agents.tools.sector_scorer import SectorScorerTool, rank_sectors
from ibd_agents.tools.valuation_metrics import (
    ValuationMetricsTool,
    calculate_alpha,
    calculate_risk_rating,
    calculate_sharpe,
    classify_alpha_category,
    classify_pe_category,
    classify_peg_category,
    classify_return_category,
    classify_sharpe_category,
    compute_all_valuation_metrics,
    compute_valuation_summary,
    calculate_peg,
    enrich_valuation_llm,
    estimate_return_pct,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CrewAI Agent Builder
# ---------------------------------------------------------------------------

def build_analyst_agent() -> "Agent":
    """Create the Analyst Agent with tools. Requires crewai."""
    if not HAS_CREWAI:
        raise ImportError("crewai is required for agentic mode. pip install crewai[tools]")
    return Agent(
        role="Quantitative Investment Analyst",
        goal=(
            "Apply Framework v4.0 Elite Screening, Tier Assignment, and Sector "
            "Ranking to Research Agent's stocks. Flag IBD Keeps. Calculate sector "
            "scores. Produce top 50 stocks with conviction, strengths, weaknesses, "
            "and catalysts."
        ),
        backstory=(
            "You are a CFA-certified quantitative analyst with deep expertise in "
            "IBD methodology. You apply rigorous screening — all 4 elite filters "
            "must pass (EPS>=85, RS>=75, SMR A/B/B-, Acc/Dis A/B/B-). You assign "
            "tiers deterministically using Framework v4.0 thresholds. You never "
            "make buy/sell decisions — you analyze, score, and rank. Your output "
            "feeds the Rotation Detector, Sector Strategist, and Portfolio Manager."
        ),
        tools=[
            EliteScreenerTool(),
            SectorScorerTool(),
            ValuationMetricsTool(),
        ],
        verbose=True,
        allow_delegation=False,
        max_iter=15,
        temperature=0.3,
    )


def build_analyst_task(agent: "Agent", research_output_json: str = "") -> "Task":
    """Create the Analyst task. Requires crewai."""
    if not HAS_CREWAI:
        raise ImportError("crewai is required for agentic mode. pip install crewai[tools]")
    return Task(
        description=f"""Analyze Research Agent output per Framework v4.0.

FOR EACH STOCK WITH RATINGS (Composite, RS, EPS):
1. Apply all 4 elite screening filters
2. Assign tier (T1 Momentum / T2 Quality Growth / T3 Defensive)
3. Flag IBD Keep candidates (Comp>=93 AND RS>=90)
4. Score conviction (1-10) with CAN SLIM reasoning
5. Identify strengths, weaknesses, catalyst

SECTOR ANALYSIS:
- Calculate sector score: (avg_comp*0.25)+(avg_rs*0.25)+(elite_pct*0.30)+(multi_list_pct*0.20)
- Rank sectors by score
- Rank stocks within each sector (Comp -> RS -> EPS desc)

OUTPUT: Top 50 stocks with full analysis, sector rankings, IBD keeps.

Research data:
{research_output_json}
""",
        expected_output=(
            "JSON object with rated_stocks (top 50), unrated_stocks, "
            "sector_rankings, ibd_keeps, elite_filter_summary, "
            "tier_distribution, methodology_notes, analysis_date, summary."
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Deterministic Pipeline
# ---------------------------------------------------------------------------

def run_analyst_pipeline(research_output: ResearchOutput) -> AnalystOutput:
    """
    Run the deterministic analyst pipeline without LLM.

    Args:
        research_output: Validated ResearchOutput from Agent 01

    Returns:
        Validated AnalystOutput
    """
    stocks = research_output.stocks

    # --- Step 1: Separate ratable vs unratable ---
    ratable: list[ResearchStock] = []
    unratable: list[ResearchStock] = []

    for s in stocks:
        if (s.composite_rating is not None
                and s.rs_rating is not None
                and s.eps_rating is not None):
            ratable.append(s)
        else:
            unratable.append(s)

    # --- Step 2: Elite screening ---
    elite_results: dict[str, dict] = {}
    failed_eps = 0
    failed_rs = 0
    failed_smr = 0
    failed_acc_dis = 0
    missing_ratings = 0
    passed_all = 0

    for s in ratable:
        p_eps = passes_elite_eps(s.eps_rating)
        p_rs = passes_elite_rs(s.rs_rating)
        p_smr = passes_elite_smr(s.smr_rating)
        p_acc_dis = passes_elite_acc_dis(s.acc_dis_rating)
        all_pass = passes_all_elite(s.eps_rating, s.rs_rating, s.smr_rating, s.acc_dis_rating)

        if not p_eps:
            failed_eps += 1
        if not p_rs:
            failed_rs += 1
        if p_smr is False:
            failed_smr += 1
        elif p_smr is None:
            missing_ratings += 1
        if p_acc_dis is False:
            failed_acc_dis += 1
        elif p_acc_dis is None:
            missing_ratings += 1
        if all_pass:
            passed_all += 1

        elite_results[s.symbol] = {
            "p_eps": p_eps,
            "p_rs": p_rs,
            "p_smr": p_smr,
            "p_acc_dis": p_acc_dis,
            "all_pass": all_pass,
        }

    # --- Step 3: Assign tiers ---
    tier_map: dict[str, Optional[int]] = {}
    t1_count = 0
    t2_count = 0
    t3_count = 0
    below_count = 0

    for s in ratable:
        tier = compute_preliminary_tier(s.composite_rating, s.rs_rating, s.eps_rating)
        tier_map[s.symbol] = tier
        if tier == 1:
            t1_count += 1
        elif tier == 2:
            t2_count += 1
        elif tier == 3:
            t3_count += 1
        else:
            below_count += 1

    # --- Step 4: Rank within sectors ---
    sector_groups: dict[str, list[ResearchStock]] = {}
    for s in ratable:
        sec = s.sector
        if sec not in sector_groups:
            sector_groups[sec] = []
        sector_groups[sec].append(s)

    sector_rank_map: dict[str, int] = {}  # symbol -> rank within sector
    for sec, sec_stocks in sector_groups.items():
        ranked = rank_stocks_in_sector(sec_stocks)
        for i, s in enumerate(ranked, 1):
            sector_rank_map[s.symbol] = i

    # --- Step 5: Sector rankings ---
    sector_rankings = rank_sectors(ratable)

    # --- Step 6-7: Conviction, strengths, weaknesses, catalyst ---
    stock_analysis: list[dict] = []
    for s in ratable:
        tier = tier_map[s.symbol]
        if tier is None:
            continue  # Below threshold — not included in top 50

        er = elite_results[s.symbol]
        conviction = calculate_conviction(
            tier=tier,
            elite=er["all_pass"],
            is_multi_source_validated=s.is_multi_source_validated,
            ibd_lists=s.ibd_lists,
            schwab_themes=s.schwab_themes,
            morningstar_rating=s.morningstar_rating,
            economic_moat=s.economic_moat,
        )
        strengths = generate_strengths(s, er["p_eps"], er["p_rs"], er["p_smr"], er["p_acc_dis"])
        weaknesses = generate_weaknesses(s, er["p_eps"], er["p_rs"], er["p_smr"], er["p_acc_dis"])
        catalyst = generate_catalyst(s)

        # Compute valuation & risk metrics
        val_metrics = compute_all_valuation_metrics(
            sector=s.sector,
            composite_rating=s.composite_rating,
            rs_rating=s.rs_rating,
            eps_rating=s.eps_rating,
        )

        stock_analysis.append({
            "stock": s,
            "tier": tier,
            "conviction": conviction,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "catalyst": catalyst,
            "elite": er,
            "sector_rank_in_sector": sector_rank_map.get(s.symbol, 1),
            "valuation": val_metrics,
        })

    # --- Step 8: Sort all tiered stocks ---
    # Sort: tier ASC, conviction DESC, validation_score DESC
    stock_analysis.sort(
        key=lambda x: (x["tier"], -x["conviction"], -x["stock"].validation_score)
    )

    # Build RatedStock objects for ALL tiered stocks
    rated_stocks: list[RatedStock] = []
    for item in stock_analysis:
        s = item["stock"]
        tier = item["tier"]
        er = item["elite"]

        # Build reasoning (>=50 chars)
        tier_label = TIER_LABELS[tier]
        reasoning_parts = [
            f"T{tier} {tier_label} stock",
            f"Composite {s.composite_rating}, RS {s.rs_rating}, EPS {s.eps_rating}",
        ]
        if er["all_pass"]:
            reasoning_parts.append("passes all 4 elite filters")
        else:
            failed = []
            if not er["p_eps"]:
                failed.append("EPS")
            if not er["p_rs"]:
                failed.append("RS")
            if er["p_smr"] is False:
                failed.append("SMR")
            if er["p_acc_dis"] is False:
                failed.append("Acc/Dis")
            if er["p_smr"] is None:
                failed.append("SMR(missing)")
            if er["p_acc_dis"] is None:
                failed.append("Acc/Dis(missing)")
            if failed:
                reasoning_parts.append(f"elite filters failed: {', '.join(failed)}")
        if s.ibd_lists:
            reasoning_parts.append(f"on {len(s.ibd_lists)} IBD lists: {', '.join(s.ibd_lists[:3])}")
        if s.schwab_themes:
            reasoning_parts.append(f"themes: {', '.join(s.schwab_themes[:2])}")
        if s.morningstar_rating:
            reasoning_parts.append(f"Morningstar {s.morningstar_rating}, moat: {s.economic_moat or 'N/A'}")
        if s.price_to_fair_value is not None and s.price_to_fair_value < 1.0:
            reasoning_parts.append(f"trading below fair value (P/FV {s.price_to_fair_value:.2f})")

        reasoning = ". ".join(reasoning_parts)
        if len(reasoning) < 50:
            reasoning += ". Awaiting deeper fundamental and technical analysis for final assessment"

        try:
            vm = item["valuation"]
            rs = RatedStock(
                symbol=s.symbol,
                company_name=s.company_name,
                sector=s.sector,
                tier=tier,
                tier_label=tier_label,
                composite_rating=s.composite_rating,
                rs_rating=s.rs_rating,
                eps_rating=s.eps_rating,
                smr_rating=s.smr_rating,
                acc_dis_rating=s.acc_dis_rating,
                passes_eps_filter=er["p_eps"],
                passes_rs_filter=er["p_rs"],
                passes_smr_filter=er["p_smr"],
                passes_acc_dis_filter=er["p_acc_dis"],
                passes_all_elite=er["all_pass"],
                is_ibd_keep=is_ibd_keep_candidate(s.composite_rating, s.rs_rating),
                is_multi_source_validated=s.is_multi_source_validated,
                ibd_lists=s.ibd_lists,
                schwab_themes=s.schwab_themes,
                validation_score=s.validation_score,
                cap_size=s.cap_size,
                morningstar_rating=s.morningstar_rating,
                economic_moat=s.economic_moat,
                fair_value=s.fair_value,
                morningstar_price=s.morningstar_price,
                price_to_fair_value=s.price_to_fair_value,
                morningstar_uncertainty=s.morningstar_uncertainty,
                conviction=item["conviction"],
                strengths=item["strengths"],
                weaknesses=item["weaknesses"],
                catalyst=item["catalyst"],
                reasoning=reasoning,
                sector_rank_in_sector=item["sector_rank_in_sector"],
                estimated_pe=vm["estimated_pe"],
                pe_category=vm["pe_category"],
                peg_ratio=vm["peg_ratio"],
                peg_category=vm["peg_category"],
                estimated_beta=vm["estimated_beta"],
                estimated_return_pct=vm["estimated_return_pct"],
                return_category=vm["return_category"],
                estimated_volatility_pct=vm["estimated_volatility_pct"],
                sharpe_ratio=vm["sharpe_ratio"],
                sharpe_category=vm["sharpe_category"],
                alpha_pct=vm["alpha_pct"],
                alpha_category=vm["alpha_category"],
                risk_rating=vm["risk_rating"],
            )
            rated_stocks.append(rs)
        except Exception as e:
            logger.warning(f"Skipping rated stock {s.symbol}: {e}")

    # --- Step 9: IBD Keeps (from ALL ratable, not just top 50) ---
    ibd_keeps: list[IBDKeep] = []
    for s in ratable:
        if not is_ibd_keep_candidate(s.composite_rating, s.rs_rating):
            continue
        tier = tier_map[s.symbol]
        if tier is None:
            continue

        rationale = (
            f"Composite {s.composite_rating} (>= 93), RS {s.rs_rating} (>= 90)"
        )
        if s.ibd_lists:
            rationale += f" — on {', '.join(s.ibd_lists[:3])}"

        override_risk = None
        if s.eps_rating is not None and s.eps_rating < 70:
            override_risk = f"Low EPS rating ({s.eps_rating}) may indicate earnings weakness"
        er = elite_results.get(s.symbol, {})
        if er.get("p_smr") is False:
            override_risk = f"Weak SMR rating ({s.smr_rating}) despite strong Comp/RS"

        try:
            keep = IBDKeep(
                symbol=s.symbol,
                composite_rating=s.composite_rating,
                rs_rating=s.rs_rating,
                eps_rating=s.eps_rating,
                ibd_lists=s.ibd_lists,
                tier=tier,
                keep_rationale=rationale,
                override_risk=override_risk,
            )
            ibd_keeps.append(keep)
        except Exception as e:
            logger.warning(f"Skipping IBD keep {s.symbol}: {e}")

    # --- Build UnratedStock objects ---
    unrated_stocks: list[UnratedStock] = []
    for s in unratable:
        missing = []
        if s.composite_rating is None:
            missing.append("Composite")
        if s.rs_rating is None:
            missing.append("RS")
        if s.eps_rating is None:
            missing.append("EPS")

        note = "Requires external rating lookup or manual assessment for tier placement"
        if s.schwab_themes:
            note += f". Has thematic exposure: {', '.join(s.schwab_themes[:2])}"
        if s.is_multi_source_validated:
            note += f". Multi-source validated (score {s.validation_score})"

        try:
            us = UnratedStock(
                symbol=s.symbol,
                company_name=s.company_name,
                sector=s.sector if s.sector != "UNKNOWN" else None,
                reason_unrated=f"Missing IBD ratings: {', '.join(missing)}",
                schwab_themes=s.schwab_themes,
                validation_score=s.validation_score,
                sources=s.sources,
                note=note,
            )
            unrated_stocks.append(us)
        except Exception as e:
            logger.warning(f"Skipping unrated stock {s.symbol}: {e}")

    # --- Step 10: Tier Distribution and Elite Filter Summary ---
    elite_summary = EliteFilterSummary(
        total_screened=len(ratable),
        passed_all_four=passed_all,
        failed_eps=failed_eps,
        failed_rs=failed_rs,
        failed_smr=failed_smr,
        failed_acc_dis=failed_acc_dis,
        missing_ratings=missing_ratings,
    )

    tier_dist = TierDistribution(
        tier_1_count=t1_count,
        tier_2_count=t2_count,
        tier_3_count=t3_count,
        below_threshold_count=below_count,
        unrated_count=len(unratable),
    )

    # --- Step 11: Valuation summary + sector enrichment ---
    valuation_summary = None
    if rated_stocks:
        vs_dict = compute_valuation_summary(rated_stocks)

        # Enrich sector rankings with avg PE/beta/volatility
        sector_metrics: dict[str, dict] = {}
        for s in rated_stocks:
            sec = s.sector
            if sec not in sector_metrics:
                sector_metrics[sec] = {"pe": [], "beta": [], "vol": []}
            if s.estimated_pe is not None:
                sector_metrics[sec]["pe"].append(s.estimated_pe)
            if s.estimated_beta is not None:
                sector_metrics[sec]["beta"].append(s.estimated_beta)
            if s.estimated_volatility_pct is not None:
                sector_metrics[sec]["vol"].append(s.estimated_volatility_pct)

        import statistics as _stats
        for sr in sector_rankings:
            sm = sector_metrics.get(sr.sector, {})
            if sm.get("pe"):
                sr.avg_pe = round(_stats.mean(sm["pe"]), 1)
            if sm.get("beta"):
                sr.avg_beta = round(_stats.mean(sm["beta"]), 2)
            if sm.get("vol"):
                sr.avg_volatility = round(_stats.mean(sm["vol"]), 1)

        valuation_summary = ValuationSummary(
            market_context=MarketContext(**vs_dict["market_context"]),
            pe_distribution=PEDistribution(**vs_dict["pe_distribution"]),
            tier_pe_stats=[TierPEStats(**t) for t in vs_dict["tier_pe_stats"]],
            top_sectors_by_pe=[SectorPEEntry(**s) for s in vs_dict["top_sectors_by_pe"]],
            peg_analysis=PEGAnalysis(**vs_dict["peg_analysis"]),
            avg_sharpe=vs_dict["avg_sharpe"],
            avg_beta=vs_dict["avg_beta"],
            avg_alpha=vs_dict["avg_alpha"],
        )

    # --- Step 11.5: LLM valuation enrichment ---
    try:
        stock_inputs = [
            {
                "symbol": s.symbol,
                "sector": s.sector,
                "company_name": s.company_name,
                "composite_rating": s.composite_rating,
                "rs_rating": s.rs_rating,
                "eps_rating": s.eps_rating,
                "tier": s.tier,
            }
            for s in rated_stocks
        ]
        llm_data = enrich_valuation_llm(stock_inputs)

        enriched_count = 0
        for s in rated_stocks:
            if s.symbol in llm_data:
                d = llm_data[s.symbol]

                # Override estimated metrics with LLM data where available
                if d.get("pe_ratio") is not None:
                    s.estimated_pe = d["pe_ratio"]
                    s.pe_category = classify_pe_category(d["pe_ratio"], s.sector)
                    s.peg_ratio = calculate_peg(d["pe_ratio"], s.eps_rating)
                    s.peg_category = classify_peg_category(s.peg_ratio)
                if d.get("beta") is not None:
                    s.estimated_beta = d["beta"]
                if d.get("annual_return_1y") is not None:
                    s.estimated_return_pct = d["annual_return_1y"]
                    s.return_category = classify_return_category(d["annual_return_1y"])
                if d.get("volatility") is not None:
                    s.estimated_volatility_pct = d["volatility"]

                # Store LLM-specific fields
                s.llm_pe_ratio = d.get("pe_ratio")
                s.llm_forward_pe = d.get("forward_pe")
                s.llm_beta = d.get("beta")
                s.llm_annual_return_1y = d.get("annual_return_1y")
                s.llm_volatility = d.get("volatility")
                s.llm_dividend_yield = d.get("dividend_yield")
                s.llm_market_cap_b = d.get("market_cap_b")
                s.llm_valuation_grade = d.get("valuation_grade")
                s.llm_risk_grade = d.get("risk_grade")
                s.llm_guidance = d.get("guidance")
                s.valuation_source = "llm"
                enriched_count += 1
            else:
                s.valuation_source = "estimated"

        # Recompute derived metrics for LLM-enriched stocks
        for s in rated_stocks:
            if s.valuation_source == "llm":
                ret = s.estimated_return_pct if s.estimated_return_pct is not None else estimate_return_pct(s.rs_rating)
                vol = s.estimated_volatility_pct if s.estimated_volatility_pct is not None else 28.0
                beta = s.estimated_beta if s.estimated_beta is not None else 1.0
                s.sharpe_ratio = calculate_sharpe(ret, vol)
                s.sharpe_category = classify_sharpe_category(s.sharpe_ratio)
                s.alpha_pct = calculate_alpha(ret, beta)
                s.alpha_category = classify_alpha_category(s.alpha_pct)
                s.risk_rating = calculate_risk_rating(s.sharpe_ratio, beta)

        # Recompute valuation summary with LLM data
        if enriched_count > 0 and rated_stocks:
            vs_dict = compute_valuation_summary(rated_stocks)
            valuation_summary = ValuationSummary(
                market_context=MarketContext(**vs_dict["market_context"]),
                pe_distribution=PEDistribution(**vs_dict["pe_distribution"]),
                tier_pe_stats=[TierPEStats(**t) for t in vs_dict["tier_pe_stats"]],
                top_sectors_by_pe=[SectorPEEntry(**s_entry) for s_entry in vs_dict["top_sectors_by_pe"]],
                peg_analysis=PEGAnalysis(**vs_dict["peg_analysis"]),
                avg_sharpe=vs_dict["avg_sharpe"],
                avg_beta=vs_dict["avg_beta"],
                avg_alpha=vs_dict["avg_alpha"],
            )

            # Re-enrich sector rankings with updated metrics
            sector_metrics_updated: dict[str, dict] = {}
            for s in rated_stocks:
                sec = s.sector
                if sec not in sector_metrics_updated:
                    sector_metrics_updated[sec] = {"pe": [], "beta": [], "vol": []}
                if s.estimated_pe is not None:
                    sector_metrics_updated[sec]["pe"].append(s.estimated_pe)
                if s.estimated_beta is not None:
                    sector_metrics_updated[sec]["beta"].append(s.estimated_beta)
                if s.estimated_volatility_pct is not None:
                    sector_metrics_updated[sec]["vol"].append(s.estimated_volatility_pct)

            import statistics as _stats2
            for sr in sector_rankings:
                sm = sector_metrics_updated.get(sr.sector, {})
                if sm.get("pe"):
                    sr.avg_pe = round(_stats2.mean(sm["pe"]), 1)
                if sm.get("beta"):
                    sr.avg_beta = round(_stats2.mean(sm["beta"]), 2)
                if sm.get("vol"):
                    sr.avg_volatility = round(_stats2.mean(sm["vol"]), 1)

        logger.info(f"LLM enriched {enriched_count}/{len(rated_stocks)} stocks")
    except Exception as e:
        logger.warning(f"LLM valuation enrichment failed, using deterministic estimates: {e}")
        # Mark all stocks as "estimated" source
        for s in rated_stocks:
            if s.valuation_source is None:
                s.valuation_source = "estimated"

    # --- Step 11.6: LLM catalyst enrichment ---
    try:
        from ibd_agents.tools.catalyst_enrichment import (
            compute_catalyst_adjustment,
            enrich_catalyst_llm,
        )
        catalyst_inputs = [
            {
                "symbol": s.symbol,
                "sector": s.sector,
                "company_name": s.company_name,
                "composite_rating": s.composite_rating,
                "rs_rating": s.rs_rating,
                "eps_rating": s.eps_rating,
                "tier": s.tier,
            }
            for s in rated_stocks
        ]
        catalyst_data = enrich_catalyst_llm(catalyst_inputs)

        catalyst_count = 0
        for s in rated_stocks:
            if s.symbol in catalyst_data:
                d = catalyst_data[s.symbol]
                s.catalyst_date = d.get("catalyst_date")
                s.catalyst_type = d.get("catalyst_type")
                adjustment = compute_catalyst_adjustment(d.get("days_until"))
                s.catalyst_conviction_adjustment = adjustment
                if adjustment != 0:
                    s.conviction = min(10, max(1, s.conviction + adjustment))
                desc = d.get("catalyst_description")
                if desc:
                    s.catalyst = desc
                s.catalyst_source = "llm"
                catalyst_count += 1
            else:
                s.catalyst_source = "template"

        logger.info(f"LLM catalyst enriched {catalyst_count}/{len(rated_stocks)} stocks")
    except Exception as e:
        logger.warning(f"LLM catalyst enrichment failed, using template catalysts: {e}")
        for s in rated_stocks:
            if s.catalyst_source is None:
                s.catalyst_source = "template"

    # --- Step 12: Process ETFs ---
    rated_etfs, etf_tier_dist = _build_rated_etfs(research_output.etfs)
    logger.info(f"[Agent 02] Rated {len(rated_etfs)} ETFs across tiers")

    # --- Step 13: Assemble output ---
    methodology = (
        "Framework v4.0 Elite Screening (EPS>=85, RS>=75, SMR A/B/B-, Acc/Dis A/B/B-). "
        "Tier assignment: T1 Momentum (Comp>=95, RS>=90, EPS>=80), "
        "T2 Quality Growth (>=85, >=80, >=75), T3 Defensive (>=80, >=75, >=70). "
        "IBD Keep: Comp>=93 AND RS>=90. "
        "Sector score: (avg_comp*0.25)+(avg_rs*0.25)+(elite_pct*0.30)+(multi_list_pct*0.20). "
        "Conviction: 1-10 deterministic (tier base + elite/validation/list/theme bonuses). "
        "Valuation metrics estimated from sector baselines + IBD ratings (no external API). "
        "P/E from sector ranges and growth factor. PEG = P/E / (EPS * 0.5). "
        "Beta adjusted from sector baseline by RS/Composite momentum. "
        "Sharpe = (Return - 4.5%) / Volatility. Alpha = Return - CAPM expected. "
        "Risk Rating combines Sharpe and Beta."
    )

    top_sector = sector_rankings[0].sector if sector_rankings else "N/A"
    summary = (
        f"Analyzed {len(ratable)} ratable stocks and {len(unratable)} unrated. "
        f"Tier distribution: {t1_count} T1 Momentum, {t2_count} T2 Quality Growth, "
        f"{t3_count} T3 Defensive, {below_count} below threshold. "
        f"{passed_all} passed all 4 elite filters. "
        f"{len(ibd_keeps)} IBD keep candidates identified. "
        f"Top sector: {top_sector} (score {sector_rankings[0].sector_score:.1f})."
        if sector_rankings else
        f"Analyzed {len(ratable)} ratable stocks. No sector rankings available."
    )

    output = AnalystOutput(
        rated_stocks=rated_stocks,
        unrated_stocks=unrated_stocks,
        sector_rankings=sector_rankings,
        ibd_keeps=ibd_keeps,
        elite_filter_summary=elite_summary,
        tier_distribution=tier_dist,
        methodology_notes=methodology,
        analysis_date=date.today().isoformat(),
        summary=summary,
        rated_etfs=rated_etfs,
        etf_tier_distribution=etf_tier_dist,
        valuation_summary=valuation_summary,
    )

    return output


# ---------------------------------------------------------------------------
# ETF Rating Helpers
# ---------------------------------------------------------------------------

def _build_rated_etfs(etfs: list) -> tuple[list[RatedETF], "ETFTierDistribution"]:
    """Build RatedETF objects with conviction, screening, and focus ranking."""
    from ibd_agents.schemas.analyst_output import ETFTierDistribution
    from ibd_agents.tools.elite_screener import (
        calculate_etf_conviction,
        passes_etf_acc_dis_screen,
        passes_etf_rs_screen,
        passes_etf_screen,
    )

    rated: list[RatedETF] = []
    for etf in etfs:
        if etf.etf_score is None:
            continue

        strengths = _etf_strengths(etf)
        weaknesses = _etf_weaknesses(etf)
        tier = etf.preliminary_tier or 2

        p_rs = passes_etf_rs_screen(etf.rs_rating)
        p_acc = passes_etf_acc_dis_screen(etf.acc_dis_rating)
        all_screen = passes_etf_screen(etf.rs_rating, etf.acc_dis_rating)

        conviction = calculate_etf_conviction(
            tier=tier,
            rs_rating=etf.rs_rating,
            acc_dis_rating=etf.acc_dis_rating,
            ytd_change=etf.ytd_change,
            schwab_themes=etf.schwab_themes,
            etf_score=etf.etf_score,
        )

        rated.append(RatedETF(
            symbol=etf.symbol,
            name=etf.name,
            tier=tier,
            rs_rating=etf.rs_rating,
            acc_dis_rating=etf.acc_dis_rating,
            ytd_change=etf.ytd_change,
            volume_pct_change=etf.volume_pct_change,
            price_change=etf.price_change,
            close_price=etf.close_price,
            div_yield=etf.div_yield,
            etf_score=etf.etf_score,
            etf_rank=etf.etf_rank or 999,
            theme_tags=etf.schwab_themes,
            strengths=strengths,
            weaknesses=weaknesses,
            conviction=conviction,
            focus=etf.focus,
            passes_rs_screen=p_rs,
            passes_acc_dis_screen=p_acc,
            passes_etf_screen=all_screen,
        ))

    # Focus group ranking — rank ETFs within each focus area by etf_score
    focus_groups: dict[str, list[RatedETF]] = {}
    for e in rated:
        key = e.focus or "General"
        focus_groups.setdefault(key, []).append(e)
    for key, group in focus_groups.items():
        group.sort(key=lambda e: e.etf_score, reverse=True)
        for i, e in enumerate(group, 1):
            e.focus_group = key
            e.focus_rank = i

    # ETF tier distribution
    t1 = sum(1 for e in rated if e.tier == 1)
    t2 = sum(1 for e in rated if e.tier == 2)
    t3 = sum(1 for e in rated if e.tier == 3)
    etf_dist = ETFTierDistribution(
        tier_1_count=t1, tier_2_count=t2, tier_3_count=t3,
        total_rated=len(rated),
    )

    return rated, etf_dist


def _etf_strengths(etf) -> list[str]:
    """Build strengths list from ETF metrics."""
    strengths = []
    if etf.rs_rating is not None:
        if etf.rs_rating >= 85:
            strengths.append(f"RS {etf.rs_rating} — strong relative momentum")
        elif etf.rs_rating >= 70:
            strengths.append(f"RS {etf.rs_rating} — solid momentum")
    if etf.acc_dis_rating in ("A", "B"):
        strengths.append(f"Acc/Dis {etf.acc_dis_rating} — institutional accumulation")
    if etf.ytd_change is not None and etf.ytd_change > 10:
        strengths.append(f"YTD +{etf.ytd_change:.1f}% — strong year-to-date performance")
    if etf.volume_pct_change is not None and etf.volume_pct_change > 20:
        strengths.append(f"Volume +{etf.volume_pct_change}% — rising interest")
    if etf.div_yield is not None and etf.div_yield > 2.0:
        strengths.append(f"Div yield {etf.div_yield:.1f}% — income component")
    if not strengths:
        strengths.append("Diversified ETF exposure")
    return strengths


def _etf_weaknesses(etf) -> list[str]:
    """Build weaknesses list from ETF metrics."""
    weaknesses = []
    if etf.rs_rating is not None and etf.rs_rating < 50:
        weaknesses.append(f"RS {etf.rs_rating} — weak relative momentum")
    if etf.acc_dis_rating in ("D", "E"):
        weaknesses.append(f"Acc/Dis {etf.acc_dis_rating} — institutional distribution")
    if etf.ytd_change is not None and etf.ytd_change < -5:
        weaknesses.append(f"YTD {etf.ytd_change:.1f}% — negative year-to-date")
    if etf.volume_pct_change is not None and etf.volume_pct_change < -30:
        weaknesses.append(f"Volume {etf.volume_pct_change}% — declining interest")
    if not weaknesses:
        weaknesses.append("Sector-specific concentration risk")
    return weaknesses
