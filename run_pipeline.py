"""Run the Research + Analyst Pipeline and write output to Excel."""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from openpyxl.styles import Font, PatternFill
from ibd_agents.agents.research_agent import run_research_pipeline
from ibd_agents.agents.analyst_agent import run_analyst_pipeline
from ibd_agents.agents.rotation_detector import run_rotation_pipeline
from ibd_agents.agents.sector_strategist import run_strategist_pipeline
from ibd_agents.agents.portfolio_manager import run_portfolio_pipeline
from ibd_agents.agents.risk_officer import run_risk_pipeline
from ibd_agents.agents.returns_projector import run_returns_projector_pipeline
from ibd_agents.agents.portfolio_reconciler import run_reconciler_pipeline
from ibd_agents.agents.educator_agent import run_educator_pipeline
from ibd_agents.agents.executive_synthesizer import run_synthesizer_pipeline


def _write_research_excel(research_output, out_path: Path) -> Path:
    """Write Agent 01 Research output to its own Excel file."""
    today = date.today().isoformat()
    filepath = out_path / f"agent01_research_{today}.xlsx"

    # --- Stocks (split known sectors vs UNKNOWN) ---
    known_rows = []
    unknown_rows = []
    for s in research_output.stocks:
        row = {
            "Sector Rank": s.sector_rank,
            "Rank in Sector": s.stock_rank_in_sector,
            "Symbol": s.symbol,
            "Company": s.company_name,
            "Sector": s.sector,
            "Composite": s.composite_rating,
            "RS": s.rs_rating,
            "EPS": s.eps_rating,
            "SMR": s.smr_rating,
            "Acc/Dis": s.acc_dis_rating,
            "Tier": s.preliminary_tier,
            "Keep Candidate": s.is_ibd_keep_candidate,
            "Multi-Source": s.is_multi_source_validated,
            "Validation Score": s.validation_score,
            "Providers": s.validation_providers,
            "IBD Lists": ", ".join(s.ibd_lists),
            "Schwab Themes": ", ".join(s.schwab_themes),
            "Fool Status": s.fool_status,
            "M* Rating": s.morningstar_rating,
            "Moat": s.economic_moat,
            "Fair Value": s.fair_value,
            "M* Price": s.morningstar_price,
            "P/FV": s.price_to_fair_value,
            "Uncertainty": s.morningstar_uncertainty,
            "Confidence": s.confidence,
            "Reasoning": s.reasoning,
            "Sources": ", ".join(s.sources),
        }
        if s.sector == "UNKNOWN":
            unknown_rows.append(row)
        else:
            known_rows.append(row)

    df_stocks = pd.DataFrame(known_rows)
    if not df_stocks.empty:
        df_stocks = df_stocks.sort_values(
            by=["Sector", "Composite"],
            ascending=[True, False],
            na_position="last",
        ).reset_index(drop=True)

    df_unknown = pd.DataFrame(unknown_rows)

    # --- ETFs ---
    etf_rows = []
    for e in research_output.etfs:
        etf_rows.append({
            "Rank": e.etf_rank,
            "Symbol": e.symbol,
            "Name": e.name,
            "Tier": e.preliminary_tier,
            "RS Rating": e.rs_rating,
            "Acc/Dis": e.acc_dis_rating,
            "YTD Chg %": e.ytd_change,
            "Price Chg": e.price_change,
            "Vol % Chg": e.volume_pct_change,
            "Close": e.close_price,
            "Div Yield": e.div_yield,
            "ETF Score": e.etf_score,
            "Focus": e.focus,
            "Schwab Themes": ", ".join(e.schwab_themes),
            "IBD Lists": ", ".join(e.ibd_lists),
            "Key Theme ETF": e.key_theme_etf,
            "Sources": ", ".join(e.sources),
        })
    df_etfs = pd.DataFrame(etf_rows)
    if not df_etfs.empty:
        df_etfs = df_etfs.sort_values(
            by=["Rank"], na_position="last",
        ).reset_index(drop=True)

    # --- Sector Patterns ---
    sector_pattern_rows = []
    for sp in research_output.sector_patterns:
        sector_pattern_rows.append({
            "Sector": sp.sector,
            "Stock Count": sp.stock_count,
            "Avg Composite": sp.avg_composite,
            "Avg RS": sp.avg_rs,
            "Elite %": sp.elite_pct,
            "Multi-List %": sp.multi_list_pct,
            "Keep Count": sp.ibd_keep_count,
            "Strength": sp.strength,
            "Trend": sp.trend_direction,
            "Evidence": sp.evidence,
        })
    df_sector_patterns = pd.DataFrame(sector_pattern_rows)

    # --- Summary ---
    summary_rows = [
        {"Field": "Analysis Date", "Value": research_output.analysis_date},
        {"Field": "Total Stocks", "Value": len(research_output.stocks)},
        {"Field": "Total ETFs", "Value": len(research_output.etfs)},
        {"Field": "Sectors", "Value": len(research_output.sector_patterns)},
        {"Field": "IBD Keep Candidates", "Value": len(research_output.ibd_keep_candidates)},
        {"Field": "Multi-Source Validated", "Value": len(research_output.multi_source_validated)},
        {"Field": "Total Securities Scanned", "Value": research_output.total_securities_scanned},
        {"Field": "Data Sources Used", "Value": ", ".join(research_output.data_sources_used)},
        {"Field": "Data Sources Failed", "Value": ", ".join(research_output.data_sources_failed) or "None"},
        {"Field": "Summary", "Value": research_output.summary},
    ]
    df_summary = pd.DataFrame(summary_rows)

    # --- Write ---
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_stocks.to_excel(writer, sheet_name="Stocks", index=False)
        if not df_unknown.empty:
            df_unknown.to_excel(writer, sheet_name="Unknown Sector", index=False)
        if not df_etfs.empty:
            df_etfs.to_excel(writer, sheet_name="ETFs", index=False)
        if not df_sector_patterns.empty:
            df_sector_patterns.to_excel(writer, sheet_name="Sector Patterns", index=False)

    return filepath


def _write_analyst_excel(analyst_output, out_path: Path, research_output=None) -> Path:
    """Write Agent 02 Analyst output to its own Excel file."""
    today = date.today().isoformat()
    filepath = out_path / f"agent02_analyst_{today}.xlsx"

    # --- Top 100 + per-tier sheets ---
    def _elite_detail(passed, rating, threshold, label):
        """Format elite filter result: PASS/FAIL with rating vs threshold."""
        if passed is None:
            return f"N/A — {label} missing"
        elif passed:
            return f"PASS — {label} {rating} (>= {threshold})"
        else:
            return f"FAIL — {label} {rating} (< {threshold})"

    def _stock_row(s, rank):
        return {
            "Rank": rank,
            "Symbol": s.symbol,
            "Company": s.company_name,
            "Sector": s.sector,
            "Tier": s.tier,
            "Tier Label": s.tier_label,
            "Conviction": s.conviction,
            "Composite": s.composite_rating,
            "RS": s.rs_rating,
            "EPS": s.eps_rating,
            "SMR": s.smr_rating,
            "Acc/Dis": s.acc_dis_rating,
            "All Elite Pass": "PASS" if s.passes_all_elite else "FAIL",
            "EPS Elite (>=85)": _elite_detail(s.passes_eps_filter, s.eps_rating, 85, "EPS"),
            "RS Elite (>=75)": _elite_detail(s.passes_rs_filter, s.rs_rating, 75, "RS"),
            "SMR Elite (A/B/B-)": _elite_detail(s.passes_smr_filter, s.smr_rating, "A/B/B-", "SMR"),
            "AccDis Elite (A/B/B-)": _elite_detail(s.passes_acc_dis_filter, s.acc_dis_rating, "A/B/B-", "Acc/Dis"),
            "IBD Keep": s.is_ibd_keep,
            "Multi-Source": s.is_multi_source_validated,
            "Validation Score": s.validation_score,
            "Sector Rank": s.sector_rank_in_sector,
            "Est P/E": s.estimated_pe,
            "P/E Category": s.pe_category,
            "PEG Ratio": s.peg_ratio,
            "PEG Category": s.peg_category,
            "Est Beta": s.estimated_beta,
            "Est Return %": s.estimated_return_pct,
            "Return Cat": s.return_category,
            "Volatility %": s.estimated_volatility_pct,
            "Sharpe Ratio": s.sharpe_ratio,
            "Sharpe Cat": s.sharpe_category,
            "Alpha %": s.alpha_pct,
            "Alpha Cat": s.alpha_category,
            "Risk Rating": s.risk_rating,
            "LLM P/E": s.llm_pe_ratio,
            "LLM Fwd P/E": s.llm_forward_pe,
            "LLM Beta": s.llm_beta,
            "LLM 1Y Return %": s.llm_annual_return_1y,
            "LLM Volatility %": s.llm_volatility,
            "Div Yield %": s.llm_dividend_yield,
            "Mkt Cap ($B)": s.llm_market_cap_b,
            "LLM Valuation": s.llm_valuation_grade,
            "LLM Risk": s.llm_risk_grade,
            "Guidance": s.llm_guidance,
            "Data Source": s.valuation_source,
            "Strengths": "; ".join(s.strengths),
            "Weaknesses": "; ".join(s.weaknesses),
            "Catalyst": s.catalyst,
            "Catalyst Date": s.catalyst_date,
            "Catalyst Type": s.catalyst_type,
            "Catalyst Adj": s.catalyst_conviction_adjustment,
            "Catalyst Source": s.catalyst_source,
            "IBD Lists": ", ".join(s.ibd_lists),
            "Schwab Themes": ", ".join(s.schwab_themes),
            "M* Rating": s.morningstar_rating,
            "Moat": s.economic_moat,
            "Fair Value": s.fair_value,
            "M* Price": s.morningstar_price,
            "P/FV": s.price_to_fair_value,
            "Uncertainty": s.morningstar_uncertainty,
        }

    t1_rows, t2_rows, t3_rows = [], [], []
    for s in analyst_output.rated_stocks:
        if s.tier == 1:
            t1_rows.append(_stock_row(s, len(t1_rows) + 1))
        elif s.tier == 2:
            t2_rows.append(_stock_row(s, len(t2_rows) + 1))
        elif s.tier == 3:
            t3_rows.append(_stock_row(s, len(t3_rows) + 1))

    # Top 100 = first 100 across all tiers (already sorted by tier, conviction, score)
    top100_rows = []
    for i, s in enumerate(analyst_output.rated_stocks[:100], 1):
        top100_rows.append(_stock_row(s, i))

    df_top100 = pd.DataFrame(top100_rows)
    df_t1 = pd.DataFrame(t1_rows)
    df_t2 = pd.DataFrame(t2_rows)
    df_t3 = pd.DataFrame(t3_rows)

    # --- Sector Rankings ---
    sector_rank_rows = []
    for sr in analyst_output.sector_rankings:
        sector_rank_rows.append({
            "Rank": sr.rank,
            "Sector": sr.sector,
            "Score": sr.sector_score,
            "Stock Count": sr.stock_count,
            "Avg Composite": sr.avg_composite,
            "Avg RS": sr.avg_rs,
            "Elite %": sr.elite_pct,
            "Multi-List %": sr.multi_list_pct,
            "IBD Keeps": sr.ibd_keep_count,
            "Avg P/E": sr.avg_pe,
            "Avg Beta": sr.avg_beta,
            "Avg Volatility": sr.avg_volatility,
            "Top Stocks": ", ".join(sr.top_stocks),
        })
    df_sector_ranks = pd.DataFrame(sector_rank_rows)

    # --- IBD Keeps ---
    keep_rows = []
    for k in analyst_output.ibd_keeps:
        keep_rows.append({
            "Symbol": k.symbol,
            "Composite": k.composite_rating,
            "RS": k.rs_rating,
            "EPS": k.eps_rating,
            "Tier": k.tier,
            "IBD Lists": ", ".join(k.ibd_lists),
            "Rationale": k.keep_rationale,
            "Override Risk": k.override_risk or "",
        })
    df_keeps = pd.DataFrame(keep_rows)

    # --- Unrated ---
    unrated_rows = []
    for u in analyst_output.unrated_stocks:
        unrated_rows.append({
            "Symbol": u.symbol,
            "Company": u.company_name,
            "Sector": u.sector or "",
            "Reason": u.reason_unrated,
            "Schwab Themes": ", ".join(u.schwab_themes),
            "Validation Score": u.validation_score,
            "Note": u.note,
        })
    df_unrated = pd.DataFrame(unrated_rows)

    # --- Summary ---
    td = analyst_output.tier_distribution
    ef = analyst_output.elite_filter_summary
    summary_rows = [
        {"Field": "Analysis Date", "Value": analyst_output.analysis_date},
        {"Field": "Total Rated Stocks", "Value": len(analyst_output.rated_stocks)},
        {"Field": "Unrated Stocks", "Value": len(analyst_output.unrated_stocks)},
        {"Field": "IBD Keeps", "Value": len(analyst_output.ibd_keeps)},
        {"Field": "Sectors Ranked", "Value": len(analyst_output.sector_rankings)},
        {"Field": "", "Value": ""},
        {"Field": "--- Tier Distribution ---", "Value": ""},
        {"Field": "T1 Momentum", "Value": td.tier_1_count},
        {"Field": "T2 Quality Growth", "Value": td.tier_2_count},
        {"Field": "T3 Defensive", "Value": td.tier_3_count},
        {"Field": "Below Threshold", "Value": td.below_threshold_count},
        {"Field": "Unrated", "Value": td.unrated_count},
        {"Field": "", "Value": ""},
        {"Field": "--- Elite Screening ---", "Value": ""},
        {"Field": "Total Screened", "Value": ef.total_screened},
        {"Field": "Passed All 4", "Value": ef.passed_all_four},
        {"Field": "Failed EPS", "Value": ef.failed_eps},
        {"Field": "Failed RS", "Value": ef.failed_rs},
        {"Field": "Failed SMR", "Value": ef.failed_smr},
        {"Field": "Failed Acc/Dis", "Value": ef.failed_acc_dis},
        {"Field": "Missing Ratings", "Value": ef.missing_ratings},
    ]

    # Valuation summary sections
    vs = analyst_output.valuation_summary
    if vs:
        mc = vs.market_context
        summary_rows.extend([
            {"Field": "", "Value": ""},
            {"Field": "--- Market Context ---", "Value": ""},
            {"Field": "S&P 500 Forward P/E", "Value": f"{mc.sp500_forward_pe}x"},
            {"Field": "10-Year Average P/E", "Value": f"{mc.sp500_10y_avg_pe}x"},
            {"Field": "Current Premium", "Value": f"+{mc.current_premium_pct}%"},
            {"Field": "Risk-Free Rate", "Value": f"{mc.risk_free_rate}%"},
            {"Field": "Market Return Assumption", "Value": f"{mc.market_return_assumption}%"},
        ])

        # P/E Statistics by Tier
        summary_rows.extend([
            {"Field": "", "Value": ""},
            {"Field": "--- P/E Statistics by Tier ---", "Value": ""},
        ])
        for t in vs.tier_pe_stats:
            label = {1: "T1 Momentum", 2: "T2 Quality Growth", 3: "T3 Defensive"}.get(t.tier, f"T{t.tier}")
            summary_rows.extend([
                {"Field": f"{label} Avg P/E", "Value": f"{t.avg_pe:.1f}x"},
                {"Field": f"{label} Median P/E", "Value": f"{t.median_pe:.1f}x"},
                {"Field": f"{label} Range", "Value": f"{t.min_pe:.1f}x - {t.max_pe:.1f}x ({t.stock_count} stocks)"},
            ])

        # P/E Distribution
        pd_dist = vs.pe_distribution
        total = pd_dist.total or 1
        summary_rows.extend([
            {"Field": "", "Value": ""},
            {"Field": "--- P/E Distribution ---", "Value": ""},
            {"Field": "Deep Value", "Value": f"{pd_dist.deep_value_count} ({pd_dist.deep_value_count/total*100:.1f}%)"},
            {"Field": "Value", "Value": f"{pd_dist.value_count} ({pd_dist.value_count/total*100:.1f}%)"},
            {"Field": "Reasonable", "Value": f"{pd_dist.reasonable_count} ({pd_dist.reasonable_count/total*100:.1f}%)"},
            {"Field": "Growth Premium", "Value": f"{pd_dist.growth_premium_count} ({pd_dist.growth_premium_count/total*100:.1f}%)"},
            {"Field": "Speculative", "Value": f"{pd_dist.speculative_count} ({pd_dist.speculative_count/total*100:.1f}%)"},
        ])

        # Top 5 Sectors by P/E
        if vs.top_sectors_by_pe:
            summary_rows.extend([
                {"Field": "", "Value": ""},
                {"Field": "--- Top 5 Sectors by P/E ---", "Value": ""},
            ])
            for i, sec in enumerate(vs.top_sectors_by_pe, 1):
                summary_rows.append(
                    {"Field": f"#{i} {sec.sector}", "Value": f"{sec.avg_pe:.1f}x ({sec.stock_count} stocks)"}
                )

        # PEG Ratio Analysis
        peg = vs.peg_analysis
        summary_rows.extend([
            {"Field": "", "Value": ""},
            {"Field": "--- PEG Ratio Analysis ---", "Value": ""},
            {"Field": "Avg PEG", "Value": f"{peg.avg_peg:.2f}" if peg.avg_peg else "N/A"},
            {"Field": "Median PEG", "Value": f"{peg.median_peg:.2f}" if peg.median_peg else "N/A"},
            {"Field": "Undervalued (PEG<1.0)", "Value": f"{peg.undervalued_count} ({peg.pct_undervalued:.1f}%)"},
            {"Field": "Fair Value (PEG 1-2)", "Value": str(peg.fair_value_count)},
            {"Field": "Expensive (PEG>2)", "Value": str(peg.expensive_count)},
        ])

        # Portfolio Risk Metrics
        summary_rows.extend([
            {"Field": "", "Value": ""},
            {"Field": "--- Portfolio Risk Metrics ---", "Value": ""},
            {"Field": "Avg Sharpe Ratio", "Value": f"{vs.avg_sharpe:.2f}" if vs.avg_sharpe else "N/A"},
            {"Field": "Avg Beta", "Value": f"{vs.avg_beta:.2f}" if vs.avg_beta else "N/A"},
            {"Field": "Avg Alpha", "Value": f"{vs.avg_alpha:.1f}%" if vs.avg_alpha else "N/A"},
        ])

    # LLM Enrichment summary
    llm_count = sum(1 for s in analyst_output.rated_stocks if s.valuation_source == "llm")
    total_rated = len(analyst_output.rated_stocks)
    summary_rows.extend([
        {"Field": "", "Value": ""},
        {"Field": "--- LLM Enrichment ---", "Value": ""},
        {"Field": "Stocks Enriched", "Value": f"{llm_count} / {total_rated}"},
        {"Field": "Data Source", "Value": "Claude Haiku (training knowledge)" if llm_count > 0 else "Deterministic estimates only"},
        {"Field": "Note", "Value": "LLM metrics from training data (early 2025), not real-time" if llm_count > 0 else "No LLM enrichment — using sector-based estimates"},
    ])

    summary_rows.extend([
        {"Field": "", "Value": ""},
        {"Field": "Methodology", "Value": analyst_output.methodology_notes},
        {"Field": "Summary", "Value": analyst_output.summary},
    ])
    df_summary = pd.DataFrame(summary_rows)

    # --- Color fill definitions ---
    _DARK_GREEN = PatternFill(start_color="006400", end_color="006400", fill_type="solid")
    _LIGHT_GREEN = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")
    _YELLOW = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
    _ORANGE = PatternFill(start_color="FFA500", end_color="FFA500", fill_type="solid")
    _RED = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")
    _GREEN = PatternFill(start_color="00CC00", end_color="00CC00", fill_type="solid")
    _WHITE_FONT = Font(color="FFFFFF")

    # Map column header -> {cell value -> (fill, use_white_font)}
    _COLOR_MAP = {
        "P/E Category": {
            "Deep Value": (_DARK_GREEN, True),
            "Value": (_LIGHT_GREEN, False),
            "Reasonable": (_YELLOW, False),
            "Growth Premium": (_ORANGE, False),
            "Speculative": (_RED, True),
        },
        "PEG Category": {
            "Undervalued": (_GREEN, False),
            "Fair Value": (_YELLOW, False),
            "Expensive": (_ORANGE, False),
        },
        "Return Cat": {
            "Strong": (_GREEN, False),
            "Good": (_LIGHT_GREEN, False),
            "Moderate": (_YELLOW, False),
            "Weak": (_RED, True),
        },
        "Sharpe Cat": {
            "Excellent": (_DARK_GREEN, True),
            "Good": (_LIGHT_GREEN, False),
            "Moderate": (_YELLOW, False),
            "Below Average": (_ORANGE, False),
        },
        "Alpha Cat": {
            "Strong Outperformer": (_DARK_GREEN, True),
            "Outperformer": (_LIGHT_GREEN, False),
            "Slight Underperformer": (_YELLOW, False),
            "Underperformer": (_RED, True),
        },
        "Risk Rating": {
            "Excellent": (_DARK_GREEN, True),
            "Good": (_LIGHT_GREEN, False),
            "Moderate": (_YELLOW, False),
            "Below Average": (_ORANGE, False),
            "Poor": (_RED, True),
        },
        "LLM Valuation": {
            "Deep Value": (_DARK_GREEN, True),
            "Value": (_LIGHT_GREEN, False),
            "Reasonable": (_YELLOW, False),
            "Growth Premium": (_ORANGE, False),
            "Speculative": (_RED, True),
        },
        "LLM Risk": {
            "Excellent": (_DARK_GREEN, True),
            "Good": (_LIGHT_GREEN, False),
            "Moderate": (_YELLOW, False),
            "Below Average": (_ORANGE, False),
            "Poor": (_RED, True),
        },
    }

    def _apply_color_formatting(ws):
        """Apply color fills to category columns based on cell values."""
        # Build column letter map from header row
        header_map = {}
        for col_idx in range(1, ws.max_column + 1):
            header = ws.cell(row=1, column=col_idx).value
            if header in _COLOR_MAP:
                header_map[col_idx] = _COLOR_MAP[header]

        for row_idx in range(2, ws.max_row + 1):
            for col_idx, value_map in header_map.items():
                cell = ws.cell(row=row_idx, column=col_idx)
                if cell.value in value_map:
                    fill, use_white = value_map[cell.value]
                    cell.fill = fill
                    if use_white:
                        cell.font = _WHITE_FONT

    # --- Rated ETFs ---
    etf_rows = []
    for e in analyst_output.rated_etfs:
        etf_rows.append({
            "Rank": e.etf_rank,
            "Symbol": e.symbol,
            "Name": e.name,
            "Focus": e.focus,
            "Focus Rank": e.focus_rank,
            "Tier": e.tier,
            "Conviction": e.conviction,
            "Screen": "PASS" if e.passes_etf_screen else "FAIL",
            "RS Rating": e.rs_rating,
            "Acc/Dis": e.acc_dis_rating,
            "YTD Chg %": e.ytd_change,
            "Vol % Chg": e.volume_pct_change,
            "Price Chg": e.price_change,
            "Close": e.close_price,
            "Div Yield": e.div_yield,
            "ETF Score": e.etf_score,
            "Themes": ", ".join(e.theme_tags),
            "Strengths": "; ".join(e.strengths),
            "Weaknesses": "; ".join(e.weaknesses),
        })
    df_etfs = pd.DataFrame(etf_rows)
    if not df_etfs.empty:
        df_etfs = df_etfs.sort_values(by=["Rank"], na_position="last").reset_index(drop=True)

    # --- Morningstar Analysis sheet ---
    # Morningstar Score: star_pts (0-40) + moat_pts (0-30) + discount_pts (0-20) + certainty_pts (0-10)
    _moat_pts = {"Wide": 30, "Narrow": 15, "None": 0}
    _unc_pts = {"Low": 10, "Medium": 7, "High": 3, "Very High": 0}
    ms_rows = []
    if research_output:
        for s in research_output.stocks:
            if s.morningstar_rating:
                # Star points: 5-star=40, 4-star=25
                star_pts = 40 if s.morningstar_rating == "5-star" else 25
                # Moat points: Wide=30, Narrow=15, None/missing=0
                moat_pts = _moat_pts.get(s.economic_moat, 0)
                # Discount points: based on P/FV (lower = more undervalued = higher score)
                # P/FV 0.50 -> 20pts, 0.70 -> 12pts, 0.90 -> 4pts, 1.00+ -> 0pts
                pfv = s.price_to_fair_value
                discount_pts = max(0, min(20, round((1.0 - pfv) * 40))) if pfv else 0
                # Certainty points: Low=10, Medium=7, High=3, Very High=0
                cert_pts = _unc_pts.get(s.morningstar_uncertainty, 0)
                ms_score = star_pts + moat_pts + discount_pts + cert_pts

                ms_rows.append({
                    "Rank": 0,  # filled after sort
                    "Symbol": s.symbol,
                    "Company": s.company_name,
                    "Sector": s.sector,
                    "M* Score": ms_score,
                    "M* Rating": s.morningstar_rating,
                    "Moat": s.economic_moat,
                    "Fair Value": s.fair_value,
                    "M* Price": s.morningstar_price,
                    "P/FV": s.price_to_fair_value,
                    "Discount %": round((1.0 - pfv) * 100, 1) if pfv else None,
                    "Uncertainty": s.morningstar_uncertainty,
                    "Composite": s.composite_rating,
                    "RS": s.rs_rating,
                    "EPS": s.eps_rating,
                    "Validation Score": s.validation_score,
                    "Multi-Source": s.is_multi_source_validated,
                    "Cap Size": s.cap_size,
                    "IBD Lists": ", ".join(s.ibd_lists) if s.ibd_lists else "",
                    "Schwab Themes": ", ".join(s.schwab_themes) if s.schwab_themes else "",
                    "Sources": ", ".join(s.sources),
                })
    df_morningstar = pd.DataFrame(ms_rows)
    if not df_morningstar.empty:
        df_morningstar = df_morningstar.sort_values(
            by=["M* Score"], ascending=False, na_position="last",
        ).reset_index(drop=True)
        df_morningstar["Rank"] = range(1, len(df_morningstar) + 1)

    # --- Write ---
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_top100.to_excel(writer, sheet_name="Top 100", index=False)
        if not df_t1.empty:
            df_t1.to_excel(writer, sheet_name="T1 Momentum", index=False)
        if not df_t2.empty:
            df_t2.to_excel(writer, sheet_name="T2 Quality Growth", index=False)
        if not df_t3.empty:
            df_t3.to_excel(writer, sheet_name="T3 Defensive", index=False)
        df_sector_ranks.to_excel(writer, sheet_name="Sector Rankings", index=False)
        if not df_keeps.empty:
            df_keeps.to_excel(writer, sheet_name="IBD Keeps", index=False)
        if not df_etfs.empty:
            df_etfs.to_excel(writer, sheet_name="Rated ETFs", index=False)
        if not df_morningstar.empty:
            df_morningstar.to_excel(writer, sheet_name="Morningstar", index=False)
        if not df_unrated.empty:
            df_unrated.to_excel(writer, sheet_name="Unrated", index=False)

        # Apply color formatting to stock sheets
        for sheet_name in ["Top 100", "T1 Momentum", "T2 Quality Growth", "T3 Defensive"]:
            if sheet_name in writer.book.sheetnames:
                _apply_color_formatting(writer.book[sheet_name])

    return filepath


def _write_rotation_excel(rotation_output, out_path: Path) -> Path:
    """Write Agent 03 Rotation Detector output to its own Excel file."""
    today = date.today().isoformat()
    filepath = out_path / f"agent03_rotation_{today}.xlsx"

    _DARK_GREEN = PatternFill(start_color="006400", end_color="006400", fill_type="solid")
    _LIGHT_GREEN = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")
    _YELLOW = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
    _RED = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")
    _WHITE_FONT = Font(color="FFFFFF")

    # --- Summary ---
    summary_rows = [
        {"Field": "Analysis Date", "Value": rotation_output.analysis_date},
        {"Field": "Verdict", "Value": rotation_output.verdict.value.upper()},
        {"Field": "Confidence", "Value": f"{rotation_output.confidence}%"},
        {"Field": "Rotation Type", "Value": rotation_output.rotation_type.value},
        {"Field": "Rotation Stage", "Value": rotation_output.rotation_stage.value if rotation_output.rotation_stage else "N/A"},
        {"Field": "Velocity", "Value": rotation_output.velocity or "N/A"},
        {"Field": "Regime", "Value": rotation_output.market_regime.regime},
        {"Field": "", "Value": ""},
        {"Field": "Signals Active", "Value": f"{rotation_output.signals.signals_active}/5"},
        {"Field": "Source Sectors", "Value": len(rotation_output.source_sectors)},
        {"Field": "Destination Sectors", "Value": len(rotation_output.destination_sectors)},
        {"Field": "Stable Sectors", "Value": len(rotation_output.stable_sectors)},
        {"Field": "", "Value": ""},
        {"Field": "Summary", "Value": rotation_output.summary},
        {"Field": "", "Value": ""},
        {"Field": "ETF Flow Summary", "Value": rotation_output.etf_flow_summary or "N/A"},
        {"Field": "", "Value": ""},
        {"Field": "Rotation Narrative", "Value": rotation_output.rotation_narrative or "N/A"},
        {"Field": "Narrative Source", "Value": rotation_output.narrative_source or "N/A"},
    ]
    df_summary = pd.DataFrame(summary_rows)

    # --- Signals ---
    signals = rotation_output.signals
    signal_rows = []
    for sig in [signals.rs_divergence, signals.leadership_change,
                signals.breadth_shift, signals.elite_concentration_shift,
                signals.ibd_keep_migration]:
        signal_rows.append({
            "Signal": sig.signal_name,
            "Triggered": "YES" if sig.triggered else "NO",
            "Value": sig.value,
            "Threshold": sig.threshold,
            "Evidence": sig.evidence,
            "ETF Confirmation": sig.etf_confirmation or "",
        })
    df_signals = pd.DataFrame(signal_rows)

    # --- Market Regime ---
    regime = rotation_output.market_regime
    regime_rows = [
        {"Field": "Regime", "Value": regime.regime},
        {"Field": "Sector Breadth %", "Value": regime.sector_breadth_pct},
        {"Field": "Bull Signals", "Value": "; ".join(regime.bull_signals_present) or "None"},
        {"Field": "Bear Signals", "Value": "; ".join(regime.bear_signals_present) or "None"},
        {"Field": "Regime Note", "Value": regime.regime_note},
    ]
    df_regime = pd.DataFrame(regime_rows)

    # --- Source Sectors (Outflow) ---
    source_rows = []
    for sf in rotation_output.source_sectors:
        source_rows.append({
            "Sector": sf.sector,
            "Cluster": sf.cluster,
            "Direction": sf.direction,
            "Rank": sf.current_rank,
            "Avg RS": sf.avg_rs,
            "Elite %": sf.elite_pct,
            "Stock Count": sf.stock_count,
            "Magnitude": sf.magnitude,
            "Evidence": sf.evidence,
        })
    df_source = pd.DataFrame(source_rows)

    # --- Destination Sectors (Inflow) ---
    dest_rows = []
    for sf in rotation_output.destination_sectors:
        dest_rows.append({
            "Sector": sf.sector,
            "Cluster": sf.cluster,
            "Direction": sf.direction,
            "Rank": sf.current_rank,
            "Avg RS": sf.avg_rs,
            "Elite %": sf.elite_pct,
            "Stock Count": sf.stock_count,
            "Magnitude": sf.magnitude,
            "Evidence": sf.evidence,
        })
    df_dest = pd.DataFrame(dest_rows)

    # --- Strategist Notes ---
    notes_rows = [{"#": i+1, "Note": note}
                  for i, note in enumerate(rotation_output.strategist_notes)]
    df_notes = pd.DataFrame(notes_rows)

    # --- Write ---
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_signals.to_excel(writer, sheet_name="Signals", index=False)
        df_regime.to_excel(writer, sheet_name="Market Regime", index=False)
        if not df_source.empty:
            df_source.to_excel(writer, sheet_name="Source Sectors", index=False)
        if not df_dest.empty:
            df_dest.to_excel(writer, sheet_name="Destination Sectors", index=False)
        df_notes.to_excel(writer, sheet_name="Strategist Notes", index=False)

        # Color-code signals: green=triggered, red=not
        ws_signals = writer.book["Signals"]
        for row_idx in range(2, ws_signals.max_row + 1):
            cell = ws_signals.cell(row=row_idx, column=2)  # Triggered column
            if cell.value == "YES":
                cell.fill = _LIGHT_GREEN
            elif cell.value == "NO":
                cell.fill = _RED
                cell.font = _WHITE_FONT

        # Color-code verdict on summary
        ws_summary = writer.book["Summary"]
        for row_idx in range(2, ws_summary.max_row + 1):
            field_cell = ws_summary.cell(row=row_idx, column=1)
            value_cell = ws_summary.cell(row=row_idx, column=2)
            if field_cell.value == "Verdict":
                if value_cell.value == "ACTIVE":
                    value_cell.fill = _RED
                    value_cell.font = _WHITE_FONT
                elif value_cell.value == "EMERGING":
                    value_cell.fill = _YELLOW
                elif value_cell.value == "NONE":
                    value_cell.fill = _LIGHT_GREEN

    return filepath


def _write_strategy_excel(strategy_output, out_path: Path) -> Path:
    """Write Agent 04 Sector Strategist output to its own Excel file."""
    today = date.today().isoformat()
    filepath = out_path / f"agent04_strategy_{today}.xlsx"

    _DARK_GREEN = PatternFill(start_color="006400", end_color="006400", fill_type="solid")
    _LIGHT_GREEN = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")
    _YELLOW = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
    _ORANGE = PatternFill(start_color="FFA500", end_color="FFA500", fill_type="solid")
    _WHITE_FONT = Font(color="FFFFFF")

    alloc = strategy_output.sector_allocations

    # --- Summary ---
    summary_rows = [
        {"Field": "Analysis Date", "Value": strategy_output.analysis_date},
        {"Field": "Rotation Response", "Value": strategy_output.rotation_response},
        {"Field": "Regime Adjustment", "Value": strategy_output.regime_adjustment},
        {"Field": "", "Value": ""},
        {"Field": "--- Tier Targets ---", "Value": ""},
        {"Field": "T1 (Momentum)", "Value": f"{alloc.tier_targets.get('T1', 0):.1f}%"},
        {"Field": "T2 (Quality)", "Value": f"{alloc.tier_targets.get('T2', 0):.1f}%"},
        {"Field": "T3 (Defensive)", "Value": f"{alloc.tier_targets.get('T3', 0):.1f}%"},
        {"Field": "Cash", "Value": f"{alloc.tier_targets.get('Cash', 0):.1f}%"},
        {"Field": "Cash Recommendation", "Value": f"{alloc.cash_recommendation:.1f}%"},
        {"Field": "", "Value": ""},
        {"Field": "Overall Sectors", "Value": len(alloc.overall_allocation)},
        {"Field": "Theme Recommendations", "Value": len(strategy_output.theme_recommendations)},
        {"Field": "Rotation Signals", "Value": len(strategy_output.rotation_signals)},
        {"Field": "", "Value": ""},
        {"Field": "Allocation Rationale", "Value": alloc.rationale},
        {"Field": "Summary", "Value": strategy_output.summary},
    ]
    df_summary = pd.DataFrame(summary_rows)

    # --- Overall Allocation ---
    overall_rows = [
        {"Sector": sector, "Allocation %": pct}
        for sector, pct in sorted(alloc.overall_allocation.items(), key=lambda x: -x[1])
    ]
    df_overall = pd.DataFrame(overall_rows)

    # --- T1 Allocation ---
    t1_rows = [
        {"Sector": sector, "Allocation %": pct}
        for sector, pct in sorted(alloc.tier_1_allocation.items(), key=lambda x: -x[1])
    ]
    df_t1 = pd.DataFrame(t1_rows)

    # --- T2 Allocation ---
    t2_rows = [
        {"Sector": sector, "Allocation %": pct}
        for sector, pct in sorted(alloc.tier_2_allocation.items(), key=lambda x: -x[1])
    ]
    df_t2 = pd.DataFrame(t2_rows)

    # --- T3 Allocation ---
    t3_rows = [
        {"Sector": sector, "Allocation %": pct}
        for sector, pct in sorted(alloc.tier_3_allocation.items(), key=lambda x: -x[1])
    ]
    df_t3 = pd.DataFrame(t3_rows)

    # --- Theme Recommendations ---
    theme_rows = []
    for rec in strategy_output.theme_recommendations:
        etf_rank_str = ""
        if rec.etf_rankings:
            etf_rank_str = "; ".join(
                f"{r.get('symbol', '?')}(score={r.get('etf_score', '?')}, rank={r.get('rank', '?')})"
                for r in rec.etf_rankings
            )
        theme_rows.append({
            "Theme": rec.theme,
            "ETFs": ", ".join(rec.recommended_etfs),
            "Tier Fit": f"T{rec.tier_fit}",
            "Allocation": rec.allocation_suggestion,
            "Conviction": rec.conviction,
            "Rationale": rec.rationale,
            "ETF Rankings": etf_rank_str,
        })
    df_themes = pd.DataFrame(theme_rows)

    # --- Rotation Signals ---
    signal_rows = []
    for sig in strategy_output.rotation_signals:
        signal_rows.append({
            "Action": sig.action,
            "Trigger": sig.trigger,
            "Confirmation": "; ".join(sig.confirmation),
            "Invalidation": "; ".join(sig.invalidation),
        })
    df_signals = pd.DataFrame(signal_rows)

    # --- Write ---
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_overall.to_excel(writer, sheet_name="Overall Allocation", index=False)
        df_t1.to_excel(writer, sheet_name="T1 Allocation", index=False)
        df_t2.to_excel(writer, sheet_name="T2 Allocation", index=False)
        df_t3.to_excel(writer, sheet_name="T3 Allocation", index=False)
        if not df_themes.empty:
            df_themes.to_excel(writer, sheet_name="Theme Recommendations", index=False)
        if not df_signals.empty:
            df_signals.to_excel(writer, sheet_name="Rotation Signals", index=False)

        # Color-code conviction in Theme Recommendations
        if "Theme Recommendations" in writer.book.sheetnames:
            ws = writer.book["Theme Recommendations"]
            # Find conviction column
            conv_col = None
            for col_idx in range(1, ws.max_column + 1):
                if ws.cell(row=1, column=col_idx).value == "Conviction":
                    conv_col = col_idx
                    break
            if conv_col:
                for row_idx in range(2, ws.max_row + 1):
                    cell = ws.cell(row=row_idx, column=conv_col)
                    if cell.value == "HIGH":
                        cell.fill = _LIGHT_GREEN
                    elif cell.value == "MEDIUM":
                        cell.fill = _YELLOW
                    elif cell.value == "LOW":
                        cell.fill = _ORANGE

    return filepath


def _write_portfolio_excel(portfolio_output, out_path: Path) -> Path:
    """Write Agent 05 Portfolio Manager output to its own Excel file."""
    today = date.today().isoformat()
    filepath = out_path / f"agent05_portfolio_{today}.xlsx"

    # --- Summary ---
    t1 = portfolio_output.tier_1
    t2 = portfolio_output.tier_2
    t3 = portfolio_output.tier_3
    summary_rows = [
        {"Field": "Analysis Date", "Value": portfolio_output.analysis_date},
        {"Field": "Total Positions", "Value": portfolio_output.total_positions},
        {"Field": "Stock Count", "Value": portfolio_output.stock_count},
        {"Field": "ETF Count", "Value": portfolio_output.etf_count},
        {"Field": "", "Value": ""},
        {"Field": "--- Tier Allocation ---", "Value": ""},
        {"Field": "T1 Momentum", "Value": f"{t1.actual_pct:.1f}% ({len(t1.positions)} pos)"},
        {"Field": "T2 Quality Growth", "Value": f"{t2.actual_pct:.1f}% ({len(t2.positions)} pos)"},
        {"Field": "T3 Defensive", "Value": f"{t3.actual_pct:.1f}% ({len(t3.positions)} pos)"},
        {"Field": "Cash", "Value": f"{portfolio_output.cash_pct:.1f}%"},
        {"Field": "", "Value": ""},
        {"Field": "--- Keeps ---", "Value": ""},
        {"Field": "Fundamental Keeps", "Value": len(portfolio_output.keeps_placement.fundamental_keeps)},
        {"Field": "IBD Keeps", "Value": len(portfolio_output.keeps_placement.ibd_keeps)},
        {"Field": "Total Keeps", "Value": portfolio_output.keeps_placement.total_keeps},
        {"Field": "", "Value": ""},
        {"Field": "Methodology", "Value": portfolio_output.construction_methodology},
        {"Field": "Summary", "Value": portfolio_output.summary},
    ]
    df_summary = pd.DataFrame(summary_rows)

    # --- Positions per tier ---
    def _pos_row(p, rank):
        return {
            "Rank": rank,
            "Symbol": p.symbol,
            "Company": p.company_name,
            "Sector": p.sector,
            "Type": p.asset_type,
            "Target %": p.target_pct,
            "Stop %": p.trailing_stop_pct,
            "Max Loss %": p.max_loss_pct,
            "Keep": p.keep_category or "",
            "Conviction": p.conviction,
            "Vol Adj %": p.volatility_adjustment,
            "Sizing Source": p.sizing_source or "",
            "Reasoning": p.reasoning,
        }

    df_t1 = pd.DataFrame([_pos_row(p, i+1) for i, p in enumerate(t1.positions)])
    df_t2 = pd.DataFrame([_pos_row(p, i+1) for i, p in enumerate(t2.positions)])
    df_t3 = pd.DataFrame([_pos_row(p, i+1) for i, p in enumerate(t3.positions)])

    # --- Sector Exposure ---
    sector_rows = [
        {"Sector": sec, "Allocation %": pct}
        for sec, pct in sorted(portfolio_output.sector_exposure.items(), key=lambda x: -x[1])
    ]
    df_sectors = pd.DataFrame(sector_rows)

    # --- Orders ---
    order_rows = []
    for o in portfolio_output.orders:
        order_rows.append({
            "Symbol": o.symbol,
            "Action": o.action,
            "Tier": o.tier,
            "Target %": o.target_pct,
            "Rationale": o.rationale,
        })
    df_orders = pd.DataFrame(order_rows)

    # --- Write ---
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        if not df_t1.empty:
            df_t1.to_excel(writer, sheet_name="T1 Momentum", index=False)
        if not df_t2.empty:
            df_t2.to_excel(writer, sheet_name="T2 Quality Growth", index=False)
        if not df_t3.empty:
            df_t3.to_excel(writer, sheet_name="T3 Defensive", index=False)
        if not df_sectors.empty:
            df_sectors.to_excel(writer, sheet_name="Sector Exposure", index=False)
        if not df_orders.empty:
            df_orders.to_excel(writer, sheet_name="Orders", index=False)

    return filepath


def _write_risk_excel(risk_output, out_path: Path) -> Path:
    """Write Agent 06 Risk Officer output to its own Excel file."""
    today = date.today().isoformat()
    filepath = out_path / f"agent06_risk_{today}.xlsx"

    # --- Summary ---
    swl = risk_output.sleep_well_scores
    summary_rows = [
        {"Field": "Analysis Date", "Value": risk_output.analysis_date},
        {"Field": "Overall Status", "Value": risk_output.overall_status},
        {"Field": "Checks Run", "Value": len(risk_output.check_results)},
        {"Field": "Vetoes", "Value": len(risk_output.vetoes)},
        {"Field": "Warnings", "Value": len(risk_output.warnings)},
        {"Field": "", "Value": ""},
        {"Field": "--- Sleep Well Scores ---", "Value": ""},
        {"Field": "Overall", "Value": f"{swl.overall_score}/10"},
        {"Field": "T1 Score", "Value": f"{swl.tier_1_score}/10"},
        {"Field": "T2 Score", "Value": f"{swl.tier_2_score}/10"},
        {"Field": "T3 Score", "Value": f"{swl.tier_3_score}/10"},
        {"Field": "Factors", "Value": "; ".join(swl.factors)},
        {"Field": "", "Value": ""},
        {"Field": "Stop-Loss Source", "Value": risk_output.stop_loss_source or "N/A"},
        {"Field": "Stop-Loss Recommendations", "Value": len(risk_output.stop_loss_recommendations)},
        {"Field": "", "Value": ""},
        {"Field": "Summary", "Value": risk_output.summary},
    ]
    df_summary = pd.DataFrame(summary_rows)

    # --- Check Results ---
    check_rows = []
    for c in risk_output.check_results:
        check_rows.append({
            "Check": c.check_name,
            "Status": c.status,
            "Findings": c.findings,
            "Details": "; ".join(c.details) if c.details else "",
        })
    df_checks = pd.DataFrame(check_rows)

    # --- Stress Tests ---
    stress_rows = []
    for s in risk_output.stress_test_results.scenarios:
        stress_rows.append({
            "Scenario": s.scenario_name,
            "Impact": s.impact_description,
            "Est Drawdown %": s.estimated_drawdown_pct,
            "Most Affected": ", ".join(s.positions_most_affected[:5]),
        })
    df_stress = pd.DataFrame(stress_rows)

    # --- Stop-Loss Recommendations ---
    stop_rows = []
    for rec in risk_output.stop_loss_recommendations:
        stop_rows.append({
            "Symbol": rec.symbol,
            "Current Stop %": rec.current_stop_pct,
            "Recommended Stop %": rec.recommended_stop_pct,
            "Reason": rec.reason,
            "Volatility": rec.volatility_flag or "",
        })
    df_stops = pd.DataFrame(stop_rows)

    # --- Write ---
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_checks.to_excel(writer, sheet_name="Check Results", index=False)
        if not df_stress.empty:
            df_stress.to_excel(writer, sheet_name="Stress Tests", index=False)
        if not df_stops.empty:
            df_stops.to_excel(writer, sheet_name="Stop-Loss Recommendations", index=False)

    return filepath


def _write_returns_projector_excel(returns_output, out_path: Path) -> Path:
    """Write Agent 07 Returns Projector output to its own Excel file."""
    today = date.today().isoformat()
    filepath = out_path / f"agent07_returns_{today}.xlsx"

    ta = returns_output.tier_allocation
    er = returns_output.expected_return
    rm = returns_output.risk_metrics
    ci = returns_output.confidence_intervals

    # --- Summary ---
    summary_rows = [
        {"Field": "Analysis Date", "Value": returns_output.analysis_date},
        {"Field": "Portfolio Value", "Value": f"${returns_output.portfolio_value:,.0f}"},
        {"Field": "Market Regime", "Value": returns_output.market_regime},
        {"Field": "Regime Source", "Value": returns_output.regime_source},
        {"Field": "", "Value": ""},
        {"Field": "--- Tier Allocation ---", "Value": ""},
        {"Field": "Tier 1 %", "Value": f"{ta.tier_1_pct:.1f}%"},
        {"Field": "Tier 2 %", "Value": f"{ta.tier_2_pct:.1f}%"},
        {"Field": "Tier 3 %", "Value": f"{ta.tier_3_pct:.1f}%"},
        {"Field": "Cash %", "Value": f"{ta.cash_pct:.1f}%"},
        {"Field": "", "Value": ""},
        {"Field": "--- Expected Returns ---", "Value": ""},
        {"Field": "Expected 3m", "Value": f"{er.expected_3m:.2f}%"},
        {"Field": "Expected 6m", "Value": f"{er.expected_6m:.2f}%"},
        {"Field": "Expected 12m", "Value": f"{er.expected_12m:.2f}%"},
        {"Field": "", "Value": ""},
        {"Field": "--- Risk Metrics ---", "Value": ""},
        {"Field": "Max DD (with stops)", "Value": f"{rm.max_drawdown_with_stops:.2f}%"},
        {"Field": "Max DD (no stops)", "Value": f"{rm.max_drawdown_without_stops:.2f}%"},
        {"Field": "Max DD ($)", "Value": f"${rm.max_drawdown_dollar:,.0f}"},
        {"Field": "Projected Volatility", "Value": f"{rm.projected_annual_volatility:.2f}%"},
        {"Field": "Sharpe (base)", "Value": f"{rm.portfolio_sharpe_base:.2f}"},
        {"Field": "Return/Risk", "Value": f"{rm.return_per_unit_risk:.2f}"},
        {"Field": "", "Value": ""},
        {"Field": "Summary", "Value": returns_output.summary},
    ]
    df_summary = pd.DataFrame(summary_rows)

    # --- Scenarios ---
    scenario_rows = []
    for s in returns_output.scenarios:
        scenario_rows.append({
            "Scenario": s.scenario.title(),
            "Probability": f"{s.probability:.0%}",
            "T1 Return": f"{s.tier_1_return_pct:.1f}%",
            "T2 Return": f"{s.tier_2_return_pct:.1f}%",
            "T3 Return": f"{s.tier_3_return_pct:.1f}%",
            "Portfolio 3m": f"{s.portfolio_return_3m:.2f}%",
            "Portfolio 6m": f"{s.portfolio_return_6m:.2f}%",
            "Portfolio 12m": f"{s.portfolio_return_12m:.2f}%",
            "12m Gain ($)": f"${s.portfolio_gain_12m:,.0f}",
            "Ending Value": f"${s.ending_value_12m:,.0f}",
            "Reasoning": s.reasoning,
        })
    df_scenarios = pd.DataFrame(scenario_rows)

    # --- Benchmarks ---
    bench_rows = []
    for b in returns_output.benchmark_comparisons:
        bench_rows.append({
            "Benchmark": f"{b.symbol} ({b.name})",
            "Return 3m": f"{b.benchmark_return_3m:.2f}%",
            "Return 6m": f"{b.benchmark_return_6m:.2f}%",
            "Return 12m": f"{b.benchmark_return_12m:.2f}%",
            "Alpha (Bull)": f"{b.alpha_bull_12m:.2f}%",
            "Alpha (Base)": f"{b.alpha_base_12m:.2f}%",
            "Alpha (Bear)": f"{b.alpha_bear_12m:.2f}%",
            "Expected Alpha": f"{b.expected_alpha_12m:.2f}%",
            "Outperform Prob": f"{b.outperform_probability:.0%}",
        })
    df_benchmarks = pd.DataFrame(bench_rows)

    # --- Tier Mix Alternatives ---
    mix_rows = []
    for m in returns_output.tier_mix_comparison:
        mix_rows.append({
            "Mix": m.label,
            "T1 %": f"{m.tier_1_pct:.0f}%",
            "T2 %": f"{m.tier_2_pct:.0f}%",
            "T3 %": f"{m.tier_3_pct:.0f}%",
            "Cash %": f"{m.cash_pct:.0f}%",
            "Expected 12m": f"{m.expected_return_12m:.2f}%",
            "Bull 12m": f"{m.bull_return_12m:.2f}%",
            "Bear 12m": f"{m.bear_return_12m:.2f}%",
            "Max DD (stops)": f"{m.max_drawdown_with_stops:.2f}%",
            "Sharpe (base)": f"{m.sharpe_ratio_base:.2f}",
            "Note": m.comparison_note,
        })
    df_mixes = pd.DataFrame(mix_rows)

    # --- Confidence Intervals ---
    ci_rows = []
    for label, pr in [("3-Month", ci.horizon_3m), ("6-Month", ci.horizon_6m), ("12-Month", ci.horizon_12m)]:
        ci_rows.append({
            "Horizon": label,
            "P10": f"{pr.p10:.2f}%",
            "P25": f"{pr.p25:.2f}%",
            "P50 (Median)": f"{pr.p50:.2f}%",
            "P75": f"{pr.p75:.2f}%",
            "P90": f"{pr.p90:.2f}%",
            "P10 ($)": f"${pr.p10_dollar:,.0f}",
            "P50 ($)": f"${pr.p50_dollar:,.0f}",
            "P90 ($)": f"${pr.p90_dollar:,.0f}",
        })
    df_ci = pd.DataFrame(ci_rows)

    # --- Caveats ---
    df_caveats = pd.DataFrame({"Caveat": returns_output.caveats})

    # --- Asset Decomposition ---
    decomp_rows = []
    if returns_output.asset_type_decomposition:
        for d in returns_output.asset_type_decomposition:
            decomp_rows.append({
                "Tier": d.tier,
                "Stock %": f"{d.stock_contribution_pct:.2f}%",
                "ETF %": f"{d.etf_contribution_pct:.2f}%",
                "Stock Count": d.stock_count,
                "ETF Count": d.etf_count,
                "Stock Vol": f"{d.stock_avg_volatility:.2f}%" if d.stock_avg_volatility is not None else "N/A",
                "ETF Vol": f"{d.etf_avg_volatility:.2f}%" if d.etf_avg_volatility is not None else "N/A",
            })
    df_decomp = pd.DataFrame(decomp_rows)

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_scenarios.to_excel(writer, sheet_name="Scenarios", index=False)
        df_benchmarks.to_excel(writer, sheet_name="Benchmarks", index=False)
        df_mixes.to_excel(writer, sheet_name="Tier Mix Alternatives", index=False)
        df_ci.to_excel(writer, sheet_name="Confidence Intervals", index=False)
        if not df_decomp.empty:
            df_decomp.to_excel(writer, sheet_name="Asset Decomposition", index=False)
        df_caveats.to_excel(writer, sheet_name="Caveats", index=False)

    return filepath


def _write_reconciler_excel(reconciler_output, out_path: Path) -> Path:
    """Write Agent 08 Reconciler output to its own Excel file."""
    today = date.today().isoformat()
    filepath = out_path / f"agent08_reconciler_{today}.xlsx"

    mf = reconciler_output.money_flow
    metrics = reconciler_output.transformation_metrics
    kv = reconciler_output.keep_verification

    # --- Summary ---
    summary_rows = [
        {"Field": "Analysis Date", "Value": reconciler_output.analysis_date},
        {"Field": "Total Actions", "Value": len(reconciler_output.actions)},
        {"Field": "Current Holdings", "Value": len(reconciler_output.current_holdings.holdings)},
        {"Field": "Total Value", "Value": f"${reconciler_output.current_holdings.total_value:,.0f}"},
        {"Field": "", "Value": ""},
        {"Field": "--- Money Flow ---", "Value": ""},
        {"Field": "Sell Proceeds", "Value": f"${mf.sell_proceeds:,.0f}"},
        {"Field": "Trim Proceeds", "Value": f"${mf.trim_proceeds:,.0f}"},
        {"Field": "Buy Cost", "Value": f"${mf.buy_cost:,.0f}"},
        {"Field": "Add Cost", "Value": f"${mf.add_cost:,.0f}"},
        {"Field": "Net Cash Change", "Value": f"${mf.net_cash_change:,.0f}"},
        {"Field": "Cash Reserve %", "Value": f"{mf.cash_reserve_pct:.1f}%"},
        {"Field": "", "Value": ""},
        {"Field": "--- Transformation ---", "Value": ""},
        {"Field": "Before Positions", "Value": metrics.before_position_count},
        {"Field": "After Positions", "Value": metrics.after_position_count},
        {"Field": "Before Sectors", "Value": metrics.before_sector_count},
        {"Field": "After Sectors", "Value": metrics.after_sector_count},
        {"Field": "Turnover %", "Value": f"{metrics.turnover_pct:.0f}%"},
        {"Field": "", "Value": ""},
        {"Field": "--- Keep Verification ---", "Value": ""},
        {"Field": "Keeps in Current", "Value": len(kv.keeps_in_current)},
        {"Field": "Keeps Missing", "Value": ", ".join(kv.keeps_missing) or "None"},
        {"Field": "Keeps to Buy", "Value": ", ".join(kv.keeps_to_buy) or "None"},
        {"Field": "", "Value": ""},
        {"Field": "Summary", "Value": reconciler_output.summary},
        {"Field": "", "Value": ""},
        {"Field": "Rationale Source", "Value": reconciler_output.rationale_source or "N/A"},
    ]
    df_summary = pd.DataFrame(summary_rows)

    # --- Actions ---
    action_rows = []
    for a in reconciler_output.actions:
        action_rows.append({
            "Symbol": a.symbol,
            "Action": a.action_type,
            "Current %": a.current_pct,
            "Target %": a.target_pct,
            "Dollar Change": f"${a.dollar_change:,.0f}",
            "Priority": a.priority,
            "Week": a.week,
            "Rationale": a.rationale,
        })
    df_actions = pd.DataFrame(action_rows)

    # --- Implementation Plan ---
    plan_rows = []
    for w in reconciler_output.implementation_plan:
        plan_rows.append({
            "Week": w.week,
            "Phase": w.phase_name,
            "Actions": len(w.actions),
        })
    df_plan = pd.DataFrame(plan_rows)

    # --- ETF Implementation Notes ---
    etf_notes_rows = [{"Note": note} for note in reconciler_output.etf_implementation_notes]
    df_etf_notes = pd.DataFrame(etf_notes_rows) if etf_notes_rows else pd.DataFrame()

    # --- Write ---
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        if not df_actions.empty:
            df_actions.to_excel(writer, sheet_name="Actions", index=False)
        df_plan.to_excel(writer, sheet_name="Implementation Plan", index=False)
        if not df_etf_notes.empty:
            df_etf_notes.to_excel(writer, sheet_name="ETF Implementation Notes", index=False)

    return filepath


def _write_educator_excel(educator_output, out_path: Path) -> Path:
    """Write Agent 09 Educator output to its own Excel file."""
    today = date.today().isoformat()
    filepath = out_path / f"agent09_educator_{today}.xlsx"

    # --- Summary ---
    summary_rows = [
        {"Field": "Analysis Date", "Value": educator_output.analysis_date},
        {"Field": "Stocks Explained", "Value": len(educator_output.stock_explanations)},
        {"Field": "Concept Lessons", "Value": len(educator_output.concept_lessons)},
        {"Field": "Glossary Entries", "Value": len(educator_output.glossary)},
        {"Field": "Action Items", "Value": len(educator_output.action_items)},
        {"Field": "", "Value": ""},
        {"Field": "Executive Summary", "Value": educator_output.executive_summary},
        {"Field": "", "Value": ""},
        {"Field": "Rotation Explanation", "Value": educator_output.rotation_explanation},
        {"Field": "", "Value": ""},
        {"Field": "Risk Explainer", "Value": educator_output.risk_explainer},
        {"Field": "", "Value": ""},
        {"Field": "Summary", "Value": educator_output.summary},
    ]
    df_summary = pd.DataFrame(summary_rows)

    # --- Stock Explanations ---
    stock_rows = []
    for se in educator_output.stock_explanations:
        stock_rows.append({
            "Symbol": se.symbol,
            "Company": se.company_name,
            "Tier": se.tier,
            "Keep": se.keep_category or "",
            "One-Liner": se.one_liner,
            "Why Selected": se.why_selected,
            "IBD Ratings": se.ibd_ratings_explained,
            "Strength": se.key_strength,
            "Risk": se.key_risk,
            "Position Context": se.position_context,
        })
    df_stocks = pd.DataFrame(stock_rows)

    # --- Concept Lessons ---
    lesson_rows = []
    for cl in educator_output.concept_lessons:
        lesson_rows.append({
            "Concept": cl.concept,
            "Explanation": cl.simple_explanation,
            "Analogy": cl.analogy,
            "Why It Matters": cl.why_it_matters,
            "Example": cl.example_from_analysis,
            "Reference": cl.framework_reference,
        })
    df_lessons = pd.DataFrame(lesson_rows)

    # --- Action Items ---
    action_rows = [{"#": i+1, "Action Item": item}
                   for i, item in enumerate(educator_output.action_items)]
    df_actions = pd.DataFrame(action_rows)

    # --- Glossary ---
    glossary_rows = [{"Term": term, "Definition": defn}
                     for term, defn in sorted(educator_output.glossary.items())]
    df_glossary = pd.DataFrame(glossary_rows)

    # --- ETF Explanations ---
    etf_expl_rows = []
    for ee in educator_output.etf_explanations:
        etf_expl_rows.append({
            "Symbol": ee.symbol,
            "Name": ee.name,
            "Tier": ee.tier,
            "One-Liner": ee.one_liner,
            "Why Selected": ee.why_selected,
            "Theme Context": ee.theme_context or "",
            "Conviction": ee.conviction_explained or "",
            "Key Strength": ee.key_strength,
            "Key Risk": ee.key_risk,
        })
    df_etf_expl = pd.DataFrame(etf_expl_rows) if etf_expl_rows else pd.DataFrame()

    # --- Write ---
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        if not df_stocks.empty:
            df_stocks.to_excel(writer, sheet_name="Stock Explanations", index=False)
        if not df_etf_expl.empty:
            df_etf_expl.to_excel(writer, sheet_name="ETF Explanations", index=False)
        if not df_lessons.empty:
            df_lessons.to_excel(writer, sheet_name="Concept Lessons", index=False)
        if not df_actions.empty:
            df_actions.to_excel(writer, sheet_name="Action Items", index=False)
        if not df_glossary.empty:
            df_glossary.to_excel(writer, sheet_name="Glossary", index=False)

    return filepath


def _write_synthesizer_excel(synthesis_output, out_path: Path) -> Path:
    """Write Agent 10 Executive Summary Synthesizer output to Excel."""
    today = date.today().isoformat()
    filepath = out_path / f"agent10_executive_summary_{today}.xlsx"

    # --- Investment Thesis ---
    thesis_rows = [
        {"Section": "Investment Thesis", "Content": synthesis_output.investment_thesis},
        {"Section": "", "Content": ""},
        {"Section": "Portfolio Narrative", "Content": synthesis_output.portfolio_narrative},
        {"Section": "", "Content": ""},
        {"Section": "Risk/Reward Assessment", "Content": synthesis_output.risk_reward_assessment},
        {"Section": "", "Content": ""},
        {"Section": "Market Context", "Content": synthesis_output.market_context},
    ]
    df_thesis = pd.DataFrame(thesis_rows)

    # --- Key Numbers Dashboard ---
    kn = synthesis_output.key_numbers
    numbers_rows = [
        {"Category": "Research", "Metric": "Total Securities Scanned", "Value": kn.total_stocks_scanned},
        {"Category": "Research", "Metric": "Stocks in Universe", "Value": kn.stocks_in_universe},
        {"Category": "Research", "Metric": "ETFs in Universe", "Value": kn.etfs_in_universe},
        {"Category": "Analyst", "Metric": "Stocks Rated", "Value": kn.stocks_rated},
        {"Category": "Analyst", "Metric": "ETFs Rated", "Value": kn.etfs_rated},
        {"Category": "Analyst", "Metric": "Tier 1 Count", "Value": kn.tier_1_count},
        {"Category": "Analyst", "Metric": "Tier 2 Count", "Value": kn.tier_2_count},
        {"Category": "Analyst", "Metric": "Tier 3 Count", "Value": kn.tier_3_count},
        {"Category": "Rotation", "Metric": "Verdict", "Value": kn.rotation_verdict},
        {"Category": "Rotation", "Metric": "Confidence", "Value": f"{kn.rotation_confidence}%"},
        {"Category": "Rotation", "Metric": "Market Regime", "Value": kn.market_regime},
        {"Category": "Portfolio", "Metric": "Total Positions", "Value": kn.portfolio_positions},
        {"Category": "Portfolio", "Metric": "Stock Count", "Value": kn.stock_count},
        {"Category": "Portfolio", "Metric": "ETF Count", "Value": kn.etf_count},
        {"Category": "Portfolio", "Metric": "Cash %", "Value": f"{kn.cash_pct:.1f}%"},
        {"Category": "Risk", "Metric": "Status", "Value": kn.risk_status},
        {"Category": "Risk", "Metric": "Sleep Well Score", "Value": f"{kn.sleep_well_score}/10"},
        {"Category": "Risk", "Metric": "Warnings", "Value": kn.risk_warnings_count},
        {"Category": "Returns", "Metric": "Bull Case 12m", "Value": f"{kn.bull_case_return_12m:.1f}%"},
        {"Category": "Returns", "Metric": "Base Case 12m", "Value": f"{kn.base_case_return_12m:.1f}%"},
        {"Category": "Returns", "Metric": "Bear Case 12m", "Value": f"{kn.bear_case_return_12m:.1f}%"},
        {"Category": "Reconciler", "Metric": "Turnover", "Value": f"{kn.turnover_pct:.0f}%"},
        {"Category": "Reconciler", "Metric": "Actions Count", "Value": kn.actions_count},
    ]
    df_numbers = pd.DataFrame(numbers_rows)

    # --- Cross-Agent Connections ---
    conn_rows = []
    for conn in synthesis_output.cross_agent_connections:
        conn_rows.append({
            "From Agent": conn.from_agent,
            "To Agent": conn.to_agent,
            "Connection": conn.connection,
            "Implication": conn.implication,
        })
    df_connections = pd.DataFrame(conn_rows)

    # --- Contradictions ---
    contra_rows = []
    for c in synthesis_output.contradictions:
        contra_rows.append({
            "Agent A": c.agent_a,
            "Agent B": c.agent_b,
            "Finding A": c.finding_a,
            "Finding B": c.finding_b,
            "Resolution": c.resolution,
        })
    df_contradictions = pd.DataFrame(contra_rows) if contra_rows else pd.DataFrame(
        [{"Agent A": "—", "Agent B": "—", "Finding A": "No contradictions detected",
          "Finding B": "—", "Resolution": "—"}]
    )

    # --- Action Items ---
    action_rows = [{"#": i+1, "Action Item": item}
                   for i, item in enumerate(synthesis_output.action_items)]
    df_actions = pd.DataFrame(action_rows)

    # --- Summary ---
    summary_rows = [
        {"Field": "Synthesis Source", "Value": synthesis_output.synthesis_source},
        {"Field": "Analysis Date", "Value": synthesis_output.analysis_date},
        {"Field": "Connections", "Value": len(synthesis_output.cross_agent_connections)},
        {"Field": "Contradictions", "Value": len(synthesis_output.contradictions)},
        {"Field": "Action Items", "Value": len(synthesis_output.action_items)},
        {"Field": "", "Value": ""},
        {"Field": "Summary", "Value": synthesis_output.summary},
    ]
    df_summary = pd.DataFrame(summary_rows)

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df_thesis.to_excel(writer, sheet_name="Investment Thesis", index=False)
        df_numbers.to_excel(writer, sheet_name="Key Numbers", index=False)
        df_connections.to_excel(writer, sheet_name="Cross-Agent Connections", index=False)
        df_contradictions.to_excel(writer, sheet_name="Contradictions", index=False)
        df_actions.to_excel(writer, sheet_name="Action Items", index=False)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)

        # Format column widths
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            for col in ws.columns:
                max_len = max(len(str(cell.value or "")) for cell in col)
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 80)

    return filepath


def main(data_dir: str = "data", output_dir: str = "output"):
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)

    # ===== PHASE 1: Research Agent =====
    print(f"[Agent 01] Running Research pipeline against '{data_dir}' ...")
    research_output = run_research_pipeline(data_dir)
    print(f"[Agent 01] Done — {len(research_output.stocks)} stocks, {len(research_output.etfs)} ETFs")

    research_file = _write_research_excel(research_output, out_path)
    print(f"[Agent 01] Saved: {research_file}")

    # ===== PHASE 2: Analyst Agent =====
    print(f"[Agent 02] Running Analyst pipeline ...")
    analyst_output = run_analyst_pipeline(research_output)
    print(f"[Agent 02] Done — {len(analyst_output.rated_stocks)} top stocks, "
          f"{len(analyst_output.ibd_keeps)} IBD keeps, "
          f"{len(analyst_output.sector_rankings)} sectors ranked")

    analyst_file = _write_analyst_excel(analyst_output, out_path, research_output)
    print(f"[Agent 02] Saved: {analyst_file}")

    # ===== PHASE 3: Rotation Detector =====
    print(f"[Agent 03] Running Rotation Detection pipeline ...")
    rotation_output = run_rotation_pipeline(analyst_output, research_output)
    print(f"[Agent 03] Done — verdict={rotation_output.verdict.value}, "
          f"{rotation_output.signals.signals_active}/5 signals, "
          f"type={rotation_output.rotation_type.value}")

    rotation_file = _write_rotation_excel(rotation_output, out_path)
    print(f"[Agent 03] Saved: {rotation_file}")

    # ===== PHASE 4: Sector Strategist =====
    print(f"[Agent 04] Running Sector Strategy pipeline ...")
    strategy_output = run_strategist_pipeline(rotation_output, analyst_output)
    print(f"[Agent 04] Done — "
          f"{len(strategy_output.sector_allocations.overall_allocation)} sectors allocated, "
          f"{len(strategy_output.theme_recommendations)} themes, "
          f"{len(strategy_output.rotation_signals)} signals")

    strategy_file = _write_strategy_excel(strategy_output, out_path)
    print(f"[Agent 04] Saved: {strategy_file}")

    # ===== PHASE 5: Portfolio Manager =====
    print(f"[Agent 05] Running Portfolio Manager pipeline ...")
    portfolio_output = run_portfolio_pipeline(strategy_output, analyst_output)
    print(f"[Agent 05] Done — {portfolio_output.total_positions} positions, "
          f"{portfolio_output.stock_count} stocks, {portfolio_output.etf_count} ETFs, "
          f"{portfolio_output.keeps_placement.total_keeps} keeps")

    portfolio_file = _write_portfolio_excel(portfolio_output, out_path)
    print(f"[Agent 05] Saved: {portfolio_file}")

    # ===== PHASE 6: Risk Officer =====
    print(f"[Agent 06] Running Risk Officer pipeline ...")
    risk_output = run_risk_pipeline(portfolio_output, strategy_output)
    print(f"[Agent 06] Done — {risk_output.overall_status}, "
          f"Sleep Well: {risk_output.sleep_well_scores.overall_score}/10, "
          f"{len(risk_output.vetoes)} vetoes, {len(risk_output.warnings)} warnings")

    risk_file = _write_risk_excel(risk_output, out_path)
    print(f"[Agent 06] Saved: {risk_file}")

    # ===== PHASE 7: Returns Projector =====
    print(f"[Agent 07] Running Returns Projector pipeline ...")
    returns_output = run_returns_projector_pipeline(
        portfolio_output, rotation_output, risk_output, analyst_output,
    )
    print(f"[Agent 07] Done — Expected 12m: {returns_output.expected_return.expected_12m:.1f}%, "
          f"Alpha vs SPY: {returns_output.alpha_analysis.net_expected_alpha_vs_spy:.1f}%, "
          f"DD w/stops: {returns_output.risk_metrics.max_drawdown_with_stops:.1f}%")

    returns_file = _write_returns_projector_excel(returns_output, out_path)
    print(f"[Agent 07] Saved: {returns_file}")

    # ===== PHASE 8: Portfolio Reconciler =====
    print(f"[Agent 08] Running Reconciler pipeline ...")
    reconciler_output = run_reconciler_pipeline(
        portfolio_output, analyst_output, returns_output=returns_output,
        rotation_output=rotation_output, strategy_output=strategy_output,
        risk_output=risk_output,
    )
    print(f"[Agent 08] Done — {len(reconciler_output.actions)} actions, "
          f"turnover: {reconciler_output.transformation_metrics.turnover_pct:.0f}%")

    reconciler_file = _write_reconciler_excel(reconciler_output, out_path)
    print(f"[Agent 08] Saved: {reconciler_file}")

    # ===== PHASE 9: Educator =====
    print(f"[Agent 09] Running Educator pipeline ...")
    educator_output = run_educator_pipeline(
        portfolio_output, analyst_output, rotation_output, strategy_output,
        risk_output, reconciler_output, returns_output=returns_output,
    )
    print(f"[Agent 09] Done — {len(educator_output.stock_explanations)} stocks explained, "
          f"{len(educator_output.concept_lessons)} lessons, "
          f"{len(educator_output.glossary)} glossary entries")

    educator_file = _write_educator_excel(educator_output, out_path)
    print(f"[Agent 09] Saved: {educator_file}")

    # ===== PHASE 10: Executive Summary Synthesizer =====
    print(f"[Agent 10] Running Executive Summary Synthesizer pipeline ...")
    synthesis_output = run_synthesizer_pipeline(
        research_output, analyst_output, rotation_output, strategy_output,
        portfolio_output, risk_output, returns_output, reconciler_output,
        educator_output,
    )
    print(f"[Agent 10] Done — source={synthesis_output.synthesis_source}, "
          f"{len(synthesis_output.cross_agent_connections)} connections, "
          f"{len(synthesis_output.contradictions)} contradictions")

    synthesis_file = _write_synthesizer_excel(synthesis_output, out_path)
    print(f"[Agent 10] Saved: {synthesis_file}")

    print(f"\nOutput directory: {out_path}/")


if __name__ == "__main__":
    data = sys.argv[1] if len(sys.argv) > 1 else "data"
    out = sys.argv[2] if len(sys.argv) > 2 else "output"
    main(data, out)
