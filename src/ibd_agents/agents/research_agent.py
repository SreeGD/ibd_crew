"""
Agent 01: Research Agent 🔍
Senior Investment Research Analyst — IBD Momentum Framework v4.0

Reads all data files, extracts stocks with IBD ratings,
calculates multi-source validation scores, tags Schwab themes,
flags IBD Keep candidates, and curates the top 200 growth stock
opportunities with tier-ready data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

try:
    from crewai import Agent, Task
    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False
    Agent = None  # type: ignore
    Task = None  # type: ignore

from ibd_agents.schemas.research_output import (
    IBD_SECTORS,
    ResearchETF,
    ResearchOutput,
    ResearchStock,
    SectorPattern,
    compute_preliminary_tier,
    is_ibd_keep_candidate,
)
from ibd_agents.tools.file_lister import ListDataFilesTool, list_data_files
from ibd_agents.tools.morningstar_reader import ReadMorningstarPDFTool, read_morningstar_pdf
from ibd_agents.tools.pdf_reader import ReadFoolPDFTool, ReadIBDPDFTool, read_fool_pdf, read_ibd_pdf
from ibd_agents.tools.theme_reader import ReadSchwabThemeTool, read_schwab_theme
from ibd_agents.tools.validation_scorer import (
    CalculateValidationScoreTool,
    StockUniverse,
)
from ibd_agents.tools.cap_classifier import CapClassifierTool, classify_caps_static
from ibd_agents.tools.sector_classifier import classify_sectors_static
from ibd_agents.tools.theme_reader import KNOWN_ETFS as THEME_KNOWN_ETFS
from ibd_agents.tools.xls_reader import ReadIBDCSVTool, ReadIBDXLSTool, read_ibd_csv, read_ibd_xls

logger = logging.getLogger(__name__)


def build_research_agent() -> "Agent":
    """Create the Research Agent with all tools. Requires crewai."""
    if not HAS_CREWAI:
        raise ImportError("crewai is required for agentic mode. pip install crewai[tools]")
    return Agent(
        role="Senior Investment Research Analyst",
        goal=(
            "Process all data files per Framework v4.0 source hierarchy, "
            "extract stocks with IBD ratings, calculate multi-source validation "
            "scores, tag Schwab themes, flag IBD Keep candidates, and curate "
            "the top 200 growth stock opportunities with tier-ready data."
        ),
        backstory=(
            "You are a 15-year veteran of IBD growth stock methodology. "
            "You systematically process every data source — IBD Excel files "
            "first (highest weight, all 5 ratings), then IBD PDFs, then "
            "Motley Fool for supplementary validation, and Schwab themes last "
            "for thematic tagging. You never make buy/sell recommendations — "
            "you only discover, rate, and organize. Your output feeds the "
            "Analyst Agent for deep analysis."
        ),
        tools=[
            ListDataFilesTool(),
            ReadIBDXLSTool(),
            ReadIBDCSVTool(),
            ReadIBDPDFTool(),
            ReadFoolPDFTool(),
            ReadMorningstarPDFTool(),
            ReadSchwabThemeTool(),
            CalculateValidationScoreTool(),
            CapClassifierTool(),
        ],
        verbose=True,
        allow_delegation=False,
        max_iter=15,
    )


def build_research_task(agent: "Agent", data_dir: str = "data") -> "Task":
    """Create the Research task. Requires crewai."""
    if not HAS_CREWAI:
        raise ImportError("crewai is required for agentic mode. pip install crewai[tools]")
    return Task(
        description=f"""Process all data files in '{data_dir}/' per Framework v4.0 source hierarchy.

PROCESSING ORDER (per spec §6 Decision 1):
1. IBD XLS files first (structured, highest weight, all 5 ratings)
2. IBD PDF files (Smart Table broadest, then Top 200, then remaining)
3. Motley Fool PDFs (supplementary validation)
3b. Morningstar Pick List PDFs (star ratings for validation scoring)
4. Schwab theme PDFs last (tagging, not primary screening)

FOR EACH STOCK:
- Extract all 5 IBD ratings: Composite, RS, EPS, SMR, Acc/Dis
- Calculate multi-source validation score using exact Framework v4.0 points
- Tag Schwab theme memberships
- Flag IBD Keep candidates (Composite ≥93 AND RS ≥90)
- Assign preliminary tier (T1/T2/T3) using rating thresholds
- Separate stocks from ETFs

OUTPUT:
Return ONLY a valid JSON object matching the ResearchOutput schema with:
- stocks: 100-200 individual stocks
- etfs: ETFs found across sources
- sector_patterns: sector-level observations
- ibd_keep_candidates: symbols with Comp≥93 AND RS≥90
- multi_source_validated: symbols with score≥5 from 2+ providers
""",
        expected_output=(
            "A JSON object with stocks (100-200), etfs, sector_patterns, "
            "data_sources_used, ibd_keep_candidates, multi_source_validated, "
            "analysis_date, and summary."
        ),
        agent=agent,
    )


def run_research_pipeline(data_dir: str = "data") -> ResearchOutput:
    """
    Run the deterministic research pipeline without LLM.

    This is the programmatic path that processes files directly.
    For the agentic path, use build_research_agent() + build_research_task().

    Args:
        data_dir: Root directory containing data subdirectories

    Returns:
        Validated ResearchOutput
    """
    data_path = Path(data_dir)
    universe = StockUniverse()

    sources_used: list[str] = []
    sources_failed: list[str] = []

    # --- Phase 1: IBD XLS (highest priority) ---
    xls_dir = data_path / "ibd_xls"
    if xls_dir.exists():
        for f in sorted(xls_dir.glob("*.xls*")):
            try:
                records = read_ibd_xls(str(f))
                if records:
                    universe.add_ibd_data(records)
                    sources_used.append(f.name)
                else:
                    sources_failed.append(f.name)
            except Exception as e:
                logger.error(f"Failed to read {f.name}: {e}")
                sources_failed.append(f.name)

    # --- Phase 1b: IBD CSV (same priority as XLS) ---
    if xls_dir.exists():
        for f in sorted(xls_dir.glob("*.csv")):
            try:
                records = read_ibd_csv(str(f))
                if records:
                    universe.add_ibd_data(records)
                    sources_used.append(f.name)
                else:
                    sources_failed.append(f.name)
            except Exception as e:
                logger.error(f"Failed to read {f.name}: {e}")
                sources_failed.append(f.name)

    # --- Phase 2: IBD PDFs ---
    pdf_dir = data_path / "ibd_pdf"
    if pdf_dir.exists():
        for f in sorted(pdf_dir.glob("*.pdf")):
            try:
                records = read_ibd_pdf(str(f))
                if records:
                    universe.add_ibd_data(records)
                    sources_used.append(f.name)
                else:
                    sources_failed.append(f.name)
            except Exception as e:
                logger.error(f"Failed to read {f.name}: {e}")
                sources_failed.append(f.name)

    # --- Phase 3: Motley Fool PDFs ---
    fool_dir = data_path / "fool_pdf"
    if fool_dir.exists():
        for f in sorted(fool_dir.glob("*.pdf")):
            try:
                records = read_fool_pdf(str(f))
                if records:
                    universe.add_fool_data(records)
                    sources_used.append(f.name)
                else:
                    sources_failed.append(f.name)
            except Exception as e:
                logger.error(f"Failed to read {f.name}: {e}")
                sources_failed.append(f.name)

    # --- Phase 3b: Morningstar Pick Lists (from data/morningstar/) ---
    ms_dir = Path(data_dir) / "morningstar"
    if ms_dir.exists():
        for f in sorted(ms_dir.glob("*.pdf")):
            try:
                records = read_morningstar_pdf(str(f))
                if records:
                    universe.add_morningstar_data(records)
                    cap = records[0].get("cap_list", "unknown")
                    friendly = "Morningstar Large Cap" if cap == "large" else "Morningstar Mid Cap" if cap == "mid" else "Morningstar"
                    sources_used.append(friendly)
                else:
                    sources_failed.append(f.name)
            except Exception as e:
                logger.error(f"Failed to read {f.name}: {e}")
                sources_failed.append(f.name)

    # --- Phase 4: Schwab Themes (tagging) ---
    theme_dir = data_path / "schwab_themes"
    if theme_dir.exists():
        for f in sorted(theme_dir.glob("*.pdf")):
            try:
                records = read_schwab_theme(str(f))
                if records:
                    universe.add_theme_data(records)
                    sources_used.append(f.name)
                # Not counting as failure if overview or empty theme
            except Exception as e:
                logger.error(f"Failed to read {f.name}: {e}")
                sources_failed.append(f.name)

    # --- Phase 5: Sector classification for remaining unknowns ---
    sectorless = universe.get_sectorless_symbols()
    if sectorless:
        static_map = classify_sectors_static(sectorless)
        for sym, sector in static_map.items():
            universe.set_sector(sym, sector)
        remaining = len(sectorless) - len(static_map)
        if static_map:
            logger.info(f"Static sector classifier: assigned {len(static_map)} stocks, {remaining} still unknown")

        # LLM classification for remaining unknowns
        still_sectorless = universe.get_sectorless_symbols()
        if still_sectorless:
            try:
                from ibd_agents.tools.sector_classifier import classify_sectors_llm
                llm_map = classify_sectors_llm(still_sectorless)
                for sym, sector in llm_map.items():
                    universe.set_sector(sym, sector)
                logger.info(f"LLM sector classifier: assigned {len(llm_map)} stocks, "
                            f"{len(still_sectorless) - len(llm_map)} still unknown")
            except Exception as e:
                logger.warning(f"LLM sector classification failed: {e}")

    # --- Phase 5b: Cap size classification ---
    ibd_lists_map = {
        sym: agg.ibd_lists for sym, agg in universe._stocks.items()
        if not agg.is_etf and agg.ibd_lists
    }
    all_stock_syms = [sym for sym, agg in universe._stocks.items() if not agg.is_etf]
    cap_map = classify_caps_static(all_stock_syms, ibd_lists_map=ibd_lists_map)
    for sym, cap in cap_map.items():
        universe.set_cap_size(sym, cap)
    if cap_map:
        logger.info(f"Static cap classifier: assigned {len(cap_map)}/{len(all_stock_syms)} stocks")

    unsized = universe.get_unsized_symbols()
    if unsized:
        try:
            from ibd_agents.tools.cap_classifier import classify_caps_llm
            llm_cap_map = classify_caps_llm(unsized)
            for sym, cap in llm_cap_map.items():
                universe.set_cap_size(sym, cap)
            logger.info(f"LLM cap classifier: assigned {len(llm_cap_map)} stocks, "
                        f"{len(unsized) - len(llm_cap_map)} still unknown")
        except Exception as e:
            logger.warning(f"LLM cap classification failed: {e}")

    # --- Build output ---
    all_scored = universe.get_all_scored()
    total_scanned = universe.count

    # Separate stocks and ETFs
    stocks: list[dict] = []
    etfs: list[dict] = []

    for item in all_scored:
        if item.get("is_etf"):
            etfs.append(item)
        else:
            stocks.append(item)

    # Compute sector patterns
    sector_data: dict[str, list[dict]] = {}
    for s in stocks:
        sec = s.get("sector", "UNKNOWN")
        if sec not in sector_data:
            sector_data[sec] = []
        sector_data[sec].append(s)

    sector_patterns: list[SectorPattern] = []
    for sector, sector_stocks in sector_data.items():
        if sector not in IBD_SECTORS:
            continue

        comps = [s["composite_rating"] for s in sector_stocks if s.get("composite_rating")]
        rs_vals = [s["rs_rating"] for s in sector_stocks if s.get("rs_rating")]
        avg_comp = sum(comps) / len(comps) if comps else None
        avg_rs = sum(rs_vals) / len(rs_vals) if rs_vals else None

        # Elite = all 4 gates pass
        elite_count = sum(
            1 for s in sector_stocks
            if (s.get("composite_rating") or 0) >= 85
            and (s.get("rs_rating") or 0) >= 75
            and s.get("smr_rating") in ("A", "B", "B-")
            and s.get("acc_dis_rating") in ("A", "B", "B-")
        )
        elite_pct = (elite_count / len(sector_stocks) * 100) if sector_stocks else 0

        multi_list_count = sum(1 for s in sector_stocks if len(s.get("ibd_lists", [])) >= 2)
        multi_list_pct = (multi_list_count / len(sector_stocks) * 100) if sector_stocks else 0

        keep_count = sum(
            1 for s in sector_stocks
            if is_ibd_keep_candidate(s.get("composite_rating"), s.get("rs_rating"))
        )

        # Determine strength and trend
        if avg_comp and avg_comp >= 90 and avg_rs and avg_rs >= 85:
            strength = "leading"
            trend = "up"
        elif avg_comp and avg_comp >= 80:
            strength = "improving"
            trend = "up"
        elif avg_comp and avg_comp >= 70:
            strength = "lagging"
            trend = "flat"
        else:
            strength = "declining"
            trend = "down"

        sector_patterns.append(SectorPattern(
            sector=sector,
            stock_count=len(sector_stocks),
            avg_composite=round(avg_comp, 1) if avg_comp else None,
            avg_rs=round(avg_rs, 1) if avg_rs else None,
            elite_pct=round(elite_pct, 1),
            multi_list_pct=round(multi_list_pct, 1),
            ibd_keep_count=keep_count,
            strength=strength,
            trend_direction=trend,
            evidence=f"{len(sector_stocks)} stocks, avg Comp={avg_comp:.0f}" if avg_comp else f"{len(sector_stocks)} stocks, ratings unavailable",
        ))

    # Compute sector rankings and stock ranks within sector for stocks missing them.
    # Rank sectors by avg_composite (desc), rank stocks within sector by composite (desc).
    _sector_avg_comp: dict[str, float] = {}
    for sp in sector_patterns:
        if sp.avg_composite is not None:
            _sector_avg_comp[sp.sector] = sp.avg_composite
    _ranked_sectors = sorted(_sector_avg_comp.keys(), key=lambda sec: _sector_avg_comp[sec], reverse=True)
    _sector_rank_map = {sec: rank + 1 for rank, sec in enumerate(_ranked_sectors)}

    # Rank stocks within each sector by composite desc, then RS desc
    for sector, sector_stocks in sector_data.items():
        if sector not in _sector_rank_map:
            continue
        sorted_stocks = sorted(
            sector_stocks,
            key=lambda s: (s.get("composite_rating") or 0, s.get("rs_rating") or 0),
            reverse=True,
        )
        for rank_pos, s in enumerate(sorted_stocks, start=1):
            if s.get("sector_rank") is None:
                s["sector_rank"] = _sector_rank_map[sector]
            if s.get("stock_rank_in_sector") is None:
                s["stock_rank_in_sector"] = rank_pos

    # Build ResearchStock objects
    research_stocks: list[ResearchStock] = []
    for s in stocks:

        tier = compute_preliminary_tier(
            s.get("composite_rating"), s.get("rs_rating"), s.get("eps_rating")
        )
        keep = is_ibd_keep_candidate(s.get("composite_rating"), s.get("rs_rating"))

        # Calculate confidence based on data completeness
        has_ratings = sum(1 for r in [s.get("composite_rating"), s.get("rs_rating"),
                                       s.get("eps_rating"), s.get("smr_rating"),
                                       s.get("acc_dis_rating")] if r is not None)
        confidence = min(1.0, 0.3 + (has_ratings * 0.14))

        # Build reasoning
        parts = []
        if s.get("composite_rating"):
            parts.append(f"IBD Comp {s['composite_rating']}")
        if s.get("ibd_lists"):
            parts.append(f"on {len(s['ibd_lists'])} IBD lists")
        if s.get("schwab_themes"):
            parts.append(f"{len(s['schwab_themes'])} Schwab themes")
        if s.get("is_multi_source_validated"):
            parts.append(f"multi-source validated (score {s['validation_score']})")
        reasoning = ", ".join(parts) if parts else "Discovered in data scan"
        if len(reasoning) < 20:
            reasoning = reasoning + " — awaiting deeper analysis"

        try:
            rs = ResearchStock(
                symbol=s["symbol"],
                company_name=s.get("company_name", s["symbol"]),
                sector=s["sector"],
                security_type="stock",
                composite_rating=s.get("composite_rating"),
                rs_rating=s.get("rs_rating"),
                eps_rating=s.get("eps_rating"),
                smr_rating=s.get("smr_rating"),
                acc_dis_rating=s.get("acc_dis_rating"),
                cap_size=s.get("cap_size"),
                morningstar_rating=s.get("morningstar_rating"),
                economic_moat=s.get("economic_moat"),
                fair_value=s.get("fair_value"),
                morningstar_price=s.get("morningstar_price"),
                price_to_fair_value=s.get("price_to_fair_value"),
                morningstar_uncertainty=s.get("morningstar_uncertainty"),
                ibd_lists=s.get("ibd_lists", []),
                schwab_themes=s.get("schwab_themes", []),
                fool_status=s.get("fool_status"),
                other_ratings=s.get("other_ratings", {}),
                validation_score=s.get("validation_score", 0),
                validation_providers=s.get("validation_providers", 0),
                is_multi_source_validated=s.get("is_multi_source_validated", False),
                is_ibd_keep_candidate=keep,
                preliminary_tier=tier,
                sector_rank=s.get("sector_rank"),
                stock_rank_in_sector=s.get("stock_rank_in_sector"),
                sources=s.get("sources", ["unknown"]),
                confidence=confidence,
                reasoning=reasoning,
            )
            research_stocks.append(rs)
        except Exception as e:
            logger.warning(f"Skipping {s['symbol']}: {e}")

    # Build ETF objects with full data, scoring, and ranking
    research_etfs: list[ResearchETF] = []
    for e in etfs:
        try:
            # Determine if this is a key theme ETF
            is_key = e["symbol"] in THEME_KNOWN_ETFS

            # Compute ETF composite score from available metrics
            etf_score = _compute_etf_score(
                e.get("rs_rating"), e.get("acc_dis_rating"),
                e.get("ytd_change"), e.get("volume_pct_change"),
            )

            # Assign preliminary tier from theme classification
            themes = e.get("schwab_themes", [])
            tier = _assign_etf_tier(themes)

            re = ResearchETF(
                symbol=e["symbol"],
                name=e.get("company_name", e["symbol"]),
                focus=themes[0] if themes else "General",
                schwab_themes=themes,
                ibd_lists=e.get("ibd_lists", []),
                sources=e.get("sources", ["unknown"]),
                key_theme_etf=is_key,
                rs_rating=e.get("rs_rating"),
                acc_dis_rating=e.get("acc_dis_rating"),
                ytd_change=e.get("ytd_change"),
                close_price=e.get("close_price"),
                price_change=e.get("price_change"),
                volume_pct_change=e.get("volume_pct_change"),
                div_yield=e.get("div_yield"),
                preliminary_tier=tier,
                etf_score=etf_score,
            )
            research_etfs.append(re)
        except Exception as ex:
            logger.warning(f"Skipping ETF {e['symbol']}: {ex}")

    # Rank ETFs by score (highest first)
    scored_etfs = [(i, etf) for i, etf in enumerate(research_etfs) if etf.etf_score is not None]
    scored_etfs.sort(key=lambda x: x[1].etf_score or 0, reverse=True)
    for rank, (idx, _) in enumerate(scored_etfs, start=1):
        research_etfs[idx].etf_rank = rank

    # Keep candidates and validated lists
    keep_candidates = [s.symbol for s in research_stocks if s.is_ibd_keep_candidate]
    multi_validated = [s.symbol for s in research_stocks if s.is_multi_source_validated]

    # Build summary
    summary = (
        f"Processed {len(sources_used)} files across IBD, Schwab, and Motley Fool sources. "
        f"Discovered {len(research_stocks)} stocks and {len(research_etfs)} ETFs across "
        f"{len(set(s.sector for s in research_stocks))} sectors. "
        f"Found {len(keep_candidates)} IBD keep candidates and "
        f"{len(multi_validated)} multi-source validated stocks."
    )

    from datetime import date
    output = ResearchOutput(
        stocks=research_stocks,
        etfs=research_etfs,
        sector_patterns=sector_patterns,
        data_sources_used=sources_used,
        data_sources_failed=sources_failed,
        total_securities_scanned=total_scanned,
        ibd_keep_candidates=keep_candidates,
        multi_source_validated=multi_validated,
        analysis_date=date.today().isoformat(),
        summary=summary,
    )

    return output


# ---------------------------------------------------------------------------
# ETF Scoring and Tiering Helpers
# ---------------------------------------------------------------------------

# Acc/Dis letter → numeric for scoring
_ACC_DIS_NUMERIC = {"A": 5, "B": 4, "B-": 3, "C": 2, "D": 1, "E": 0}


def _compute_etf_score(
    rs_rating: int | None,
    acc_dis_rating: str | None,
    ytd_change: float | None,
    volume_pct_change: float | int | None,
) -> float | None:
    """
    Compute composite ETF ranking score from available metrics.

    Weights: RS×0.35 + Acc/Dis×0.25 + YTD×0.25 + Vol%×0.15
    All components normalized to 0-100 scale.
    Returns None if no data available.
    """
    components: list[tuple[float, float]] = []  # (value_0_100, weight)

    if rs_rating is not None:
        components.append((float(rs_rating), 0.35))

    if acc_dis_rating is not None:
        numeric = _ACC_DIS_NUMERIC.get(acc_dis_rating, 2)
        components.append((numeric * 20.0, 0.25))  # Scale 0-5 → 0-100

    if ytd_change is not None:
        # Clamp YTD to [-50, 100] then normalize to 0-100
        clamped = max(-50.0, min(100.0, ytd_change))
        normalized = (clamped + 50.0) / 150.0 * 100.0
        components.append((normalized, 0.25))

    if volume_pct_change is not None:
        # Clamp vol change to [-100, 200] then normalize to 0-100
        clamped = max(-100.0, min(200.0, float(volume_pct_change)))
        normalized = (clamped + 100.0) / 300.0 * 100.0
        components.append((normalized, 0.15))

    if not components:
        return None

    # Weighted average, renormalized if not all components available
    total_weight = sum(w for _, w in components)
    score = sum(v * w for v, w in components) / total_weight
    return round(score, 2)


def _assign_etf_tier(themes: list[str]) -> int | None:
    """
    Assign preliminary tier based on theme classification.

    Growth themes → T1, Defensive themes → T3, else → T2.
    Returns None if no themes available.
    """
    # Import here to avoid circular dependency
    from ibd_agents.schemas.strategy_output import GROWTH_THEMES, DEFENSIVE_THEMES

    if not themes:
        return None

    has_growth = any(t in GROWTH_THEMES for t in themes)
    has_defensive = any(t in DEFENSIVE_THEMES for t in themes)

    if has_growth and not has_defensive:
        return 1
    elif has_defensive and not has_growth:
        return 3
    else:
        return 2
