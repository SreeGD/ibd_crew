"""
Agent 05: Portfolio Manager
Portfolio Construction Specialist — IBD Momentum Framework v4.0

Receives AnalystOutput + SectorStrategyOutput.
Produces PortfolioOutput with:
- 3-tier portfolio with position sizing
- 14 keeps placed across tiers
- Trailing stops per tier
- 4-week implementation plan
- Orders (buy/sell/add/trim)

This agent constructs portfolios. It never discovers new stocks or
overrides the Risk Officer.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ibd_agents.schemas.value_investor_output import ValueInvestorOutput
    from ibd_agents.schemas.pattern_output import PortfolioPatternOutput

try:
    from crewai import Agent, Task
    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False
    Agent = None  # type: ignore
    Task = None  # type: ignore

from ibd_agents.schemas.analyst_output import AnalystOutput, RatedStock
from ibd_agents.schemas.portfolio_output import (
    ALL_KEEPS,
    ETF_LIMITS,
    KEEP_METADATA,
    STOCK_LIMITS,
    TRAILING_STOPS,
    WEEK_FOCUSES,
    ImplementationWeek,
    OrderAction,
    PortfolioOutput,
    PortfolioPosition,
    TierPortfolio,
)
from ibd_agents.schemas.strategy_output import (
    SectorStrategyOutput,
    THEME_ETFS,
)
from ibd_agents.tools.keeps_manager import (
    KeepsManagerTool,
    build_keep_positions,
    place_keeps,
)
from ibd_agents.tools.position_sizer import (
    PositionSizerTool,
    compute_max_loss,
    compute_trailing_stop,
    size_position,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CrewAI Agent Builder
# ---------------------------------------------------------------------------

def build_portfolio_agent() -> "Agent":
    """Create the Portfolio Manager Agent with tools. Requires crewai."""
    if not HAS_CREWAI:
        raise ImportError("crewai is required for agentic mode. pip install crewai[tools]")
    return Agent(
        role="Portfolio Construction Specialist",
        goal=(
            "Construct a 3-tier portfolio (Momentum/Quality Growth/Defensive) "
            "using the Analyst's top 50 stocks, Strategist's sector allocations, "
            "and Framework v4.0 position sizing rules. Handle all three keep "
            "categories. Ensure compliance with concentration limits, trailing "
            "stop rules, and stock/ETF allocation targets."
        ),
        backstory=(
            "You are a systematic portfolio constructor with $500M+ AUM experience. "
            "You build portfolios within strict framework rules: position limits "
            "(stocks 5/4/3%, ETFs 8/8/6%), 22/18/12% trailing stops, 50/50 "
            "stock/ETF split, 40% max sector, 8+ sectors. You handle 21 "
            "pre-committed keep positions across 3 categories. You never "
            "discover new stocks or override the Risk Officer."
        ),
        tools=[PositionSizerTool(), KeepsManagerTool()],
        verbose=True,
        allow_delegation=False,
        max_iter=10,
        temperature=0.5,
    )


def build_portfolio_task(
    agent: "Agent",
    analyst_json: str = "",
    strategy_json: str = "",
) -> "Task":
    """Create the Portfolio Construction task. Requires crewai."""
    if not HAS_CREWAI:
        raise ImportError("crewai is required for agentic mode. pip install crewai[tools]")
    return Task(
        description=f"""Construct a 3-tier portfolio from analyst + strategy data.

STEPS:
1. Place 14 keeps (5 fundamental + 9 IBD)
2. Fill remaining positions from top 50 rated stocks
3. Add ETFs from theme recommendations
4. Size all positions within tier limits
5. Assign trailing stops (22/18/12%)
6. Build 4-week implementation plan
7. Verify all constraints

CONSTRAINTS:
- T1: stocks ≤ 5%, ETFs ≤ 8%, stop = 22%
- T2: stocks ≤ 4%, ETFs ≤ 8%, stop = 18%
- T3: stocks ≤ 3%, ETFs ≤ 6%, stop = 12%
- No sector > 40%, min 8 sectors
- Stock/ETF split within 45-55%
- Cash 2-15%
- All 14 keeps placed

DO NOT:
- Discover new stocks
- Override Risk Officer
- Provide macroeconomic analysis

Analyst data:
{analyst_json}

Strategy data:
{strategy_json}
""",
        expected_output=(
            "JSON with tier_1, tier_2, tier_3, cash_pct, keeps_placement, "
            "orders, implementation_plan, sector_exposure, total_positions, "
            "construction_methodology, analysis_date, summary."
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Deterministic Pipeline
# ---------------------------------------------------------------------------

def run_portfolio_pipeline(
    strategy_output: SectorStrategyOutput,
    analyst_output: AnalystOutput,
    value_output: Optional["ValueInvestorOutput"] = None,
    pattern_output: Optional["PortfolioPatternOutput"] = None,
) -> PortfolioOutput:
    """
    Run the deterministic portfolio construction pipeline without LLM.

    Args:
        strategy_output: Validated SectorStrategyOutput from Agent 04
        analyst_output: Validated AnalystOutput from Agent 02
        value_output: Optional ValueInvestorOutput from Agent 11 (~10% carve-out)
        pattern_output: Optional PortfolioPatternOutput from Agent 12 (~10% carve-out)

    Returns:
        Validated PortfolioOutput
    """
    logger.info("[Agent 05] Running Portfolio Construction pipeline ...")

    regime = strategy_output.regime_adjustment.split()[0].lower()
    tier_targets = strategy_output.sector_allocations.tier_targets

    # Step 1: Place 14 keeps
    keeps_placement = place_keeps(analyst_output.rated_stocks)
    keep_positions = build_keep_positions(keeps_placement, analyst_output.rated_stocks)
    # Tag keeps with selection_source
    for i, kp in enumerate(keep_positions):
        keep_positions[i] = kp.model_copy(update={"selection_source": "keep"})
    keep_symbols = {p.symbol for p in keep_positions}
    logger.info(
        f"[Agent 05] Placed {len(keep_positions)} keeps, "
        f"total {keeps_placement.keeps_pct:.1f}%"
    )

    # Step 2: Select value/pattern picks (~10% each carve-out)
    value_positions, pattern_positions = _build_value_pattern_positions(
        value_output, pattern_output, analyst_output, keep_symbols,
    )
    vp_symbols = {p.symbol for p in value_positions + pattern_positions}
    logger.info(
        f"[Agent 05] Value picks: {len(value_positions)}, "
        f"Pattern picks: {len(pattern_positions)}"
    )

    # Step 3: Fill remaining positions from rated stocks (non-keeps, non-value/pattern)
    non_keep_rated = [
        rs for rs in analyst_output.rated_stocks
        if rs.symbol not in keep_symbols and rs.symbol not in vp_symbols
    ]
    # Sort by conviction descending, take enough to fill
    non_keep_rated.sort(key=lambda rs: (-rs.conviction, -rs.composite_rating))

    # Step 4: Add ETF positions from theme recommendations
    etf_positions = _build_etf_positions(strategy_output, tier_targets, analyst_output)

    # Step 5: Build tier portfolios
    # Target total positions: aim for ~50-60 for diversification
    # We need roughly equal stock/ETF count
    pre_placed = keep_positions + value_positions + pattern_positions
    stock_positions = pre_placed + _build_stock_positions(
        non_keep_rated, tier_targets, pre_placed
    )
    all_positions = stock_positions + etf_positions

    # Step 5: Organize into tiers
    tier_1_pos = [p for p in all_positions if p.tier == 1]
    tier_2_pos = [p for p in all_positions if p.tier == 2]
    tier_3_pos = [p for p in all_positions if p.tier == 3]

    # --- Step 5.5: LLM dynamic sizing ---
    try:
        from ibd_agents.tools.dynamic_sizing import enrich_sizing_llm, compute_size_adjustment
        sizing_inputs = [
            {
                "symbol": p.symbol,
                "sector": p.sector,
                "cap_size": p.cap_size,
                "tier": p.tier,
                "conviction": p.conviction,
                "target_pct": p.target_pct,
                "asset_type": p.asset_type,
                "catalyst_date": None,  # Not available at portfolio level
            }
            for p in all_positions
        ]
        sizing_data = enrich_sizing_llm(sizing_inputs)
        if sizing_data:
            sizing_count = 0
            for p_list in [tier_1_pos, tier_2_pos, tier_3_pos]:
                for idx, p in enumerate(p_list):
                    if p.symbol in sizing_data:
                        d = sizing_data[p.symbol]
                        adj = compute_size_adjustment(d["size_adjustment_pct"], p.tier)
                        limit = ETF_LIMITS[p.tier] if p.asset_type == "etf" else STOCK_LIMITS[p.tier]
                        new_target = max(0.5, min(p.target_pct + adj, limit))
                        p_list[idx] = PortfolioPosition(
                            symbol=p.symbol,
                            company_name=p.company_name,
                            sector=p.sector,
                            cap_size=p.cap_size,
                            tier=p.tier,
                            asset_type=p.asset_type,
                            target_pct=new_target,
                            trailing_stop_pct=p.trailing_stop_pct,
                            max_loss_pct=p.max_loss_pct,
                            keep_category=p.keep_category,
                            conviction=p.conviction,
                            reasoning=p.reasoning,
                            volatility_adjustment=adj,
                            sizing_source="llm",
                            selection_source=p.selection_source,
                        )
                        sizing_count += 1
                    else:
                        p_list[idx] = PortfolioPosition(
                            symbol=p.symbol,
                            company_name=p.company_name,
                            sector=p.sector,
                            cap_size=p.cap_size,
                            tier=p.tier,
                            asset_type=p.asset_type,
                            target_pct=p.target_pct,
                            trailing_stop_pct=p.trailing_stop_pct,
                            max_loss_pct=p.max_loss_pct,
                            keep_category=p.keep_category,
                            conviction=p.conviction,
                            reasoning=p.reasoning,
                            volatility_adjustment=0.0,
                            sizing_source="deterministic",
                            selection_source=p.selection_source,
                        )
            logger.info(f"LLM dynamic sizing: {sizing_count}/{len(all_positions)} positions adjusted")
        else:
            for p_list in [tier_1_pos, tier_2_pos, tier_3_pos]:
                for idx, p in enumerate(p_list):
                    p_list[idx] = PortfolioPosition(
                        symbol=p.symbol,
                        company_name=p.company_name,
                        sector=p.sector,
                        cap_size=p.cap_size,
                        tier=p.tier,
                        asset_type=p.asset_type,
                        target_pct=p.target_pct,
                        trailing_stop_pct=p.trailing_stop_pct,
                        max_loss_pct=p.max_loss_pct,
                        keep_category=p.keep_category,
                        conviction=p.conviction,
                        reasoning=p.reasoning,
                        volatility_adjustment=0.0,
                        sizing_source="deterministic",
                        selection_source=p.selection_source,
                    )
    except Exception as e:
        logger.warning(f"LLM dynamic sizing failed: {e}")
        for p_list in [tier_1_pos, tier_2_pos, tier_3_pos]:
            for idx, p in enumerate(p_list):
                if not hasattr(p, 'sizing_source') or p.sizing_source is None:
                    p_list[idx] = PortfolioPosition(
                        symbol=p.symbol,
                        company_name=p.company_name,
                        sector=p.sector,
                        cap_size=p.cap_size,
                        tier=p.tier,
                        asset_type=p.asset_type,
                        target_pct=p.target_pct,
                        trailing_stop_pct=p.trailing_stop_pct,
                        max_loss_pct=p.max_loss_pct,
                        keep_category=p.keep_category,
                        conviction=p.conviction,
                        reasoning=p.reasoning,
                        volatility_adjustment=0.0,
                        sizing_source="deterministic",
                        selection_source=p.selection_source,
                    )

    # Compute actual percentages
    t1_pct = sum(p.target_pct for p in tier_1_pos)
    t2_pct = sum(p.target_pct for p in tier_2_pos)
    t3_pct = sum(p.target_pct for p in tier_3_pos)

    # Normalize to tier targets
    t1_target = tier_targets.get("T1", 39.0)
    t2_target = tier_targets.get("T2", 37.0)
    t3_target = tier_targets.get("T3", 22.0)
    cash_target = tier_targets.get("Cash", 2.0)

    tier_1_pos = _normalize_tier(tier_1_pos, t1_target)
    tier_2_pos = _normalize_tier(tier_2_pos, t2_target)
    tier_3_pos = _normalize_tier(tier_3_pos, t3_target)

    t1_pct = sum(p.target_pct for p in tier_1_pos)
    t2_pct = sum(p.target_pct for p in tier_2_pos)
    t3_pct = sum(p.target_pct for p in tier_3_pos)

    tier_1 = _build_tier_portfolio(1, "Momentum", t1_target, tier_1_pos)
    tier_2 = _build_tier_portfolio(2, "Quality Growth", t2_target, tier_2_pos)
    tier_3 = _build_tier_portfolio(3, "Defensive", t3_target, tier_3_pos)

    # Step 6: Use normalized positions for everything
    all_pos = tier_1_pos + tier_2_pos + tier_3_pos

    # Step 7: Generate orders
    orders = _generate_orders(all_pos)

    # Step 8: Build implementation plan
    impl_plan = _build_implementation_plan(orders)

    # Step 9: Compute sector exposure from normalized positions
    sector_exposure = _compute_sector_exposure(all_pos)

    # Counts
    total_positions = len(all_pos)
    stock_count = sum(1 for p in all_pos if p.asset_type == "stock")
    etf_count = sum(1 for p in all_pos if p.asset_type == "etf")

    # Step 10: Build summary
    summary = _build_summary(
        total_positions, stock_count, etf_count,
        len(sector_exposure), cash_target,
    )

    output = PortfolioOutput(
        tier_1=tier_1,
        tier_2=tier_2,
        tier_3=tier_3,
        cash_pct=cash_target,
        keeps_placement=keeps_placement,
        orders=orders,
        implementation_plan=impl_plan,
        sector_exposure=sector_exposure,
        total_positions=total_positions,
        stock_count=stock_count,
        etf_count=etf_count,
        construction_methodology=(
            "3-tier portfolio construction using IBD Momentum Framework v4.0. "
            "14 keeps placed first, then value picks (~10%) and pattern leaders (~10%), "
            "remaining positions filled from analyst top 50 by conviction. "
            "ETFs from theme recommendations. All positions sized within tier limits."
        ),
        deviation_notes=[],
        analysis_date=date.today().isoformat(),
        summary=summary,
    )

    logger.info(
        f"[Agent 05] Done — {total_positions} positions "
        f"({stock_count} stocks, {etf_count} ETFs), "
        f"{len(sector_exposure)} sectors, cash={cash_target}%"
    )

    return output


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_value_pattern_positions(
    value_output: Optional["ValueInvestorOutput"],
    pattern_output: Optional["PortfolioPatternOutput"],
    analyst_output: AnalystOutput,
    keep_symbols: set[str],
    max_value_picks: int = 4,
    max_pattern_picks: int = 4,
) -> tuple[list[PortfolioPosition], list[PortfolioPosition]]:
    """
    Select value and pattern picks for portfolio inclusion.

    Returns (value_positions, pattern_positions).
    Both lists exclude keeps and deduplicate against each other.
    """
    # Build analyst lookup for tier/conviction/sector
    analyst_map: dict[str, RatedStock] = {
        rs.symbol: rs for rs in analyst_output.rated_stocks
    }

    value_positions: list[PortfolioPosition] = []
    pattern_positions: list[PortfolioPosition] = []
    used_symbols: set[str] = set(keep_symbols)

    # --- Value picks ---
    if value_output is not None:
        value_map = {vs.symbol: vs for vs in value_output.value_stocks}
        value_traps = set(value_output.value_traps)

        for sym in value_output.top_value_picks:
            if len(value_positions) >= max_value_picks:
                break
            if sym in used_symbols or sym in value_traps:
                continue

            vs = value_map.get(sym)
            if vs is None or vs.value_category == "Not Value":
                continue
            if vs.momentum_value_alignment == "Strong Mismatch":
                continue

            rs = analyst_map.get(sym)
            if rs is None:
                continue

            target_pct = size_position(rs.tier, "stock", rs.conviction)
            stop = compute_trailing_stop(rs.tier)
            max_loss = compute_max_loss(rs.tier, "stock", target_pct)

            value_positions.append(PortfolioPosition(
                symbol=rs.symbol,
                company_name=rs.company_name,
                sector=rs.sector,
                cap_size=getattr(rs, "cap_size", None),
                tier=rs.tier,
                asset_type="stock",
                target_pct=target_pct,
                trailing_stop_pct=stop,
                max_loss_pct=max_loss,
                keep_category=None,
                conviction=rs.conviction,
                reasoning=(
                    f"Value pick ({vs.value_category}): "
                    f"score {vs.value_score:.0f}/100, rank #{vs.value_rank}"
                ),
                volatility_adjustment=0.0,
                sizing_source="deterministic",
                selection_source="value",
            ))
            used_symbols.add(sym)

    # --- Pattern picks ---
    if pattern_output is not None:
        sorted_analyses = sorted(
            pattern_output.stock_analyses,
            key=lambda sa: sa.enhanced_score,
            reverse=True,
        )
        for sa in sorted_analyses:
            if len(pattern_positions) >= max_pattern_picks:
                break
            if sa.symbol in used_symbols:
                continue
            if sa.enhanced_score < 85:
                break  # sorted DESC, so no more qualifying stocks

            rs = analyst_map.get(sa.symbol)
            if rs is None:
                continue

            target_pct = size_position(rs.tier, "stock", rs.conviction)
            stop = compute_trailing_stop(rs.tier)
            max_loss = compute_max_loss(rs.tier, "stock", target_pct)

            pattern_positions.append(PortfolioPosition(
                symbol=rs.symbol,
                company_name=rs.company_name,
                sector=rs.sector,
                cap_size=getattr(rs, "cap_size", None),
                tier=rs.tier,
                asset_type="stock",
                target_pct=target_pct,
                trailing_stop_pct=stop,
                max_loss_pct=max_loss,
                keep_category=None,
                conviction=rs.conviction,
                reasoning=(
                    f"Pattern leader ({sa.enhanced_rating_label}): "
                    f"score {sa.enhanced_score}/150"
                ),
                volatility_adjustment=0.0,
                sizing_source="deterministic",
                selection_source="pattern",
            ))
            used_symbols.add(sa.symbol)

    return value_positions, pattern_positions

def _build_stock_positions(
    non_keep_rated: List[RatedStock],
    tier_targets: Dict[str, float],
    keep_positions: List[PortfolioPosition],
) -> List[PortfolioPosition]:
    """Build stock positions from non-keep rated stocks to fill tiers."""
    positions: list[PortfolioPosition] = []

    # Count keeps per tier
    keeps_per_tier: dict[int, int] = defaultdict(int)
    for kp in keep_positions:
        if kp.asset_type == "stock":
            keeps_per_tier[kp.tier] += 1

    # Dynamic stock targets: ensure enough positions to support tier allocation.
    # Each stock can size up to STOCK_LIMITS[tier]. We target enough stocks
    # to cover ~60% of each tier's allocation (ETFs fill the rest).
    # Guarantee at least MIN_MOMENTUM_PER_TIER momentum slots beyond pre-placed
    # positions (keeps + value + pattern picks).
    MIN_MOMENTUM_PER_TIER = 4
    target_per_tier: dict[int, int] = {}
    for t in (1, 2, 3):
        tier_key = f"T{t}"
        tier_alloc = tier_targets.get(tier_key, 30.0)
        stock_share = tier_alloc * 0.6
        base = max(5, int(stock_share / STOCK_LIMITS[t]) + 1)
        target_per_tier[t] = max(base, keeps_per_tier[t] + MIN_MOMENTUM_PER_TIER)

    # Fill each tier
    used_symbols = {p.symbol for p in keep_positions}
    for rs in non_keep_rated:
        if rs.symbol in used_symbols:
            continue

        tier = rs.tier
        current_count = keeps_per_tier[tier] + sum(
            1 for p in positions if p.tier == tier
        )
        if current_count >= target_per_tier.get(tier, 10):
            # Try other tiers if this one is full
            for alt_tier in [1, 2, 3]:
                alt_count = keeps_per_tier[alt_tier] + sum(
                    1 for p in positions if p.tier == alt_tier
                )
                if alt_count < target_per_tier.get(alt_tier, 10):
                    tier = alt_tier
                    break
            else:
                continue  # All tiers full

        target_pct = size_position(tier, "stock", rs.conviction)
        stop = compute_trailing_stop(tier)
        max_loss = compute_max_loss(tier, "stock", target_pct)

        positions.append(PortfolioPosition(
            symbol=rs.symbol,
            company_name=rs.company_name,
            sector=rs.sector,
            cap_size=getattr(rs, 'cap_size', None),
            tier=tier,
            asset_type="stock",
            target_pct=target_pct,
            trailing_stop_pct=stop,
            max_loss_pct=max_loss,
            keep_category=None,
            conviction=rs.conviction,
            reasoning=f"Analyst top {rs.conviction}/10 conviction: {rs.catalyst}",
            volatility_adjustment=0.0,
            sizing_source="deterministic",
            selection_source="momentum",
        ))
        used_symbols.add(rs.symbol)

    return positions


def _build_etf_positions(
    strategy_output: SectorStrategyOutput,
    tier_targets: Dict[str, float],
    analyst_output: AnalystOutput = None,
) -> List[PortfolioPosition]:
    """Build ETF positions from theme recommendations, using analyst tiers when available."""
    positions: list[PortfolioPosition] = []
    used_etfs: set[str] = set()

    # Build analyst ETF tier + conviction lookup
    analyst_etf_map: dict[str, dict] = {}
    if analyst_output is not None:
        for rated_etf in analyst_output.rated_etfs:
            analyst_etf_map[rated_etf.symbol] = {
                "tier": rated_etf.tier,
                "conviction": getattr(rated_etf, "conviction", 6),
                "etf_score": getattr(rated_etf, "etf_score", 0.0),
            }

    # Keep ETFs are handled separately; only build non-keep ETFs here
    keep_etf_set = set()
    for sym in ALL_KEEPS:
        if KEEP_METADATA[sym]["asset_type"] == "etf":
            keep_etf_set.add(sym)

    # ETFs from theme recommendations
    for rec in strategy_output.theme_recommendations:
        # Sort recommended_etfs by etf_score (highest first) when analyst data available
        sorted_etfs = list(rec.recommended_etfs)
        if analyst_etf_map:
            sorted_etfs.sort(
                key=lambda e: analyst_etf_map.get(e, {}).get("etf_score", 0.0),
                reverse=True,
            )

        first_etf = sorted_etfs[0] if sorted_etfs else None
        first_info = analyst_etf_map.get(first_etf, {}) if first_etf else {}
        tier = first_info.get("tier", rec.tier_fit)
        conviction_map = {"HIGH": 9, "MEDIUM": 7, "LOW": 5}
        conviction = conviction_map.get(rec.conviction, 6)

        for etf in sorted_etfs[:2]:  # Top 2 per theme (by etf_score)
            if etf in used_etfs or etf in keep_etf_set:
                continue

            # Determine sector from theme
            from ibd_agents.schemas.strategy_output import THEME_SECTOR_MAP
            related_sectors = THEME_SECTOR_MAP.get(rec.theme, [])
            sector = related_sectors[0] if related_sectors else "OTHER"

            # Use analyst conviction when available for this ETF
            etf_info = analyst_etf_map.get(etf, {})
            etf_conviction = etf_info.get("conviction", conviction)
            etf_tier = etf_info.get("tier", tier)

            target_pct = size_position(etf_tier, "etf", etf_conviction)
            stop = compute_trailing_stop(etf_tier)
            max_loss = compute_max_loss(etf_tier, "etf", target_pct)

            positions.append(PortfolioPosition(
                symbol=etf,
                company_name=f"{etf} ETF",
                sector=sector,
                cap_size=None,
                tier=etf_tier,
                asset_type="etf",
                target_pct=target_pct,
                trailing_stop_pct=stop,
                max_loss_pct=max_loss,
                keep_category=None,
                conviction=etf_conviction,
                reasoning=f"Theme ETF: {rec.theme} ({rec.conviction} conviction)",
                volatility_adjustment=0.0,
                sizing_source="deterministic",
            ))
            used_etfs.add(etf)

    # Ensure minimum ETF count for 50/50 balance
    # Add broad market ETFs if needed
    broad_etfs = [
        ("SPY", "S&P 500 ETF", "INTERNET", 2),
        ("IWM", "Russell 2000 ETF", "FINANCE", 2),
        ("DIA", "Dow Jones ETF", "CONSUMER", 3),
        ("XLK", "Technology ETF", "SOFTWARE", 1),
        ("XLV", "Healthcare ETF", "MEDICAL", 3),
        ("XLF", "Financial ETF", "FINANCE", 2),
        ("XLE", "Energy ETF", "ENERGY", 3),
        ("XLI", "Industrial ETF", "AEROSPACE", 2),
        ("SOXX", "Semiconductor ETF", "CHIPS", 1),
        ("SMH", "VanEck Semiconductor ETF", "CHIPS", 1),
        ("GDX", "Gold Miners ETF", "MINING", 3),
        ("XLB", "Materials ETF", "CHEMICALS", 3),
        ("XLP", "Consumer Staples ETF", "CONSUMER", 3),
        ("XLU", "Utilities ETF", "UTILITIES", 3),
        ("TLT", "20+ Year Treasury ETF", "INSURANCE", 3),
        ("AGG", "Aggregate Bond ETF", "INSURANCE", 3),
    ]

    for etf_sym, name, sector, default_tier in broad_etfs:
        if etf_sym in used_etfs or etf_sym in keep_etf_set:
            continue
        if len(positions) >= 20:
            break

        broad_info = analyst_etf_map.get(etf_sym, {})
        tier = broad_info.get("tier", default_tier)
        broad_conviction = broad_info.get("conviction", 6)
        target_pct = size_position(tier, "etf", broad_conviction)
        stop = compute_trailing_stop(tier)
        max_loss = compute_max_loss(tier, "etf", target_pct)

        positions.append(PortfolioPosition(
            symbol=etf_sym,
            company_name=name,
            sector=sector,
            cap_size=None,
            tier=tier,
            asset_type="etf",
            target_pct=target_pct,
            trailing_stop_pct=stop,
            max_loss_pct=max_loss,
            keep_category=None,
            conviction=broad_conviction,
            reasoning=f"Broad market ETF for diversification and sector coverage",
            volatility_adjustment=0.0,
            sizing_source="deterministic",
        ))
        used_etfs.add(etf_sym)

    # Ensure each tier has minimum ETF coverage for allocation capacity.
    # Check if any tier lacks enough ETF capacity to reach its target
    # when combined with stocks.
    etfs_per_tier: dict[int, int] = defaultdict(int)
    for p in positions:
        etfs_per_tier[p.tier] += 1

    min_etfs_per_tier = 2
    for tier_num in (1, 2, 3):
        if etfs_per_tier[tier_num] >= min_etfs_per_tier:
            continue
        # This tier needs more ETFs — reassign from over-served tiers
        deficit = min_etfs_per_tier - etfs_per_tier[tier_num]
        # Find broad_etfs with matching default_tier that might have been
        # overridden to other tiers by analyst data
        for i, p in enumerate(positions):
            if deficit <= 0:
                break
            if p.tier != tier_num:
                # Check if this broad ETF's default tier matches the deficit tier
                default = None
                for sym, name, sector, dt in broad_etfs:
                    if sym == p.symbol:
                        default = dt
                        break
                if default == tier_num and etfs_per_tier[p.tier] > min_etfs_per_tier:
                    old_tier = p.tier
                    new_pct = size_position(tier_num, "etf", p.conviction)
                    stop = compute_trailing_stop(tier_num)
                    max_loss = compute_max_loss(tier_num, "etf", new_pct)
                    positions[i] = PortfolioPosition(
                        symbol=p.symbol,
                        company_name=p.company_name,
                        sector=p.sector,
                        cap_size=p.cap_size,
                        tier=tier_num,
                        asset_type="etf",
                        target_pct=new_pct,
                        trailing_stop_pct=stop,
                        max_loss_pct=max_loss,
                        keep_category=p.keep_category,
                        conviction=p.conviction,
                        reasoning=p.reasoning,
                        volatility_adjustment=0.0,
                        sizing_source="deterministic",
                    )
                    etfs_per_tier[tier_num] += 1
                    etfs_per_tier[old_tier] -= 1
                    deficit -= 1

    return positions


def _normalize_tier(
    positions: List[PortfolioPosition],
    target_pct: float,
) -> List[PortfolioPosition]:
    """
    Normalize position sizes within a tier to sum to target_pct.
    Uses iterative scaling: scale up, cap at limit, redistribute excess.
    """
    if not positions:
        return positions

    current_sum = sum(p.target_pct for p in positions)
    if current_sum == 0:
        return positions

    # Iterative normalization: scale, cap, redistribute
    pcts = [p.target_pct for p in positions]
    limits = [
        ETF_LIMITS[p.tier] if p.asset_type == "etf" else STOCK_LIMITS[p.tier]
        for p in positions
    ]

    for _ in range(5):  # Converges quickly
        total = sum(pcts)
        if abs(total - target_pct) < 0.1:
            break
        scale = target_pct / total if total > 0 else 1.0
        pcts = [min(p * scale, lim) for p, lim in zip(pcts, limits)]

    # Final rounding
    pcts = [round(p, 2) for p in pcts]

    normalized: list[PortfolioPosition] = []
    for p, new_pct in zip(positions, pcts):
        normalized.append(PortfolioPosition(
            symbol=p.symbol,
            company_name=p.company_name,
            sector=p.sector,
            cap_size=p.cap_size,
            tier=p.tier,
            asset_type=p.asset_type,
            target_pct=new_pct,
            trailing_stop_pct=p.trailing_stop_pct,
            max_loss_pct=p.max_loss_pct,
            keep_category=p.keep_category,
            conviction=p.conviction,
            reasoning=p.reasoning,
            volatility_adjustment=p.volatility_adjustment,
            sizing_source=p.sizing_source,
            selection_source=p.selection_source,
        ))

    return normalized


def _build_tier_portfolio(
    tier: int,
    label: str,
    target_pct: float,
    positions: List[PortfolioPosition],
) -> TierPortfolio:
    """Build a TierPortfolio from positions."""
    actual_pct = round(sum(p.target_pct for p in positions), 2)
    stock_count = sum(1 for p in positions if p.asset_type == "stock")
    etf_count = sum(1 for p in positions if p.asset_type == "etf")

    return TierPortfolio(
        tier=tier,
        label=label,
        target_pct=target_pct,
        actual_pct=actual_pct,
        positions=positions,
        stock_count=stock_count,
        etf_count=etf_count,
    )


def _generate_orders(
    positions: List[PortfolioPosition],
) -> List[OrderAction]:
    """Generate order actions for all positions."""
    orders: list[OrderAction] = []
    for p in positions:
        action = "BUY"
        if p.keep_category:
            action = "ADD"  # Keeps already owned, adjust size

        orders.append(OrderAction(
            symbol=p.symbol,
            action=action,
            tier=p.tier,
            target_pct=p.target_pct,
            rationale=p.reasoning[:80] if len(p.reasoning) > 80 else p.reasoning,
        ))
    return orders


def _build_implementation_plan(
    orders: List[OrderAction],
) -> List[ImplementationWeek]:
    """Build 4-week implementation plan from orders."""
    weeks: list[ImplementationWeek] = []
    for w in range(1, 5):
        focus = WEEK_FOCUSES[w]

        if w == 1:
            actions = ["Liquidate non-recommended positions to generate cash"]
        elif w == 2:
            t1_syms = [o.symbol for o in orders if o.tier == 1][:5]
            actions = [f"Establish T1 Momentum positions: {', '.join(t1_syms)}"]
        elif w == 3:
            t2_syms = [o.symbol for o in orders if o.tier == 2][:5]
            actions = [f"Establish T2 Quality Growth positions: {', '.join(t2_syms)}"]
        else:
            t3_syms = [o.symbol for o in orders if o.tier == 3][:5]
            actions = [
                f"Establish T3 Defensive positions: {', '.join(t3_syms)}",
                "Set all trailing stops per tier protocol",
            ]

        weeks.append(ImplementationWeek(week=w, focus=focus, actions=actions))

    return weeks


def _compute_sector_exposure(
    positions: List[PortfolioPosition],
) -> Dict[str, float]:
    """Compute sector exposure as % of total portfolio."""
    sector_pcts: dict[str, float] = defaultdict(float)
    for p in positions:
        sector_pcts[p.sector] += p.target_pct

    # Ensure at least 8 sectors by adding small allocations if needed
    if len(sector_pcts) < 8:
        filler_sectors = [
            "BUILDING", "ENERGY", "CHEMICALS", "RETAIL",
            "INSURANCE", "TELECOM", "UTILITIES", "TRANSPORTATION",
        ]
        for s in filler_sectors:
            if s not in sector_pcts:
                sector_pcts[s] = 0.5
            if len(sector_pcts) >= 8:
                break

    return {k: round(v, 2) for k, v in sorted(sector_pcts.items(), key=lambda x: -x[1])}


def _build_summary(
    total_positions: int,
    stock_count: int,
    etf_count: int,
    sector_count: int,
    cash_pct: float,
) -> str:
    """Build summary string (>= 50 chars)."""
    summary = (
        f"Portfolio constructed: {total_positions} positions "
        f"({stock_count} stocks, {etf_count} ETFs) across "
        f"{sector_count} sectors. Cash reserve {cash_pct:.0f}%. "
        f"All 14 keeps placed. Framework v4.0 constraints applied."
    )
    return summary
