"""
PDF Summary Report Generator
IBD Momentum Investment Framework v4.0

Generates a comprehensive PDF report summarizing analysis from all 12 agents.
Uses ReportLab for PDF generation.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ibd_agents.schemas.analyst_output import AnalystOutput
from ibd_agents.schemas.historical_output import HistoricalAnalysisOutput
from ibd_agents.schemas.pattern_output import PortfolioPatternOutput
from ibd_agents.schemas.portfolio_output import PortfolioOutput
from ibd_agents.schemas.reconciliation_output import ReconciliationOutput
from ibd_agents.schemas.research_output import ResearchOutput
from ibd_agents.schemas.returns_projection_output import ReturnsProjectionOutput
from ibd_agents.schemas.risk_output import RiskAssessment
from ibd_agents.schemas.rotation_output import RotationDetectionOutput
from ibd_agents.schemas.strategy_output import SectorStrategyOutput
from ibd_agents.schemas.synthesis_output import SynthesisOutput
from ibd_agents.schemas.value_investor_output import ValueInvestorOutput

try:
    from ibd_agents.schemas.educator_output import EducatorOutput
except ImportError:
    EducatorOutput = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

GREEN = colors.HexColor("#00B050")
LIGHT_GREEN = colors.HexColor("#92D050")
YELLOW = colors.HexColor("#FFC000")
ORANGE = colors.HexColor("#FFA500")
RED = colors.HexColor("#FF0000")
BLUE = colors.HexColor("#4472C4")
DARK_GREEN = colors.HexColor("#006400")
GOLD = colors.HexColor("#FFD700")
GRAY_LIGHT = colors.HexColor("#F2F2F2")
GRAY_MED = colors.HexColor("#D9D9D9")
GRAY_DARK = colors.HexColor("#404040")
HEADER_BG = colors.HexColor("#1F4E79")
WHITE = colors.white
BLACK = colors.black

STATUS_COLORS = {
    "APPROVED": GREEN,
    "CONDITIONAL": ORANGE,
    "REJECTED": RED,
}

TIER_COLORS = {
    1: DARK_GREEN,
    2: GOLD,
    3: BLUE,
}

# ---------------------------------------------------------------------------
# Custom Styles
# ---------------------------------------------------------------------------

_BASE_STYLES = getSampleStyleSheet()


def _styles():
    """Create custom paragraph styles."""
    styles = {}

    styles["Title"] = ParagraphStyle(
        "CustomTitle",
        parent=_BASE_STYLES["Title"],
        fontSize=28,
        leading=34,
        spaceAfter=6,
        textColor=HEADER_BG,
        alignment=TA_CENTER,
    )
    styles["Subtitle"] = ParagraphStyle(
        "CustomSubtitle",
        parent=_BASE_STYLES["Normal"],
        fontSize=16,
        leading=20,
        spaceAfter=12,
        textColor=GRAY_DARK,
        alignment=TA_CENTER,
    )
    styles["H1"] = ParagraphStyle(
        "CustomH1",
        parent=_BASE_STYLES["Heading1"],
        fontSize=18,
        leading=22,
        spaceBefore=16,
        spaceAfter=8,
        textColor=HEADER_BG,
        borderWidth=1,
        borderColor=HEADER_BG,
        borderPadding=4,
    )
    styles["H2"] = ParagraphStyle(
        "CustomH2",
        parent=_BASE_STYLES["Heading2"],
        fontSize=14,
        leading=17,
        spaceBefore=10,
        spaceAfter=6,
        textColor=GRAY_DARK,
    )
    styles["Body"] = ParagraphStyle(
        "CustomBody",
        parent=_BASE_STYLES["BodyText"],
        fontSize=10,
        leading=14,
        spaceAfter=8,
    )
    styles["Small"] = ParagraphStyle(
        "CustomSmall",
        parent=_BASE_STYLES["Normal"],
        fontSize=8,
        leading=10,
        textColor=GRAY_DARK,
    )
    styles["Center"] = ParagraphStyle(
        "CustomCenter",
        parent=_BASE_STYLES["Normal"],
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
    )
    styles["Disclaimer"] = ParagraphStyle(
        "CustomDisclaimer",
        parent=_BASE_STYLES["Normal"],
        fontSize=9,
        leading=12,
        textColor=GRAY_DARK,
        spaceBefore=4,
        spaceAfter=4,
    )
    styles["Reasoning"] = ParagraphStyle(
        "CustomReasoning",
        parent=_BASE_STYLES["Normal"],
        fontSize=8,
        leading=11,
        textColor=GRAY_DARK,
        leftIndent=12,
        spaceAfter=2,
    )
    styles["CellWrap"] = ParagraphStyle(
        "CustomCellWrap",
        parent=_BASE_STYLES["Normal"],
        fontSize=8,
        leading=10,
        spaceBefore=1,
        spaceAfter=1,
    )
    styles["CellBold"] = ParagraphStyle(
        "CustomCellBold",
        parent=_BASE_STYLES["Normal"],
        fontSize=8,
        leading=10,
        spaceBefore=1,
        spaceAfter=1,
        fontName="Helvetica-Bold",
    )
    return styles


# ---------------------------------------------------------------------------
# Table Helpers
# ---------------------------------------------------------------------------

HEADER_STYLE = [
    ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
    ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, 0), 10),
    ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
    ("TOPPADDING", (0, 0), (-1, 0), 6),
    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
    ("FONTSIZE", (0, 1), (-1, -1), 9),
    ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
    ("TOPPADDING", (0, 1), (-1, -1), 4),
    ("GRID", (0, 0), (-1, -1), 0.5, GRAY_MED),
]


def _zebra_rows(n_rows: int):
    """Return alternating row background commands starting from row 1."""
    cmds = []
    for i in range(1, n_rows):
        if i % 2 == 0:
            cmds.append(("BACKGROUND", (0, i), (-1, i), GRAY_LIGHT))
    return cmds


def _make_table(data, col_widths=None, extra_style=None):
    """Create a styled table."""
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style_cmds = list(HEADER_STYLE) + _zebra_rows(len(data))
    if extra_style:
        style_cmds.extend(extra_style)
    t.setStyle(TableStyle(style_cmds))
    return t


# ---------------------------------------------------------------------------
# Section Builders
# ---------------------------------------------------------------------------


def _title_page(synthesis: SynthesisOutput, s) -> List[Any]:
    """Title page."""
    elements = []
    elements.append(Spacer(1, 2.0 * inch))
    elements.append(Paragraph("IBD Momentum Investment Framework v4.0", s["Title"]))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(
        Paragraph("Multi-Agent Portfolio Analysis Report", s["Subtitle"])
    )
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(
        Paragraph(
            f"Analysis Date: {synthesis.analysis_date}",
            s["Center"],
        )
    )
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(
        Paragraph(
            f"Synthesis Source: {synthesis.synthesis_source.upper()}",
            s["Center"],
        )
    )
    elements.append(Spacer(1, 1.5 * inch))
    elements.append(
        Paragraph(
            "This report discovers, scores, and organizes investment "
            "opportunities. It never makes buy or sell recommendations.",
            s["Disclaimer"],
        )
    )
    return elements


def _executive_summary(synthesis: SynthesisOutput, s) -> List[Any]:
    """Executive summary from Agent 10."""
    elements = []
    elements.append(Paragraph("Executive Summary", s["H1"]))

    elements.append(Paragraph("Investment Thesis", s["H2"]))
    elements.append(Paragraph(synthesis.investment_thesis, s["Body"]))

    elements.append(Paragraph("Portfolio Narrative", s["H2"]))
    elements.append(Paragraph(synthesis.portfolio_narrative, s["Body"]))

    elements.append(Paragraph("Market Context", s["H2"]))
    elements.append(Paragraph(synthesis.market_context, s["Body"]))

    elements.append(Paragraph("Risk / Reward Assessment", s["H2"]))
    elements.append(Paragraph(synthesis.risk_reward_assessment, s["Body"]))
    return elements


def _key_numbers(synthesis: SynthesisOutput, s) -> List[Any]:
    """Key numbers dashboard from Agent 10."""
    elements = []
    elements.append(Paragraph("Key Numbers Dashboard", s["H1"]))

    kn = synthesis.key_numbers
    data = [
        ["Category", "Metric", "Value"],
        # Research
        ["Research", "Total Stocks Scanned", f"{kn.total_stocks_scanned:,}"],
        ["Research", "Stocks in Universe", f"{kn.stocks_in_universe:,}"],
        ["Research", "ETFs in Universe", f"{kn.etfs_in_universe:,}"],
        # Analyst
        ["Analyst", "Stocks Rated", f"{kn.stocks_rated:,}"],
        ["Analyst", "ETFs Rated", f"{kn.etfs_rated:,}"],
        ["Analyst", "Tier 1 (Momentum)", str(kn.tier_1_count)],
        ["Analyst", "Tier 2 (Quality Growth)", str(kn.tier_2_count)],
        ["Analyst", "Tier 3 (Defensive)", str(kn.tier_3_count)],
        # Rotation
        ["Rotation", "Verdict", kn.rotation_verdict],
        ["Rotation", "Confidence", f"{kn.rotation_confidence}%"],
        ["Rotation", "Market Regime", kn.market_regime],
        # Portfolio
        ["Portfolio", "Total Positions", str(kn.portfolio_positions)],
        ["Portfolio", "Stocks", str(kn.stock_count)],
        ["Portfolio", "ETFs", str(kn.etf_count)],
        ["Portfolio", "Cash", f"{kn.cash_pct:.1f}%"],
        # Risk
        ["Risk", "Overall Status", kn.risk_status],
        ["Risk", "Sleep Well Score", f"{kn.sleep_well_score}/10"],
        ["Risk", "Warnings", str(kn.risk_warnings_count)],
        # Returns
        ["Returns", "Bull Case 12m", f"{kn.bull_case_return_12m:.1f}%"],
        ["Returns", "Base Case 12m", f"{kn.base_case_return_12m:.1f}%"],
        ["Returns", "Bear Case 12m", f"{kn.bear_case_return_12m:.1f}%"],
        # Reconciler
        ["Reconciler", "Portfolio Turnover", f"{kn.turnover_pct:.0f}%"],
        ["Reconciler", "Actions", str(kn.actions_count)],
    ]

    widths = [1.5 * inch, 2.5 * inch, 2.5 * inch]
    elements.append(_make_table(data, widths))
    return elements


def _rotation_section(
    rotation: RotationDetectionOutput, s
) -> List[Any]:
    """Market & rotation analysis from Agent 03."""
    elements = []
    elements.append(Paragraph("Market &amp; Rotation Analysis", s["H1"]))

    # Verdict summary
    regime = rotation.market_regime
    regime_str = f"{regime.regime.upper()}"
    elements.append(
        Paragraph(
            f"<b>Verdict:</b> {rotation.verdict.value.upper()} &nbsp; | &nbsp; "
            f"<b>Confidence:</b> {rotation.confidence}% &nbsp; | &nbsp; "
            f"<b>Type:</b> {rotation.rotation_type.value} &nbsp; | &nbsp; "
            f"<b>Regime:</b> {regime_str}",
            s["Body"],
        )
    )

    # Signals
    elements.append(Paragraph("Rotation Signals", s["H2"]))
    sig = rotation.signals
    signal_readings = [
        ("RS Divergence", sig.rs_divergence),
        ("Leadership Change", sig.leadership_change),
        ("Breadth Shift", sig.breadth_shift),
        ("Elite Concentration", sig.elite_concentration_shift),
        ("IBD Keep Migration", sig.ibd_keep_migration),
    ]
    sig_data = [["Signal", "Status", "Evidence"]]
    for name, reading in signal_readings:
        status = "Active" if reading.triggered else "Inactive"
        evidence = reading.evidence[:60] if reading.evidence else ""
        sig_data.append([name, status, evidence])
    active_style = []
    for i in range(1, 6):
        if sig_data[i][1] == "Active":
            active_style.append(("TEXTCOLOR", (1, i), (1, i), GREEN))
        else:
            active_style.append(("TEXTCOLOR", (1, i), (1, i), RED))
    elements.append(
        _make_table(
            sig_data, [1.8 * inch, 1.0 * inch, 3.7 * inch], active_style
        )
    )
    elements.append(
        Paragraph(
            f"<b>Active Signals:</b> {sig.signals_active}/5",
            s["Body"],
        )
    )

    # Sector flows
    if rotation.source_sectors or rotation.destination_sectors:
        elements.append(Paragraph("Sector Flows", s["H2"]))
        flow_data = [["Direction", "Sector", "Avg RS", "Magnitude"]]
        for sf in rotation.source_sectors[:5]:
            flow_data.append(
                ["Outflow", sf.sector, f"{sf.avg_rs:.0f}", sf.magnitude]
            )
        for sf in rotation.destination_sectors[:5]:
            flow_data.append(
                ["Inflow", sf.sector, f"{sf.avg_rs:.0f}", sf.magnitude]
            )
        elements.append(
            _make_table(
                flow_data, [1.5 * inch, 2.5 * inch, 1.5 * inch, 1.5 * inch]
            )
        )

    # Rotation narrative (if LLM provided one)
    if rotation.rotation_narrative:
        elements.append(Paragraph("Historical Context", s["H2"]))
        elements.append(Paragraph(rotation.rotation_narrative, s["Body"]))

    return elements


def _position_table(positions, s) -> List[Any]:
    """Render a table of positions with reasoning for a single tier."""
    elements = []
    if not positions:
        elements.append(Paragraph("No positions in this tier.", s["Body"]))
        return elements

    sorted_pos = sorted(positions, key=lambda p: (-p.target_pct,))
    pos_data = [
        ["Symbol", "Company", "Type", "Sector", "Wt%", "Stop%", "MaxLoss%", "Conv", "Source"],
    ]
    col_widths = [
        0.6 * inch, 1.5 * inch, 0.5 * inch, 1.2 * inch,
        0.55 * inch, 0.55 * inch, 0.7 * inch, 0.45 * inch, 0.85 * inch,
    ]

    # Source label and color mapping
    _SOURCE_LABELS = {
        "keep": "Keep",
        "value": "Value",
        "pattern": "Pattern",
        "momentum": "Momentum",
    }
    _SOURCE_COLORS = {
        "keep": DARK_GREEN,
        "value": BLUE,
        "pattern": colors.HexColor("#8B008B"),  # dark magenta
        "momentum": ORANGE,
    }

    for p in sorted_pos:
        src = _SOURCE_LABELS.get(p.selection_source or "", p.selection_source or "")
        pos_data.append([
            p.symbol,
            p.company_name[:22],
            p.asset_type[:3].upper(),
            p.sector[:18],
            f"{p.target_pct:.2f}",
            f"{p.trailing_stop_pct:.0f}",
            f"{p.max_loss_pct:.2f}",
            str(p.conviction),
            src,
        ])

    # Color-code conviction and source columns
    conv_style = []
    for i in range(1, len(pos_data)):
        conv_val = int(pos_data[i][7])
        if conv_val >= 8:
            conv_style.append(("TEXTCOLOR", (7, i), (7, i), GREEN))
        elif conv_val >= 5:
            conv_style.append(("TEXTCOLOR", (7, i), (7, i), BLUE))
        else:
            conv_style.append(("TEXTCOLOR", (7, i), (7, i), ORANGE))
        # Color-code source column
        src_key = sorted_pos[i - 1].selection_source
        if src_key in _SOURCE_COLORS:
            conv_style.append(("TEXTCOLOR", (8, i), (8, i), _SOURCE_COLORS[src_key]))
    # Right-align numeric columns
    conv_style.append(("ALIGN", (4, 0), (7, -1), "RIGHT"))

    elements.append(_make_table(pos_data, col_widths, conv_style))

    # Reasoning table — two columns: Symbol | Reasoning (word-wrapped)
    elements.append(Spacer(1, 0.15 * inch))
    elements.append(Paragraph("<b>Position Reasoning:</b>", s["Small"]))
    reason_data = [[
        Paragraph("<b>Symbol</b>", s["CellBold"]),
        Paragraph("<b>Reasoning</b>", s["CellBold"]),
    ]]
    for p in sorted_pos:
        vol_note = ""
        if p.volatility_adjustment and abs(p.volatility_adjustment) > 0.01:
            direction = "+" if p.volatility_adjustment > 0 else ""
            vol_note = (
                f' <font color="#4472C4">[Vol adj: {direction}'
                f'{p.volatility_adjustment:.2f}%, source: {p.sizing_source or "det"}]</font>'
            )
        reason_data.append([
            Paragraph(f"<b>{p.symbol}</b>", s["CellWrap"]),
            Paragraph(f"{p.reasoning}{vol_note}", s["CellWrap"]),
        ])
    reason_widths = [0.8 * inch, 5.7 * inch]
    reason_style = list(HEADER_STYLE) + _zebra_rows(len(reason_data))
    reason_style.append(("VALIGN", (0, 0), (-1, -1), "TOP"))
    reason_tbl = Table(reason_data, colWidths=reason_widths, repeatRows=1)
    reason_tbl.setStyle(TableStyle(reason_style))
    elements.append(reason_tbl)
    return elements


def _portfolio_section(portfolio: PortfolioOutput, s) -> List[Any]:
    """Portfolio construction from Agent 05."""
    elements = []
    elements.append(Paragraph("Portfolio Construction", s["H1"]))

    # Tier allocation summary
    elements.append(Paragraph("Tier Allocation", s["H2"]))
    tier_data = [
        ["Tier", "Label", "Target %", "Actual %", "Stocks", "ETFs"],
        [
            "Tier 1",
            "Momentum",
            f"{portfolio.tier_1.target_pct:.1f}%",
            f"{portfolio.tier_1.actual_pct:.1f}%",
            str(portfolio.tier_1.stock_count),
            str(portfolio.tier_1.etf_count),
        ],
        [
            "Tier 2",
            "Quality Growth",
            f"{portfolio.tier_2.target_pct:.1f}%",
            f"{portfolio.tier_2.actual_pct:.1f}%",
            str(portfolio.tier_2.stock_count),
            str(portfolio.tier_2.etf_count),
        ],
        [
            "Tier 3",
            "Defensive",
            f"{portfolio.tier_3.target_pct:.1f}%",
            f"{portfolio.tier_3.actual_pct:.1f}%",
            str(portfolio.tier_3.stock_count),
            str(portfolio.tier_3.etf_count),
        ],
        [
            "Cash",
            "",
            f"{portfolio.cash_pct:.1f}%",
            f"{portfolio.cash_pct:.1f}%",
            "",
            "",
        ],
    ]
    total_actual = (
        portfolio.tier_1.actual_pct
        + portfolio.tier_2.actual_pct
        + portfolio.tier_3.actual_pct
        + portfolio.cash_pct
    )
    tier_data.append(
        ["Total", "", "", f"{total_actual:.1f}%",
         str(portfolio.stock_count), str(portfolio.etf_count)]
    )
    elements.append(
        _make_table(
            tier_data,
            [1.0 * inch, 1.5 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch],
        )
    )

    # Sector exposure (top 10)
    elements.append(Paragraph("Sector Exposure (Top 10)", s["H2"]))
    sorted_sectors = sorted(
        portfolio.sector_exposure.items(), key=lambda x: -x[1]
    )[:10]
    sector_data = [["Sector", "Allocation %"]]
    for sec, pct in sorted_sectors:
        sector_data.append([sec, f"{pct:.1f}%"])
    elements.append(
        _make_table(sector_data, [4 * inch, 2.5 * inch])
    )

    # Keeps placement
    kp = portfolio.keeps_placement
    elements.append(Paragraph("Keeps Placement", s["H2"]))
    elements.append(
        Paragraph(
            f"<b>Total Keeps:</b> {kp.total_keeps} &nbsp; | &nbsp; "
            f"<b>Fundamental:</b> {len(kp.fundamental_keeps)} &nbsp; | &nbsp; "
            f"<b>IBD:</b> {len(kp.ibd_keeps)} &nbsp; | &nbsp; "
            f"<b>Allocation:</b> {kp.keeps_pct:.1f}%",
            s["Body"],
        )
    )
    elements.append(PageBreak())

    # --- Full Position Details per Tier ---
    tier_info = [
        (1, "Tier 1 — Momentum Positions", portfolio.tier_1),
        (2, "Tier 2 — Quality Growth Positions", portfolio.tier_2),
        (3, "Tier 3 — Defensive Positions", portfolio.tier_3),
    ]

    for tier_num, title, tier_obj in tier_info:
        elements.append(Paragraph(title, s["H1"]))
        elements.append(
            Paragraph(
                f"<b>Positions:</b> {len(tier_obj.positions)} &nbsp; | &nbsp; "
                f"<b>Stocks:</b> {tier_obj.stock_count} &nbsp; | &nbsp; "
                f"<b>ETFs:</b> {tier_obj.etf_count} &nbsp; | &nbsp; "
                f"<b>Actual:</b> {tier_obj.actual_pct:.1f}% of portfolio",
                s["Body"],
            )
        )
        elements.extend(_position_table(tier_obj.positions, s))
        elements.append(PageBreak())

    return elements


def _top_picks_section(
    analyst: AnalystOutput,
    value: ValueInvestorOutput,
    pattern: PortfolioPatternOutput,
    s,
) -> List[Any]:
    """Top picks from Agents 02, 11, 12."""
    elements = []
    elements.append(Paragraph("Top-Rated Securities", s["H1"]))

    # Top 10 Analyst (sorted by conviction)
    elements.append(Paragraph("Top 10 by Analyst Rating", s["H2"]))
    top_analyst = sorted(
        analyst.rated_stocks,
        key=lambda x: (-x.conviction, -x.composite_rating),
    )[:10]
    a_data = [["#", "Symbol", "Company", "Tier", "Comp", "RS", "EPS", "Conv"]]
    for i, rs in enumerate(top_analyst, 1):
        a_data.append([
            str(i), rs.symbol, rs.company_name[:25],
            str(rs.tier), str(rs.composite_rating),
            str(rs.rs_rating), str(rs.eps_rating),
            str(rs.conviction),
        ])
    elements.append(
        _make_table(
            a_data,
            [0.4 * inch, 0.7 * inch, 2.0 * inch, 0.5 * inch,
             0.6 * inch, 0.6 * inch, 0.6 * inch, 0.6 * inch],
        )
    )

    # Top Value Opportunities (Agent 11)
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph("Top Value Opportunities", s["H2"]))
    if value.top_value_picks:
        v_data = [["#", "Symbol", "Company", "Score", "Category", "Moat"]]
        for i, sym in enumerate(value.top_value_picks[:10], 1):
            stock = next(
                (vs for vs in value.value_stocks if vs.symbol == sym), None
            )
            if stock:
                v_data.append([
                    str(i), stock.symbol, stock.company_name[:25],
                    f"{stock.value_score:.1f}",
                    stock.value_category,
                    stock.economic_moat or "N/A",
                ])
        if len(v_data) > 1:
            elements.append(
                _make_table(
                    v_data,
                    [0.4 * inch, 0.7 * inch, 2.0 * inch,
                     0.7 * inch, 1.2 * inch, 1.5 * inch],
                )
            )
        else:
            elements.append(
                Paragraph("No value picks identified in this analysis.", s["Body"])
            )
    else:
        elements.append(
            Paragraph("No value picks identified in this analysis.", s["Body"])
        )

    # Top Pattern Leaders (Agent 12)
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph("Top Pattern Leaders", s["H2"]))
    pattern_sorted = sorted(
        pattern.stock_analyses,
        key=lambda x: x.enhanced_score,
        reverse=True,
    )[:10]
    p_data = [["#", "Symbol", "Company", "Score", "Rating", "Pattern"]]
    for i, sa in enumerate(pattern_sorted, 1):
        dominant = (
            sa.pattern_score.dominant_pattern if sa.pattern_score else "N/A"
        )
        p_data.append([
            str(i), sa.symbol, sa.company_name[:25],
            f"{sa.enhanced_score}/150",
            sa.enhanced_rating_label,
            dominant,
        ])
    elements.append(
        _make_table(
            p_data,
            [0.4 * inch, 0.7 * inch, 2.0 * inch,
             0.8 * inch, 1.0 * inch, 1.6 * inch],
        )
    )

    return elements


def _risk_section(risk: RiskAssessment, s) -> List[Any]:
    """Risk assessment from Agent 06."""
    elements = []
    elements.append(Paragraph("Risk Assessment", s["H1"]))

    # Overall status
    status_color = STATUS_COLORS.get(risk.overall_status, BLACK)
    hex_str = status_color.hexval() if hasattr(status_color, "hexval") else "#000000"
    elements.append(
        Paragraph(
            f'Overall Status: <font color="{hex_str}">'
            f"<b>{risk.overall_status}</b></font>",
            s["H2"],
        )
    )

    # Sleep Well Scores
    elements.append(Paragraph("Sleep Well Scores", s["H2"]))
    sw = risk.sleep_well_scores
    sw_data = [
        ["Metric", "Score"],
        ["Tier 1 (Momentum)", f"{sw.tier_1_score}/10"],
        ["Tier 2 (Quality Growth)", f"{sw.tier_2_score}/10"],
        ["Tier 3 (Defensive)", f"{sw.tier_3_score}/10"],
        ["Overall", f"{sw.overall_score}/10"],
    ]
    elements.append(_make_table(sw_data, [4 * inch, 2.5 * inch]))

    # Vetoes
    if risk.vetoes:
        elements.append(Paragraph("Vetoes (Action Required)", s["H2"]))
        for v in risk.vetoes:
            elements.append(
                Paragraph(
                    f'<font color="#FF0000"><b>{v.check_name}:</b> '
                    f"{v.reason}</font>",
                    s["Body"],
                )
            )
            elements.append(
                Paragraph(f"Required Fix: {v.required_fix}", s["Small"])
            )

    # Warnings
    if risk.warnings:
        elements.append(Paragraph("Warnings", s["H2"]))
        for w in risk.warnings:
            sev_colors = {"HIGH": "#FF0000", "MEDIUM": "#FFA500", "LOW": "#FFC000"}
            sev_hex = sev_colors.get(w.severity, "#000000")
            elements.append(
                Paragraph(
                    f'<font color="{sev_hex}">[{w.severity}]</font> '
                    f"<b>{w.check_name}:</b> {w.description}",
                    s["Body"],
                )
            )

    # Stress tests
    elements.append(Paragraph("Stress Test Results", s["H2"]))
    st_data = [["Scenario", "Impact", "Drawdown"]]
    for sc in risk.stress_test_results.scenarios:
        st_data.append([
            sc.scenario_name,
            sc.impact_description[:60],
            f"{sc.estimated_drawdown_pct:.1f}%",
        ])
    elements.append(
        _make_table(st_data, [1.5 * inch, 3.5 * inch, 1.5 * inch])
    )
    elements.append(
        Paragraph(
            f"<b>Overall Resilience:</b> "
            f"{risk.stress_test_results.overall_resilience}",
            s["Body"],
        )
    )

    return elements


def _returns_section(returns: ReturnsProjectionOutput, s) -> List[Any]:
    """Returns projection from Agent 07."""
    elements = []
    elements.append(Paragraph("Returns Projection", s["H1"]))

    # Scenario table
    elements.append(Paragraph("Scenario Analysis", s["H2"]))
    sc_data = [["Scenario", "Prob", "3m", "6m", "12m", "12m Gain"]]
    for sc in returns.scenarios:
        sc_data.append([
            sc.scenario.capitalize(),
            f"{sc.probability:.0%}",
            f"{sc.portfolio_return_3m:.1f}%",
            f"{sc.portfolio_return_6m:.1f}%",
            f"{sc.portfolio_return_12m:.1f}%",
            f"${sc.portfolio_gain_12m:,.0f}",
        ])
    elements.append(
        _make_table(
            sc_data,
            [1.2 * inch, 0.7 * inch, 1.0 * inch, 1.0 * inch,
             1.0 * inch, 1.6 * inch],
        )
    )

    # Expected returns
    er = returns.expected_return
    elements.append(Paragraph("Expected Returns (Probability-Weighted)", s["H2"]))
    er_data = [
        ["Horizon", "Expected Return"],
        ["3 Months", f"{er.expected_3m:.1f}%"],
        ["6 Months", f"{er.expected_6m:.1f}%"],
        ["12 Months", f"{er.expected_12m:.1f}%"],
    ]
    elements.append(_make_table(er_data, [3.25 * inch, 3.25 * inch]))

    # Benchmark comparison — summary
    elements.append(Paragraph("Benchmark Comparison (12-Month)", s["H2"]))
    bm_data = [["Benchmark", "Return", "Alpha", "Outperform Prob"]]
    for bc in returns.benchmark_comparisons:
        bm_data.append([
            f"{bc.symbol} ({bc.name})",
            f"{bc.benchmark_return_12m:.1f}%",
            f"{bc.expected_alpha_12m:+.1f}%",
            f"{bc.outperform_probability:.0%}",
        ])
    elements.append(
        _make_table(
            bm_data, [2.0 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch]
        )
    )

    # Portfolio vs S&P 500 & Nasdaq — scenario-by-scenario
    spy_bc = next((bc for bc in returns.benchmark_comparisons if bc.symbol == "SPY"), None)
    qqq_bc = next((bc for bc in returns.benchmark_comparisons if bc.symbol == "QQQ"), None)
    if spy_bc and qqq_bc:
        elements.append(Paragraph("Portfolio vs S&amp;P 500 &amp; Nasdaq-100 (by Scenario)", s["H2"]))
        vs_data = [
            ["Scenario", "Portfolio", "S&amp;P 500", "Alpha vs SPY", "Nasdaq-100", "Alpha vs QQQ"],
        ]
        scenario_map = {sc.scenario: sc for sc in returns.scenarios}
        for sc_name in ("bull", "base", "bear"):
            sc = scenario_map.get(sc_name)
            if not sc:
                continue
            spy_alpha = getattr(spy_bc, f"alpha_{sc_name}_12m", 0.0)
            qqq_alpha = getattr(qqq_bc, f"alpha_{sc_name}_12m", 0.0)
            # Derive per-scenario benchmark return from portfolio return - alpha
            spy_return = sc.portfolio_return_12m - spy_alpha
            qqq_return = sc.portfolio_return_12m - qqq_alpha
            vs_data.append([
                sc_name.capitalize(),
                f"{sc.portfolio_return_12m:.1f}%",
                f"{spy_return:.1f}%",
                f"{spy_alpha:+.1f}%",
                f"{qqq_return:.1f}%",
                f"{qqq_alpha:+.1f}%",
            ])
        # Expected (probability-weighted)
        vs_data.append([
            "Expected",
            f"{returns.expected_return.expected_12m:.1f}%",
            f"{spy_bc.benchmark_return_12m:.1f}%",
            f"{spy_bc.expected_alpha_12m:+.1f}%",
            f"{qqq_bc.benchmark_return_12m:.1f}%",
            f"{qqq_bc.expected_alpha_12m:+.1f}%",
        ])

        vs_style = []
        for i in range(1, len(vs_data)):
            # Color alpha vs SPY
            alpha_str = vs_data[i][3]
            if alpha_str.startswith("+"):
                vs_style.append(("TEXTCOLOR", (3, i), (3, i), GREEN))
            elif alpha_str.startswith("-"):
                vs_style.append(("TEXTCOLOR", (3, i), (3, i), RED))
            # Color alpha vs QQQ
            alpha_str_q = vs_data[i][5]
            if alpha_str_q.startswith("+"):
                vs_style.append(("TEXTCOLOR", (5, i), (5, i), GREEN))
            elif alpha_str_q.startswith("-"):
                vs_style.append(("TEXTCOLOR", (5, i), (5, i), RED))
        vs_style.append(("ALIGN", (1, 0), (5, -1), "RIGHT"))
        elements.append(
            _make_table(
                vs_data,
                [1.0 * inch, 1.0 * inch, 1.0 * inch, 1.1 * inch, 1.1 * inch, 1.3 * inch],
                vs_style,
            )
        )

    # Risk metrics
    rm = returns.risk_metrics
    elements.append(Paragraph("Risk Metrics", s["H2"]))
    rm_data = [
        ["Metric", "Value"],
        ["Max Drawdown (with stops)", f"{rm.max_drawdown_with_stops:.1f}%"],
        ["Max Drawdown (without stops)", f"{rm.max_drawdown_without_stops:.1f}%"],
        ["Portfolio Volatility", f"{rm.projected_annual_volatility:.1f}%"],
        ["Sharpe Ratio (base)", f"{rm.portfolio_sharpe_base:.2f}"],
        ["SPY Sharpe (base)", f"{rm.spy_sharpe_base:.2f}"],
        [
            "Expected Stops Triggered (12m)",
            str(rm.expected_stops_triggered_12m),
        ],
    ]
    elements.append(_make_table(rm_data, [3.25 * inch, 3.25 * inch]))

    # Alpha analysis
    aa = returns.alpha_analysis
    elements.append(
        Paragraph(
            f"<b>Net Alpha vs SPY:</b> {aa.net_expected_alpha_vs_spy:+.1f}% &nbsp; | &nbsp; "
            f"<b>vs DIA:</b> {aa.net_expected_alpha_vs_dia:+.1f}% &nbsp; | &nbsp; "
            f"<b>vs QQQ:</b> {aa.net_expected_alpha_vs_qqq:+.1f}%",
            s["Body"],
        )
    )

    return elements


def _implementation_section(reconciler: ReconciliationOutput, s) -> List[Any]:
    """Implementation plan from Agent 08."""
    elements = []
    elements.append(Paragraph("Implementation Plan", s["H1"]))

    # Transformation metrics
    tm = reconciler.transformation_metrics
    elements.append(
        Paragraph(
            f"<b>Turnover:</b> {tm.turnover_pct:.0f}% &nbsp; | &nbsp; "
            f"<b>Actions:</b> {len(reconciler.actions)} &nbsp; | &nbsp; "
            f"<b>Positions:</b> {tm.before_position_count} → {tm.after_position_count} &nbsp; | &nbsp; "
            f"<b>Sectors:</b> {tm.before_sector_count} → {tm.after_sector_count}",
            s["Body"],
        )
    )

    # Money flow
    mf = reconciler.money_flow
    elements.append(Paragraph("Capital Flow", s["H2"]))
    mf_data = [
        ["Flow", "Amount"],
        ["Sell Proceeds", f"${mf.sell_proceeds:,.0f}"],
        ["Trim Proceeds", f"${mf.trim_proceeds:,.0f}"],
        ["Purchase Cost", f"${mf.buy_cost:,.0f}"],
        ["Add Cost", f"${mf.add_cost:,.0f}"],
        ["Net Cash Change", f"${mf.net_cash_change:,.0f}"],
        ["Cash Reserve", f"{mf.cash_reserve_pct:.1f}%"],
    ]
    elements.append(_make_table(mf_data, [3.25 * inch, 3.25 * inch]))

    # 4-week plan
    elements.append(Paragraph("4-Week Timeline", s["H2"]))
    for week in reconciler.implementation_plan:
        elements.append(
            Paragraph(
                f"<b>Week {week.week}:</b> {week.phase_name}",
                s["Body"],
            )
        )
        action_count = len(week.actions)
        if action_count > 0:
            elements.append(
                Paragraph(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;{action_count} actions scheduled",
                    s["Small"],
                )
            )

    return elements


def _current_vs_tiers_section(
    reconciler: ReconciliationOutput,
    portfolio: PortfolioOutput,
    analyst: AnalystOutput,
    research: ResearchOutput,
    s,
) -> List[Any]:
    """Current portfolio analysis against recommended tiers."""
    elements = []
    elements.append(Paragraph("Current Portfolio vs Recommended Tiers", s["H1"]))

    # Build sector lookup: research → analyst → portfolio (later overwrites)
    sector_lookup: dict[str, str] = {}
    for rs in research.stocks:
        sector_lookup[rs.symbol] = rs.sector
    for re_ in research.etfs:
        sector_lookup[re_.symbol] = re_.focus
    for rs in analyst.rated_stocks:
        sector_lookup[rs.symbol] = rs.sector
    for re_ in analyst.rated_etfs:
        sector_lookup[re_.symbol] = re_.focus or ", ".join(re_.theme_tags[:1]) or "ETF"

    # Build tier lookup from recommended portfolio
    tier_lookup: dict[str, int] = {}  # symbol → tier
    rec_lookup: dict[str, Any] = {}   # symbol → PortfolioPosition
    for tier_num, tier_obj in [(1, portfolio.tier_1), (2, portfolio.tier_2), (3, portfolio.tier_3)]:
        for pos in tier_obj.positions:
            tier_lookup[pos.symbol] = tier_num
            rec_lookup[pos.symbol] = pos
            # Also populate sector from portfolio positions
            if pos.symbol not in sector_lookup:
                sector_lookup[pos.symbol] = pos.sector

    # Build action lookup
    action_lookup: dict[str, Any] = {}  # symbol → PositionAction
    for act in reconciler.actions:
        action_lookup[act.symbol] = act

    # --- Current Holdings Summary ---
    ch = reconciler.current_holdings
    elements.append(
        Paragraph(
            f"<b>Current Portfolio:</b> {len(ch.holdings)} positions &nbsp; | &nbsp; "
            f"<b>Total Value:</b> ${ch.total_value:,.0f} &nbsp; | &nbsp; "
            f"<b>Accounts:</b> {ch.account_count}",
            s["Body"],
        )
    )

    # --- Tier alignment table ---
    # Categorize current holdings by their recommended tier
    tier_buckets: dict[str, list] = {
        "Tier 1": [], "Tier 2": [], "Tier 3": [], "Not Recommended": [],
    }
    for h in ch.holdings:
        tier_num = tier_lookup.get(h.symbol)
        if tier_num:
            tier_buckets[f"Tier {tier_num}"].append(h)
        else:
            tier_buckets["Not Recommended"].append(h)

    # Tier alignment summary
    elements.append(Paragraph("Current Holdings by Recommended Tier", s["H2"]))
    align_data = [["Category", "Positions", "Market Value", "% of Current", "Status"]]
    for cat in ["Tier 1", "Tier 2", "Tier 3", "Not Recommended"]:
        bucket = tier_buckets[cat]
        count = len(bucket)
        value = sum(h.market_value for h in bucket)
        pct = (value / ch.total_value * 100) if ch.total_value > 0 else 0
        status = "Aligned" if cat != "Not Recommended" else "To Sell/Liquidate"
        align_data.append([cat, str(count), f"${value:,.0f}", f"{pct:.1f}%", status])

    total_aligned = sum(len(tier_buckets[t]) for t in ["Tier 1", "Tier 2", "Tier 3"])
    total_not = len(tier_buckets["Not Recommended"])
    align_data.append([
        "Total", str(len(ch.holdings)),
        f"${ch.total_value:,.0f}", "100.0%",
        f"{total_aligned} aligned, {total_not} to exit",
    ])

    align_style = []
    for i in range(1, len(align_data)):
        if align_data[i][0] == "Not Recommended":
            align_style.append(("TEXTCOLOR", (4, i), (4, i), RED))
        elif align_data[i][0].startswith("Tier"):
            align_style.append(("TEXTCOLOR", (4, i), (4, i), GREEN))
    elements.append(
        _make_table(
            align_data,
            [1.3 * inch, 0.9 * inch, 1.3 * inch, 1.2 * inch, 1.8 * inch],
            align_style,
        )
    )

    # --- Current vs Recommended per position ---
    elements.append(PageBreak())
    elements.append(Paragraph("Position-Level Gap Analysis", s["H1"]))

    # Collect all symbols: from current holdings + all recommended
    all_symbols = set(h.symbol for h in ch.holdings) | set(tier_lookup.keys())
    current_map = {}
    for h in ch.holdings:
        raw = sector_lookup.get(h.symbol, h.sector)
        resolved_sector = raw if raw != "UNKNOWN" else "—"
        if h.symbol in current_map:
            current_map[h.symbol]["value"] += h.market_value
            current_map[h.symbol]["shares"] += h.shares
        else:
            current_map[h.symbol] = {
                "value": h.market_value, "shares": h.shares,
                "sector": resolved_sector, "account": h.account,
            }

    # Group actions by type for the summary
    action_counts: dict[str, int] = {"KEEP": 0, "SELL": 0, "BUY": 0, "ADD": 0, "TRIM": 0}
    action_values: dict[str, float] = {"KEEP": 0, "SELL": 0, "BUY": 0, "ADD": 0, "TRIM": 0}
    for act in reconciler.actions:
        action_counts[act.action_type] = action_counts.get(act.action_type, 0) + 1
        action_values[act.action_type] = action_values.get(act.action_type, 0) + abs(act.dollar_change)

    # Action summary
    elements.append(Paragraph("Action Summary", s["H2"]))
    act_data = [["Action", "Count", "Dollar Impact"]]
    action_colors_map = {
        "KEEP": GREEN, "BUY": BLUE, "ADD": LIGHT_GREEN,
        "SELL": RED, "TRIM": ORANGE,
    }
    act_style = []
    for i, atype in enumerate(["KEEP", "BUY", "ADD", "TRIM", "SELL"], 1):
        cnt = action_counts.get(atype, 0)
        val = action_values.get(atype, 0)
        prefix = "+" if atype in ("BUY", "ADD") else "-" if atype in ("SELL", "TRIM") else ""
        act_data.append([atype, str(cnt), f"{prefix}${val:,.0f}"])
        color = action_colors_map.get(atype, BLACK)
        act_style.append(("TEXTCOLOR", (0, i), (0, i), color))
    act_data.append(["Total", str(len(reconciler.actions)), ""])
    elements.append(
        _make_table(act_data, [2.0 * inch, 1.5 * inch, 3.0 * inch], act_style)
    )

    # --- Per-tier gap tables ---
    for tier_num, tier_label in [(1, "Tier 1 — Momentum"), (2, "Tier 2 — Quality Growth"), (3, "Tier 3 — Defensive")]:
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph(f"{tier_label} — Gap Analysis", s["H2"]))

        # Get all recommended positions for this tier
        tier_obj = [portfolio.tier_1, portfolio.tier_2, portfolio.tier_3][tier_num - 1]
        if not tier_obj.positions:
            elements.append(Paragraph("No positions in this tier.", s["Body"]))
            continue

        gap_data = [["Symbol", "Company", "Action", "Cur%", "Tgt%", "Gap%", "$ Change", "Sharpe", "Alpha%"]]
        gap_style_extra = []
        row_idx = 1

        sorted_positions = sorted(tier_obj.positions, key=lambda p: -p.target_pct)
        for pos in sorted_positions:
            act = action_lookup.get(pos.symbol)
            action_type = act.action_type if act else "BUY"
            current_pct = act.current_pct if act else 0.0
            target_pct = act.target_pct if act else pos.target_pct
            gap = target_pct - current_pct
            dollar = act.dollar_change if act else 0.0
            sharpe = act.sharpe_ratio if act and act.sharpe_ratio is not None else None
            alpha = act.alpha_pct if act and act.alpha_pct is not None else None

            gap_data.append([
                pos.symbol,
                pos.company_name[:18],
                action_type,
                f"{current_pct:.2f}",
                f"{target_pct:.2f}",
                f"{gap:+.2f}",
                f"${dollar:+,.0f}",
                f"{sharpe:.2f}" if sharpe is not None else "—",
                f"{alpha:+.1f}" if alpha is not None else "—",
            ])

            # Color action type
            color = action_colors_map.get(action_type, BLACK)
            gap_style_extra.append(("TEXTCOLOR", (2, row_idx), (2, row_idx), color))
            # Color gap
            if gap > 0.1:
                gap_style_extra.append(("TEXTCOLOR", (5, row_idx), (5, row_idx), GREEN))
            elif gap < -0.1:
                gap_style_extra.append(("TEXTCOLOR", (5, row_idx), (5, row_idx), RED))
            # Color Sharpe
            if sharpe is not None:
                if sharpe >= 1.0:
                    gap_style_extra.append(("TEXTCOLOR", (7, row_idx), (7, row_idx), GREEN))
                elif sharpe < 0.5:
                    gap_style_extra.append(("TEXTCOLOR", (7, row_idx), (7, row_idx), RED))
            # Color Alpha
            if alpha is not None:
                if alpha > 0:
                    gap_style_extra.append(("TEXTCOLOR", (8, row_idx), (8, row_idx), GREEN))
                else:
                    gap_style_extra.append(("TEXTCOLOR", (8, row_idx), (8, row_idx), RED))
            row_idx += 1

        gap_style_extra.append(("ALIGN", (3, 0), (8, -1), "RIGHT"))
        elements.append(
            _make_table(
                gap_data,
                [0.65 * inch, 1.2 * inch, 0.6 * inch, 0.6 * inch,
                 0.6 * inch, 0.6 * inch, 1.0 * inch, 0.65 * inch, 0.65 * inch],
                gap_style_extra,
            )
        )

    # --- Holdings not in recommended portfolio (to sell) ---
    not_recommended = tier_buckets["Not Recommended"]
    if not_recommended:
        elements.append(PageBreak())
        elements.append(Paragraph("Holdings Not in Recommended Portfolio", s["H2"]))
        elements.append(
            Paragraph(
                "These positions are in the current portfolio but not in any recommended tier. "
                "They are scheduled for liquidation.",
                s["Body"],
            )
        )
        nr_data = [[
            Paragraph("<b>Symbol</b>", s["CellBold"]),
            Paragraph("<b>Sector</b>", s["CellBold"]),
            Paragraph("<b>Value</b>", s["CellBold"]),
            Paragraph("<b>Cur%</b>", s["CellBold"]),
            Paragraph("<b>Sharpe</b>", s["CellBold"]),
            Paragraph("<b>Alpha%</b>", s["CellBold"]),
            Paragraph("<b>Rationale</b>", s["CellBold"]),
        ]]
        nr_style = []
        for i, h in enumerate(sorted(not_recommended, key=lambda x: -x.market_value), 1):
            act = action_lookup.get(h.symbol)
            current_pct = act.current_pct if act else (h.market_value / ch.total_value * 100 if ch.total_value else 0)
            rationale = act.rationale if act else "Not in recommended portfolio"
            raw_sector = sector_lookup.get(h.symbol, h.sector)
            sector = raw_sector if raw_sector != "UNKNOWN" else "—"
            sharpe = act.sharpe_ratio if act and act.sharpe_ratio is not None else None
            alpha = act.alpha_pct if act and act.alpha_pct is not None else None
            nr_data.append([
                Paragraph(f"<b>{h.symbol}</b>", s["CellWrap"]),
                Paragraph(sector, s["CellWrap"]),
                Paragraph(f"${h.market_value:,.0f}", s["CellWrap"]),
                Paragraph(f"{current_pct:.2f}%", s["CellWrap"]),
                Paragraph(f"{sharpe:.2f}" if sharpe is not None else "—", s["CellWrap"]),
                Paragraph(f"{alpha:+.1f}%" if alpha is not None else "—", s["CellWrap"]),
                Paragraph(rationale, s["CellWrap"]),
            ])
            # Color Sharpe
            if sharpe is not None:
                if sharpe >= 1.0:
                    nr_style.append(("TEXTCOLOR", (4, i), (4, i), GREEN))
                elif sharpe < 0.5:
                    nr_style.append(("TEXTCOLOR", (4, i), (4, i), RED))
            # Color Alpha
            if alpha is not None:
                if alpha > 0:
                    nr_style.append(("TEXTCOLOR", (5, i), (5, i), GREEN))
                else:
                    nr_style.append(("TEXTCOLOR", (5, i), (5, i), RED))
        nr_style.append(("VALIGN", (0, 0), (-1, -1), "TOP"))
        nr_style.extend(HEADER_STYLE)
        nr_style.extend(_zebra_rows(len(nr_data)))
        nr_tbl = Table(
            nr_data,
            colWidths=[0.6 * inch, 1.0 * inch, 0.8 * inch, 0.55 * inch, 0.55 * inch, 0.6 * inch, 2.4 * inch],
            repeatRows=1,
        )
        nr_tbl.setStyle(TableStyle(nr_style))
        elements.append(nr_tbl)

    # --- Keep verification ---
    kv = reconciler.keep_verification
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph("Keep Position Verification", s["H2"]))
    elements.append(
        Paragraph(
            f"<b>Keeps already held:</b> {len(kv.keeps_in_current)} &nbsp; | &nbsp; "
            f"<b>Keeps missing (to buy):</b> {len(kv.keeps_to_buy)} &nbsp; | &nbsp; "
            f"<b>Total keeps verified:</b> {len(kv.keeps_in_current) + len(kv.keeps_to_buy)}",
            s["Body"],
        )
    )
    if kv.keeps_in_current:
        elements.append(
            Paragraph(
                f'<font color="#00B050"><b>Held:</b></font> {", ".join(sorted(kv.keeps_in_current))}',
                s["Reasoning"],
            )
        )
    if kv.keeps_to_buy:
        elements.append(
            Paragraph(
                f'<font color="#4472C4"><b>To acquire:</b></font> {", ".join(sorted(kv.keeps_to_buy))}',
                s["Reasoning"],
            )
        )

    return elements


def _historical_section(
    historical: HistoricalAnalysisOutput, s
) -> List[Any]:
    """Historical context analysis from Agent 13."""
    elements = []
    elements.append(Paragraph("Historical Context Analysis", s["H1"]))

    # Store metadata
    meta = historical.store_meta
    if not meta.store_available:
        elements.append(
            Paragraph(
                "Historical store is not available. Install chromadb "
                "and ingest past IBD data to enable historical context.",
                s["Body"],
            )
        )
        return elements

    elements.append(
        Paragraph(
            f"<b>Source:</b> {historical.historical_source} &nbsp; | &nbsp; "
            f"<b>Snapshots:</b> {meta.total_snapshots} &nbsp; | &nbsp; "
            f"<b>Records:</b> {meta.total_records:,} &nbsp; | &nbsp; "
            f"<b>Symbols Tracked:</b> {meta.unique_symbols:,}",
            s["Body"],
        )
    )
    if meta.date_range:
        elements.append(
            Paragraph(
                f"<b>Date Range:</b> {meta.date_range[0]} to {meta.date_range[-1]}",
                s["Body"],
            )
        )

    # Momentum summary
    total = len(historical.stock_analyses)
    improving = len(historical.improving_stocks)
    deteriorating = len(historical.deteriorating_stocks)
    stable = total - improving - deteriorating

    elements.append(Paragraph("Momentum Overview", s["H2"]))
    mom_data = [
        ["Direction", "Count", "% of Analyzed"],
        [
            "Improving",
            str(improving),
            f"{improving / total * 100:.0f}%" if total else "0%",
        ],
        [
            "Stable",
            str(stable),
            f"{stable / total * 100:.0f}%" if total else "0%",
        ],
        [
            "Deteriorating",
            str(deteriorating),
            f"{deteriorating / total * 100:.0f}%" if total else "0%",
        ],
        ["Total Analyzed", str(total), "100%"],
    ]
    mom_style = []
    if len(mom_data) > 1:
        mom_style.append(("TEXTCOLOR", (0, 1), (0, 1), GREEN))
        mom_style.append(("TEXTCOLOR", (0, 3), (0, 3), RED))
    elements.append(
        _make_table(mom_data, [2.5 * inch, 2.0 * inch, 2.0 * inch], mom_style)
    )

    # Top improving stocks
    if historical.improving_stocks:
        elements.append(Paragraph("Top Improving Stocks", s["H2"]))
        analyzed = {sa.symbol: sa for sa in historical.stock_analyses}
        imp_data = [["#", "Symbol", "Momentum Score", "Weeks Tracked"]]
        top_imp = sorted(
            [analyzed[sym] for sym in historical.improving_stocks if sym in analyzed],
            key=lambda x: -x.momentum_score,
        )[:10]
        for i, sa in enumerate(top_imp, 1):
            imp_data.append([
                str(i), sa.symbol,
                f"{sa.momentum_score:+.1f}",
                str(sa.weeks_tracked),
            ])
        elements.append(
            _make_table(
                imp_data,
                [0.5 * inch, 1.5 * inch, 2.5 * inch, 2.0 * inch],
            )
        )

    # Deteriorating stocks
    if historical.deteriorating_stocks:
        elements.append(Paragraph("Deteriorating Stocks", s["H2"]))
        det_data = [["#", "Symbol", "Momentum Score", "Weeks Tracked"]]
        top_det = sorted(
            [analyzed[sym] for sym in historical.deteriorating_stocks if sym in analyzed],
            key=lambda x: x.momentum_score,
        )[:10]
        for i, sa in enumerate(top_det, 1):
            det_data.append([
                str(i), sa.symbol,
                f"{sa.momentum_score:+.1f}",
                str(sa.weeks_tracked),
            ])
        elements.append(
            _make_table(
                det_data,
                [0.5 * inch, 1.5 * inch, 2.5 * inch, 2.0 * inch],
            )
        )

    # Historical analogs
    if historical.historical_analogs:
        elements.append(Paragraph("Historical Analogs", s["H2"]))
        elements.append(
            Paragraph(
                "Past stocks with similar rating profiles to current pipeline stocks:",
                s["Body"],
            )
        )
        analog_data = [
            ["Current", "Analog", "Date", "Sector", "Comp", "RS", "Similarity"]
        ]
        for a in historical.historical_analogs[:15]:
            analog_data.append([
                a.current_symbol, a.symbol, a.analog_date,
                a.sector[:12],
                str(a.composite_rating) if a.composite_rating else "-",
                str(a.rs_rating) if a.rs_rating else "-",
                f"{a.similarity_score:.0%}",
            ])
        elements.append(
            _make_table(
                analog_data,
                [0.8 * inch, 0.8 * inch, 1.0 * inch, 1.2 * inch,
                 0.7 * inch, 0.7 * inch, 1.0 * inch],
            )
        )

    # Sector trends
    if historical.sector_trends:
        elements.append(Paragraph("Sector Historical Trends", s["H2"]))
        sec_data = [["Sector", "RS Trend", "Stock Count", "Elite %"]]
        for st in historical.sector_trends:
            sec_data.append([
                st.sector, st.avg_rs_trend.capitalize(),
                st.stock_count_trend.capitalize(),
                st.elite_pct_trend.capitalize(),
            ])
        trend_style = []
        for i in range(1, len(sec_data)):
            trend_val = sec_data[i][1]
            if trend_val == "Improving":
                trend_style.append(("TEXTCOLOR", (1, i), (1, i), GREEN))
            elif trend_val == "Deteriorating":
                trend_style.append(("TEXTCOLOR", (1, i), (1, i), RED))
        elements.append(
            _make_table(
                sec_data,
                [2.0 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch],
                trend_style,
            )
        )

    # IBD list movements
    if historical.new_to_ibd_lists or historical.dropped_from_ibd_lists:
        elements.append(Paragraph("IBD List Movements", s["H2"]))
        if historical.new_to_ibd_lists:
            new_syms = ", ".join(historical.new_to_ibd_lists[:20])
            elements.append(
                Paragraph(
                    f'<font color="#00B050"><b>New entries:</b></font> {new_syms}',
                    s["Body"],
                )
            )
        if historical.dropped_from_ibd_lists:
            drop_syms = ", ".join(historical.dropped_from_ibd_lists[:20])
            elements.append(
                Paragraph(
                    f'<font color="#FF0000"><b>Dropped:</b></font> {drop_syms}',
                    s["Body"],
                )
            )

    return elements


def _agent_summaries(
    research: ResearchOutput,
    analyst: AnalystOutput,
    rotation: RotationDetectionOutput,
    strategy: SectorStrategyOutput,
    portfolio: PortfolioOutput,
    risk: RiskAssessment,
    returns: ReturnsProjectionOutput,
    reconciler: ReconciliationOutput,
    educator,
    synthesis: SynthesisOutput,
    value: ValueInvestorOutput,
    pattern: PortfolioPatternOutput,
    historical: Optional[HistoricalAnalysisOutput],
    s,
) -> List[Any]:
    """Summary paragraph for each of the 13 agents."""
    elements = []
    elements.append(Paragraph("Agent Summaries", s["H1"]))

    agents = [
        ("Agent 01: Research", research.summary),
        ("Agent 02: Analyst", analyst.summary),
        ("Agent 03: Rotation Detector", rotation.summary),
        ("Agent 04: Sector Strategist", strategy.summary),
        ("Agent 05: Portfolio Manager", portfolio.summary),
        ("Agent 06: Risk Officer", risk.summary),
        ("Agent 07: Returns Projector", returns.summary),
        ("Agent 08: Reconciler", reconciler.summary),
        (
            "Agent 09: Educator",
            educator.summary if educator else "Educator output not available.",
        ),
        ("Agent 10: Synthesizer", synthesis.summary),
        ("Agent 11: Value Investor", value.summary),
        ("Agent 12: PatternAlpha", pattern.summary),
        (
            "Agent 13: Historical Analyst",
            historical.summary if historical else "Historical analysis not available.",
        ),
    ]

    for name, summary in agents:
        elements.append(Paragraph(f"<b>{name}</b>", s["H2"]))
        elements.append(Paragraph(summary, s["Body"]))

    return elements


def _action_items(synthesis: SynthesisOutput, s) -> List[Any]:
    """Action items from Agent 10."""
    elements = []
    elements.append(Paragraph("Action Items", s["H1"]))
    elements.append(
        Paragraph("Prioritized next steps from cross-agent analysis:", s["Body"])
    )
    for i, item in enumerate(synthesis.action_items, 1):
        elements.append(
            Paragraph(f"<b>{i}.</b> {item}", s["Body"])
        )
    return elements


def _disclaimer(synthesis: SynthesisOutput, s) -> List[Any]:
    """Disclaimer page."""
    elements = []
    elements.append(Paragraph("Disclaimer", s["H1"]))
    elements.append(
        Paragraph(
            "This report was generated by the IBD Momentum Investment "
            "Framework v4.0, a multi-agent analytical system. The framework "
            "discovers, scores, and organizes investment opportunities based "
            "on quantitative metrics from Investor's Business Daily, "
            "Schwab thematic research, and Motley Fool recommendations.",
            s["Body"],
        )
    )
    elements.append(
        Paragraph(
            "<b>This framework never makes buy or sell recommendations.</b> "
            "All analysis is for informational and educational purposes only. "
            "Securities are rated, tiered, and organized — not recommended "
            "for purchase or sale. Position sizing reflects framework "
            "construction rules, not investment advice.",
            s["Body"],
        )
    )
    elements.append(
        Paragraph(
            "Past performance does not guarantee future results. "
            "All projections are based on historical characteristics "
            "and statistical models. Actual returns depend on market "
            "conditions, execution timing, and numerous other factors "
            "not captured in this analysis.",
            s["Body"],
        )
    )
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(
        Paragraph(
            f"Report Date: {synthesis.analysis_date} &nbsp; | &nbsp; "
            f"Source: {synthesis.synthesis_source.upper()}",
            s["Disclaimer"],
        )
    )
    return elements


# ---------------------------------------------------------------------------
# Page Number Footer
# ---------------------------------------------------------------------------


def _add_page_number(canvas, doc):
    """Add page number to each page."""
    page_num = canvas.getPageNumber()
    text = f"IBD Momentum Framework v4.0 — Page {page_num}"
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#808080"))
    canvas.drawString(0.75 * inch, 0.5 * inch, text)
    canvas.drawRightString(
        letter[0] - 0.75 * inch,
        0.5 * inch,
        date.today().isoformat(),
    )
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def generate_pdf_report(
    research_output: ResearchOutput,
    analyst_output: AnalystOutput,
    rotation_output: RotationDetectionOutput,
    strategy_output: SectorStrategyOutput,
    portfolio_output: PortfolioOutput,
    risk_output: RiskAssessment,
    returns_output: ReturnsProjectionOutput,
    reconciler_output: ReconciliationOutput,
    educator_output,
    synthesis_output: SynthesisOutput,
    value_output: ValueInvestorOutput,
    pattern_output: PortfolioPatternOutput,
    out_path: Path,
    historical_output: Optional[HistoricalAnalysisOutput] = None,
) -> Path:
    """
    Generate comprehensive PDF report from all 13 agent outputs.

    Args:
        *_output: Validated output from each of the 13 agents.
        out_path: Directory to write the PDF file.

    Returns:
        Path to the generated PDF file.
    """
    today = date.today().isoformat()
    filepath = out_path / f"framework_report_{today}.pdf"
    s = _styles()

    doc = SimpleDocTemplate(
        str(filepath),
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        title="IBD Momentum Investment Framework v4.0 Report",
        author="IBD Multi-Agent System",
    )

    story: list[Any] = []

    # 1. Title page
    story.extend(_title_page(synthesis_output, s))
    story.append(PageBreak())

    # 2. Executive summary
    story.extend(_executive_summary(synthesis_output, s))
    story.append(PageBreak())

    # 3. Key numbers dashboard
    story.extend(_key_numbers(synthesis_output, s))
    story.append(PageBreak())

    # 4. Market & rotation
    story.extend(_rotation_section(rotation_output, s))
    story.append(PageBreak())

    # 5. Portfolio construction
    story.extend(_portfolio_section(portfolio_output, s))
    story.append(PageBreak())

    # 6. Top picks
    story.extend(
        _top_picks_section(analyst_output, value_output, pattern_output, s)
    )
    story.append(PageBreak())

    # 7. Risk assessment
    story.extend(_risk_section(risk_output, s))
    story.append(PageBreak())

    # 8. Returns projection
    story.extend(_returns_section(returns_output, s))
    story.append(PageBreak())

    # 9. Implementation plan
    story.extend(_implementation_section(reconciler_output, s))
    story.append(PageBreak())

    # 10. Current portfolio vs recommended tiers
    story.extend(
        _current_vs_tiers_section(
            reconciler_output, portfolio_output, analyst_output,
            research_output, s,
        )
    )
    story.append(PageBreak())

    # 11. Historical context analysis
    if historical_output:
        story.extend(_historical_section(historical_output, s))
        story.append(PageBreak())

    # 11. Agent summaries
    story.extend(
        _agent_summaries(
            research_output, analyst_output, rotation_output,
            strategy_output, portfolio_output, risk_output,
            returns_output, reconciler_output, educator_output,
            synthesis_output, value_output, pattern_output,
            historical_output, s,
        )
    )
    story.append(PageBreak())

    # 11. Action items
    story.extend(_action_items(synthesis_output, s))
    story.append(PageBreak())

    # 12. Disclaimer
    story.extend(_disclaimer(synthesis_output, s))

    # Build PDF
    doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)

    logger.info(f"[PDF] Generated report: {filepath}")
    return filepath
