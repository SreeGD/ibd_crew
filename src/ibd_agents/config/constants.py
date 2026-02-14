"""
Centralized configuration for IBD Framework v4.0

This module defines all magic numbers, thresholds, and configuration values
used throughout the IBD Crew system. Centralizing these values makes it easier
to tune the system and understand decision boundaries.

Reference: IBD Momentum Investment Framework v4.0
"""

# ============================================================================
# ELITE SCREENING THRESHOLDS (§3.2.1)
# ============================================================================
# These are the four gates for elite stock screening:
ELITE_COMPOSITE_THRESHOLD = 85
"""Minimum Composite Rating to pass elite screening"""

ELITE_RS_THRESHOLD = 75
"""Minimum Relative Strength rating to pass elite screening"""

ELITE_SMR_GRADES = {"A", "B", "B-"}
"""Acceptable SMR (Sales+Profit+ROE) grades for elite screening"""

ELITE_ACC_DIS_GRADES = {"A", "B", "B-"}
"""Acceptable Accumulation/Distribution grades for elite screening"""

# ============================================================================
# IBD KEEP CANDIDATE CRITERIA (§3.2.2)
# ============================================================================
# IBD Keep candidates are stocks meeting these strict criteria:
KEEP_MIN_COMPOSITE = 93
"""Minimum Composite Rating for IBD Keep candidate designation"""

KEEP_MIN_RS = 90
"""Minimum RS Rating for IBD Keep candidate designation"""

# ============================================================================
# TIER ASSIGNMENT THRESHOLDS (§4.1)
# ============================================================================
# Used to classify stocks into momentum (T1), quality (T2), defensive (T3)

# Tier 1: Momentum Leaders
TIER_1_COMPOSITE_MIN = 85
"""Minimum Composite for Tier 1 (Momentum)"""

TIER_1_RS_MIN = 75
"""Minimum RS for Tier 1 (Momentum)"""

TIER_1_EPS_MIN = 70
"""Minimum EPS for Tier 1 (Momentum)"""

# Tier 2: Quality Growth
TIER_2_COMPOSITE_MIN = 70
"""Minimum Composite for Tier 2 (Quality Growth)"""

TIER_2_RS_MIN = 60
"""Minimum RS for Tier 2 (Quality Growth)"""

# Tier 3: Defensive Value
TIER_3_COMPOSITE_MIN = 60
"""Minimum Composite for Tier 3 (Defensive)"""

TIER_3_RS_MIN = 40
"""Minimum RS for Tier 3 (Defensive)"""

# ============================================================================
# CONFIDENCE & VALIDATION SCORING (§3.3)
# ============================================================================
# Confidence is based on data completeness from multiple sources

CONFIDENCE_BASE = 0.3
"""Base confidence score (30% with no ratings data)"""

CONFIDENCE_PER_RATING = 0.14
"""Increment per IBD rating available (0.14 × 5 ratings = 70% max)"""

# Multi-source validation score thresholds
VALIDATION_SCORE_MULTI_SOURCE_MIN = 5
"""Minimum validation score for multi-source validation badge"""

VALIDATION_SOURCES_MIN = 2
"""Minimum number of sources for multi-source validation"""

# ============================================================================
# VALUATION METRICS & P/E CATEGORIES (§5.1)
# ============================================================================
# Used to classify stocks by valuation relative to growth

PE_DEEP_VALUE_MAX = 10
"""P/E threshold for Deep Value category (undervalued)"""

PE_VALUE_MAX = 15
"""P/E threshold for Value category"""

PE_REASONABLE_MAX = 22
"""P/E threshold for Reasonable category (fair value)"""

PE_GROWTH_PREMIUM_MAX = 35
"""P/E threshold for Growth Premium category"""

# Everything above 35 is Speculative

PEG_UNDERVALUED_THRESHOLD = 1.0
"""PEG ≤ 1.0 indicates undervalued stock"""

PEG_FAIR_VALUE_THRESHOLD = 2.0
"""PEG ≤ 2.0 indicates fairly valued stock"""

# PEG > 2.0 is Expensive

# ============================================================================
# RETURN & RISK CATEGORIES (§5.2)
# ============================================================================
# Used to classify return expectations and Sharpe ratios

RETURN_STRONG_MIN = 0.15  # 15%
"""Minimum annual return expectation for Strong category"""

RETURN_GOOD_MIN = 0.10  # 10%
"""Minimum annual return expectation for Good category"""

RETURN_MODERATE_MIN = 0.05  # 5%
"""Minimum annual return expectation for Moderate category"""

SHARPE_EXCELLENT_MIN = 1.0
"""Sharpe ratio ≥ 1.0 indicates Excellent risk-adjusted returns"""

SHARPE_GOOD_MIN = 0.5
"""Sharpe ratio ≥ 0.5 indicates Good risk-adjusted returns"""

SHARPE_MODERATE_MIN = 0.0
"""Sharpe ratio ≥ 0.0 indicates Moderate risk-adjusted returns"""

ALPHA_STRONG_OUTPERFORMER_MIN = 5.0  # 5%
"""Alpha ≥ 5% indicates Strong Outperformer"""

ALPHA_OUTPERFORMER_MIN = 2.0  # 2%
"""Alpha ≥ 2% indicates Outperformer"""

# ============================================================================
# ROTATION DETECTION THRESHOLDS (§6.1)
# ============================================================================
# Used by rotation detector to identify sector shifts

ROTATION_CONFIDENCE_THRESHOLD = 0.60
"""Minimum confidence (0-1) to declare rotation active"""

ROTATION_RS_DIVERGENCE_THRESHOLD = 10
"""Minimum RS divergence between source and destination sectors (%)"""

ROTATION_ELITE_SHIFT_THRESHOLD = 5
"""Minimum elite concentration shift (% points) for signal trigger"""

ROTATION_BREADTH_SHIFT_THRESHOLD = 0.15
"""Minimum breadth shift (decimal) for signal trigger"""

# ============================================================================
# RISK MANAGEMENT PARAMETERS (§7.1)
# ============================================================================

# Sleep Well Score components
SLEEP_WELL_EXCELLENT_MIN = 8.0
"""Sleep Well score ≥ 8.0 indicates Excellent risk management"""

SLEEP_WELL_GOOD_MIN = 6.0
"""Sleep Well score ≥ 6.0 indicates Good risk management"""

SLEEP_WELL_MODERATE_MIN = 4.0
"""Sleep Well score ≥ 4.0 indicates Moderate risk management"""

# Drawdown estimation
DEFAULT_MAX_DRAWDOWN_PERCENT = 0.20  # 20%
"""Default maximum drawdown assumption if not calculated"""

VOLATILITY_ADJUSTMENT_EXTREME = 1.5
"""Position size reduction for extreme volatility (>30% annualized)"""

VOLATILITY_ADJUSTMENT_HIGH = 1.25
"""Position size reduction for high volatility (20-30% annualized)"""

VOLATILITY_ADJUSTMENT_NORMAL = 1.0
"""Position size for normal volatility (10-20% annualized)"""

# ============================================================================
# PORTFOLIO CONSTRUCTION (§8.1)
# ============================================================================

# Tier allocation targets
TIER_1_ALLOCATION_TARGET = 0.40  # 40%
"""Target allocation to Tier 1 (Momentum) positions"""

TIER_2_ALLOCATION_TARGET = 0.35  # 35%
"""Target allocation to Tier 2 (Quality) positions"""

TIER_3_ALLOCATION_TARGET = 0.15  # 15%
"""Target allocation to Tier 3 (Defensive) positions"""

CASH_ALLOCATION_TARGET = 0.10  # 10%
"""Target cash reserve allocation"""

# Position sizing
MAX_SINGLE_POSITION = 0.08  # 8%
"""Maximum portfolio weight for single position"""

MIN_POSITION_SIZE = 0.01  # 1%
"""Minimum portfolio weight for position to be held"""

# ============================================================================
# FILE PROCESSING ORDER (§2.1)
# ============================================================================
# Determines priority and weighting in data source hierarchy

FILE_PROCESSING_ORDER = [
    "ibd_xls",          # Priority 1: Structured data, all 5 ratings
    "ibd_pdf",          # Priority 2: IBD's official lists
    "fool_pdf",         # Priority 3: Supplementary validation
    "morningstar",      # Priority 4: Star ratings and moat analysis
    "schwab_themes",    # Priority 5: Thematic categorization only
]

"""Order in which data sources are processed. Earlier sources take precedence."""

# ============================================================================
# EXCEL OUTPUT FORMATTING (§9.1)
# ============================================================================

EXCEL_COLORS = {
    "dark_green": "006400",
    "light_green": "90EE90",
    "yellow": "FFFF00",
    "orange": "FFA500",
    "red": "FF0000",
    "white": "FFFFFF",
}
"""Color palette for Excel output formatting"""

EXCEL_COLUMN_WIDTH_DEFAULT = 20
"""Default column width in Excel output"""

EXCEL_COLUMN_WIDTH_WIDE = 50
"""Wide column width for text fields"""

# ============================================================================
# LOGGING & PERFORMANCE (§10.1)
# ============================================================================

LOG_LEVEL_DEFAULT = "INFO"
"""Default logging level"""

LOG_LEVEL_DEBUG = "DEBUG"
"""Debug logging level (verbose)"""

MAX_LOG_FILE_SIZE = 10 * 1024 * 1024  # 10MB
"""Maximum size of a log file before rotation"""

LOG_BACKUP_COUNT = 5
"""Number of old log files to keep"""

# Performance thresholds for warnings
PERF_WARN_FILE_PROCESSING_SEC = 5.0
"""Log warning if file processing takes > 5 seconds"""

PERF_WARN_AGENT_RUN_SEC = 30.0
"""Log warning if agent run takes > 30 seconds"""

# ============================================================================
# TESTING & DEVELOPMENT (§11.1)
# ============================================================================

# Looser thresholds for testing/development
TEST_MODE_ELITE_COMPOSITE = 80  # Lower than production (85)
"""Elite threshold for test mode (allows broader testing)"""

TEST_MODE_KEEP_MIN_COMPOSITE = 88  # Lower than production (93)
"""Keep candidate threshold for test mode"""

CACHE_ENABLED_DEFAULT = True
"""Whether to enable file caching by default"""

CACHE_DIR = ".cache"
"""Directory for caching expensive operations"""
