"""
Research Agent Tool: File Lister
Discovers and classifies available data files in the data directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from crewai.tools import BaseTool
except ImportError:
    from pydantic import BaseModel as BaseTool
from pydantic import BaseModel, Field


# Expected files per spec §4.2
EXPECTED_FILES = {
    "ibd_xls": [
        "BIG_CAP_20_1.xls",
        "IBD_50_3.xls",
        "SECTOR_LEADERS_1.xls",
        "STOCK_SPOTLIGHT.xls",
        "IPO_LEADERS.xls",
        "RISING_PROFIT_ESTIMATES.xls",
        "RELATIVE_STRENGTH_AT_NEW_HIGH.xls",
    ],
    "ibd_pdf": [
        "IBD_Smart_NYSE___Nasdaq_Tables_--_Jan_09__2026___Investor_s_Business_Daily.pdf",
        "IBD_Tech_Leaders__Jan_09_2026___Investors_Business_Daily.pdf",
        "ETF_Leaders__Jan_09_2026___Investors_Business_Daily.pdf",
        "Top_Sector_ETFs__Jan_09_2026___Investors_Business_Daily.pdf",
        "ETF_Tables__Jan_09_2026___Investors_Business_Daily.pdf",
        "Top_200_Composite_Stocks__Jan_09_2026___Investors_Business_Daily.pdf",
        "IBD_Large_MidCap_Leaders_Index__Jan_09_2026___Investors_Business_Daily.pdf",
        "TopRanked_LowPriced_Stocks__Jan_09_2026___Investors_Business_Daily.pdf",
        "NYSE__Nasdaq_Most_Active_and_Most__Up__Jan_09_2026___Investors_Business_Daily.pdf",
        "TimeSaver_Table__Jan_09_2026___Investors_Business_Daily.pdf",
    ],
    "fool_pdf": [
        "Rankings_Epic___The_Motley_Fool.pdf",
        "New_Recs___Epic___The_Motley_Fool.pdf",
    ],
    "schwab_themes": [
        "Schwab_Investing_Themes___Charles_Schwab.pdf",
        "Schwab_AI.pdf",
        "Schwab_RR.pdf",
        "Schwab_AD.pdf",
        "Schwab_RenEn.pdf",
        "Schwab_CC.pdf",
        "SchwabSpaceEc.pdf",
        "Schwab_CS.pdf",
        "Schwab_BD.pdf",
        "Schwab3d.pdf",
        "Schwab_DLL.pdf",
        "Schwab_Ecom.pdf",
        "Schwab_PM.pdf",
        "Schwab_CBT.pdf",
        "SchwabAging.pdf",
        "Schwab_HAE.pdf",
        "Schwab_China.pdf",
        "Schwab_Can.pdf",
        "SchwabDef.pdf",
        "SchwabBC.pdf",
        "Schwab_OMV.pdf",
    ],
    "morningstar_pdf": [
        "Morningstar_US_Large_Cap_Pick_List.pdf",
        "Morningstar_US_Mid_Cap_Pick_List.pdf",
    ],
}


@dataclass
class FileInfo:
    """Info about a discovered file."""
    path: str
    filename: str
    category: str  # ibd_xls, ibd_pdf, fool_pdf, schwab_themes, morningstar_pdf, unknown
    size_bytes: int
    exists: bool


@dataclass
class FileListResult:
    """Result of scanning the data directory."""
    found: list[FileInfo]
    missing: list[str]  # Expected but not found
    extra: list[FileInfo]  # Found but not expected
    total_expected: int
    total_found: int
    coverage_pct: float


def classify_file(filename: str) -> str:
    """Classify a file into a category based on its name."""
    for category, expected in EXPECTED_FILES.items():
        if filename in expected:
            return category

    # Heuristic fallback
    lower = filename.lower()
    if lower.endswith(".xls") or lower.endswith(".xlsx") or lower.endswith(".csv"):
        return "ibd_xls"
    if "schwab" in lower:
        return "schwab_themes"
    if "motley" in lower or "fool" in lower:
        return "fool_pdf"
    if "morningstar" in lower:
        return "morningstar_pdf"
    if lower.endswith(".pdf"):
        return "ibd_pdf"
    return "unknown"


def list_data_files(data_dir: str = "data") -> FileListResult:
    """
    Scan the data directory and return found/missing/extra files.

    Args:
        data_dir: Root data directory path

    Returns:
        FileListResult with complete inventory
    """
    data_path = Path(data_dir)

    # Collect all expected files
    all_expected: dict[str, str] = {}  # filename → category
    for category, filenames in EXPECTED_FILES.items():
        for fn in filenames:
            all_expected[fn] = category

    found: list[FileInfo] = []
    found_names: set[str] = set()
    extra: list[FileInfo] = []

    # Walk subdirectories
    for category_dir in ["ibd_xls", "ibd_pdf", "fool_pdf", "schwab_themes", "morningstar_pdf"]:
        dir_path = data_path / category_dir
        if not dir_path.exists():
            continue
        for entry in dir_path.iterdir():
            if entry.is_file():
                fi = FileInfo(
                    path=str(entry),
                    filename=entry.name,
                    category=classify_file(entry.name),
                    size_bytes=entry.stat().st_size,
                    exists=True,
                )
                if entry.name in all_expected:
                    found.append(fi)
                    found_names.add(entry.name)
                else:
                    extra.append(fi)

    # Also check root data dir
    if data_path.exists():
        for entry in data_path.iterdir():
            if entry.is_file() and entry.name not in found_names:
                fi = FileInfo(
                    path=str(entry),
                    filename=entry.name,
                    category=classify_file(entry.name),
                    size_bytes=entry.stat().st_size,
                    exists=True,
                )
                if entry.name in all_expected:
                    found.append(fi)
                    found_names.add(entry.name)
                else:
                    extra.append(fi)

    # Find missing
    missing = [fn for fn in all_expected if fn not in found_names]

    total_expected = len(all_expected)
    total_found = len(found)
    coverage = (total_found / total_expected * 100) if total_expected > 0 else 0.0

    return FileListResult(
        found=found,
        missing=missing,
        extra=extra,
        total_expected=total_expected,
        total_found=total_found,
        coverage_pct=round(coverage, 1),
    )


class ListDataFilesInput(BaseModel):
    """Input for ListDataFilesTool."""
    data_dir: str = Field(default="data", description="Root data directory path")


class ListDataFilesTool(BaseTool):
    """Discover and classify available data files."""
    name: str = "list_data_files"
    description: str = (
        "Scan the data directory to discover available IBD, Schwab, and Motley Fool "
        "data files. Returns found files, missing files, and coverage percentage."
    )
    args_schema: type[BaseModel] = ListDataFilesInput

    def _run(self, data_dir: str = "data") -> str:
        result = list_data_files(data_dir)
        lines = [
            f"=== Data File Inventory ===",
            f"Found: {result.total_found}/{result.total_expected} ({result.coverage_pct}%)",
            "",
        ]
        if result.found:
            lines.append("FOUND:")
            for fi in sorted(result.found, key=lambda x: x.category):
                lines.append(f"  [{fi.category}] {fi.filename} ({fi.size_bytes:,} bytes)")
        if result.missing:
            lines.append(f"\nMISSING ({len(result.missing)}):")
            for fn in result.missing:
                lines.append(f"  ❌ {fn}")
        if result.extra:
            lines.append(f"\nEXTRA ({len(result.extra)}):")
            for fi in result.extra:
                lines.append(f"  ➕ [{fi.category}] {fi.filename}")
        return "\n".join(lines)
