"""
Shared test fixtures for Research Agent tests.
Provides mock data, sample XLS files, and golden dataset helpers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent


_IBD_COLUMN_MAP = {
    "symbol": "Symbol",
    "company_name": "Company Name",
    "composite_rating": "Composite Rating",
    "rs_rating": "RS Rating",
    "eps_rating": "EPS Rating",
    "smr_rating": "SMR Rating",
    "acc_dis_rating": "Acc/Dis Rating",
    "sector": "Sector",
}


def create_mock_xls(filepath: str, stocks: list[dict]) -> None:
    """Create a mock IBD XLS file with stock data."""
    df = pd.DataFrame(stocks)
    df = df.rename(columns={k: v for k, v in _IBD_COLUMN_MAP.items() if k in df.columns})
    df.to_excel(filepath, index=False, engine="openpyxl")


def create_mock_csv(filepath: str, stocks: list[dict]) -> None:
    """Create a mock IBD CSV file with stock data."""
    df = pd.DataFrame(stocks)
    df = df.rename(columns={k: v for k, v in _IBD_COLUMN_MAP.items() if k in df.columns})
    df.to_csv(filepath, index=False)


# Sample stocks for testing (real IBD data from spec §3.2)
SAMPLE_IBD_STOCKS = [
    {"symbol": "MU", "company_name": "Micron Technology", "composite_rating": 99, "rs_rating": 99, "eps_rating": 81, "smr_rating": "A", "acc_dis_rating": "B", "sector": "CHIPS"},
    {"symbol": "CLS", "company_name": "Celestica", "composite_rating": 99, "rs_rating": 97, "eps_rating": 99, "smr_rating": "A", "acc_dis_rating": "A", "sector": "COMPUTER"},
    {"symbol": "FIX", "company_name": "Comfort Systems", "composite_rating": 99, "rs_rating": 96, "eps_rating": 99, "smr_rating": "A", "acc_dis_rating": "B", "sector": "BUILDING"},
    {"symbol": "KGC", "company_name": "Kinross Gold", "composite_rating": 99, "rs_rating": 96, "eps_rating": 96, "smr_rating": "A", "acc_dis_rating": "A", "sector": "MINING"},
    {"symbol": "NEM", "company_name": "Newmont", "composite_rating": 99, "rs_rating": 95, "eps_rating": 91, "smr_rating": "B", "acc_dis_rating": "A", "sector": "MINING"},
    {"symbol": "GOOGL", "company_name": "Alphabet", "composite_rating": 99, "rs_rating": 94, "eps_rating": 92, "smr_rating": "A", "acc_dis_rating": "A", "sector": "INTERNET"},
    {"symbol": "PLTR", "company_name": "Palantir", "composite_rating": 99, "rs_rating": 93, "eps_rating": 99, "smr_rating": "A", "acc_dis_rating": "B", "sector": "SOFTWARE"},
    {"symbol": "AEM", "company_name": "Agnico Eagle", "composite_rating": 99, "rs_rating": 92, "eps_rating": 98, "smr_rating": "A", "acc_dis_rating": "A", "sector": "MINING"},
    {"symbol": "AGI", "company_name": "Alamos Gold", "composite_rating": 99, "rs_rating": 92, "eps_rating": 97, "smr_rating": "A", "acc_dis_rating": "B", "sector": "MINING"},
    {"symbol": "TFPM", "company_name": "Triple Flag Precious", "composite_rating": 99, "rs_rating": 92, "eps_rating": 99, "smr_rating": "A", "acc_dis_rating": "A", "sector": "MINING"},
    {"symbol": "RGLD", "company_name": "Royal Gold", "composite_rating": 99, "rs_rating": 90, "eps_rating": 95, "smr_rating": "A", "acc_dis_rating": "A", "sector": "MINING"},
    {"symbol": "GE", "company_name": "GE Aerospace", "composite_rating": 99, "rs_rating": 89, "eps_rating": 98, "smr_rating": "A", "acc_dis_rating": "B", "sector": "AEROSPACE"},
    {"symbol": "LLY", "company_name": "Eli Lilly", "composite_rating": 99, "rs_rating": 86, "eps_rating": 98, "smr_rating": "A", "acc_dis_rating": "A", "sector": "MEDICAL"},
    {"symbol": "ISRG", "company_name": "Intuitive Surgical", "composite_rating": 99, "rs_rating": 80, "eps_rating": 96, "smr_rating": "A", "acc_dis_rating": "B", "sector": "MEDICAL"},
    {"symbol": "APP", "company_name": "AppLovin", "composite_rating": 98, "rs_rating": 95, "eps_rating": 80, "smr_rating": "A", "acc_dis_rating": "A", "sector": "SOFTWARE"},
    {"symbol": "SCCO", "company_name": "Southern Copper", "composite_rating": 98, "rs_rating": 94, "eps_rating": 79, "smr_rating": "B", "acc_dis_rating": "A", "sector": "MINING"},
    {"symbol": "GMED", "company_name": "Globus Medical", "composite_rating": 98, "rs_rating": 91, "eps_rating": 95, "smr_rating": "A", "acc_dis_rating": "B", "sector": "MEDICAL"},
    {"symbol": "HWM", "company_name": "Howmet Aerospace", "composite_rating": 98, "rs_rating": 90, "eps_rating": 99, "smr_rating": "A", "acc_dis_rating": "A", "sector": "AEROSPACE"},
    {"symbol": "AVGO", "company_name": "Broadcom", "composite_rating": 98, "rs_rating": 88, "eps_rating": 99, "smr_rating": "A", "acc_dis_rating": "A", "sector": "CHIPS"},
    {"symbol": "NVDA", "company_name": "NVIDIA", "composite_rating": 98, "rs_rating": 81, "eps_rating": 99, "smr_rating": "A", "acc_dis_rating": "B", "sector": "CHIPS"},
    {"symbol": "KLAC", "company_name": "KLA Corp", "composite_rating": 97, "rs_rating": 94, "eps_rating": 96, "smr_rating": "A", "acc_dis_rating": "B", "sector": "CHIPS"},
    {"symbol": "TPR", "company_name": "Tapestry", "composite_rating": 97, "rs_rating": 93, "eps_rating": 94, "smr_rating": "B", "acc_dis_rating": "A", "sector": "CONSUMER"},
    {"symbol": "GS", "company_name": "Goldman Sachs", "composite_rating": 93, "rs_rating": 91, "eps_rating": 76, "smr_rating": "B", "acc_dis_rating": "B", "sector": "FINANCE"},
]

# Lower-rated stocks to pad universe
PADDING_STOCKS = [
    {"symbol": f"PAD{i:02d}", "company_name": f"Padding Co {i}",
     "composite_rating": 70 + (i % 20), "rs_rating": 65 + (i % 25),
     "eps_rating": 60 + (i % 30), "smr_rating": "C", "acc_dis_rating": "C",
     "sector": ["CHIPS", "SOFTWARE", "MEDICAL", "MINING", "FINANCE",
                "AEROSPACE", "INTERNET", "CONSUMER", "BUILDING", "ENERGY"][i % 10]}
    for i in range(40)
]


@pytest.fixture
def mock_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with mock XLS files."""
    # Create subdirectories
    xls_dir = tmp_path / "ibd_xls"
    xls_dir.mkdir()

    # Big Cap 20 — top 20 stocks
    create_mock_xls(
        str(xls_dir / "BIG_CAP_20_1.xls"),
        SAMPLE_IBD_STOCKS[:20],
    )

    # Sector Leaders — subset
    create_mock_xls(
        str(xls_dir / "SECTOR_LEADERS_1.xls"),
        [s for s in SAMPLE_IBD_STOCKS if s["composite_rating"] >= 99],
    )

    # IBD 50 — broader list
    create_mock_xls(
        str(xls_dir / "IBD_50_3.xls"),
        SAMPLE_IBD_STOCKS + PADDING_STOCKS[:27],
    )

    # Rising Profit Estimates
    create_mock_xls(
        str(xls_dir / "RISING_PROFIT_ESTIMATES.xls"),
        [s for s in SAMPLE_IBD_STOCKS if s.get("eps_rating", 0) >= 95],
    )

    return tmp_path


@pytest.fixture
def sample_stocks() -> list[dict]:
    """Return sample IBD stock data."""
    return SAMPLE_IBD_STOCKS.copy()
