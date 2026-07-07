"""
Build the dashboard's base dataset: clean the 5-yr XLSX and (optionally) append
an EODHD extension, then write ``safcom_base.csv``.

Run:  python data/merge_data.py   (from the safcom_dashboard/ folder)

This is the ONE place raw sources are touched. Everything else in the app reads
the tidy ``safcom_base.csv`` this produces.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent                 # .../dashboard/data

# Resolve the raw dataset across likely layouts (repo /data, alongside this script,
# or the original dev tree) so the rebuild works wherever the project is checked out.
_XLSX_CANDIDATES = [
    HERE.parents[2] / "data" / "SAFCOM_5YR_PRICE.xlsx",          # repo root /data
    HERE / "SAFCOM_5YR_PRICE.xlsx",                              # alongside this script
    HERE.parents[1] / "Jeff_project" / "SAFCOM_5YR_PRICE.xlsx",  # original dev layout
]
XLSX_PATH = next((p for p in _XLSX_CANDIDATES if p.exists()), _XLSX_CANDIDATES[0])
EODHD_PATH = HERE / "safcom_eodhd_2025_2026.csv"        # optional, produced locally
OUTPUT_PATH = HERE / "safcom_base.csv"

# Vodacom structural-break dates (see modules/preprocessor.tag_structural_breaks)
VODACOM_ANNOUNCE = pd.Timestamp("2025-12-04")
VODACOM_CLOSE = pd.Timestamp("2026-06-30")


def load_base() -> pd.DataFrame:
    """Read and normalise the 5-yr XLSX (Date / Last Price / Volume)."""
    base = pd.read_excel(XLSX_PATH)
    base.columns = [c.strip().lower().replace(" ", "_") for c in base.columns]
    base = base.rename(columns={"last_price": "close"})
    base["date"] = pd.to_datetime(base["date"])
    base["source"] = "xlsx"
    # XLSX ships newest-first; sort ascending so returns line up chronologically.
    base = base.sort_values("date").reset_index(drop=True)
    keep = [c for c in ["date", "close", "volume", "source"] if c in base.columns]
    return base[keep]


def append_eodhd(base: pd.DataFrame) -> pd.DataFrame:
    """Concatenate any EODHD rows newer than the XLSX (if the file exists)."""
    if not EODHD_PATH.exists():
        print("No EODHD extension found — using base XLSX only.")
        return base

    ext = pd.read_csv(EODHD_PATH, parse_dates=["date"])
    ext = ext[["date", "close", "volume"]].copy()
    ext["source"] = "eodhd"
    ext_new = ext[ext["date"] > base["date"].max()].copy()
    print(f"Extension: {len(ext_new)} new rows appended from EODHD.")
    return pd.concat([base, ext_new], ignore_index=True)


def main() -> None:
    base = load_base()
    print(f"Base data: {len(base)} rows, {base['date'].min().date()} -> {base['date'].max().date()}")

    merged = append_eodhd(base).sort_values("date").reset_index(drop=True)

    merged["log_return"] = np.log(merged["close"] / merged["close"].shift(1))

    merged["vodacom_era"] = "pre_announcement"
    merged.loc[merged["date"] >= VODACOM_ANNOUNCE, "vodacom_era"] = "announcement_to_close"
    merged.loc[merged["date"] > VODACOM_CLOSE, "vodacom_era"] = "post_takeover"

    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"\nMerged dataset: {len(merged)} rows -> {OUTPUT_PATH}")
    print(merged["vodacom_era"].value_counts().to_string())


if __name__ == "__main__":
    main()
