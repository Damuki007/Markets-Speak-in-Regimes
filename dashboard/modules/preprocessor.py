"""
Feature engineering / preprocessing for the SAFCOM dashboard.

Pure functions: each takes a DataFrame and returns a new one, never mutating a
global. That makes them trivial to unit-test and safe to chain.
"""
import numpy as np
import pandas as pd

# No business-day filtering is applied here: the SAFCOM source data is already a
# clean NSE trading calendar (0 weekends, 0 public holidays, and exchange closures
# already excluded by the provider — verified on SAFCOM_5YR_PRICE.xlsx). Re-filtering
# already-clean data with a civil-holiday list (holidays.Kenya) would be redundant
# and risks false-positive removals, because the civil calendar is NOT identical to
# the NSE trading calendar. If a future raw feed ever needs de-weekending, do that at
# the ingestion step (data/merge_data.py), weekend-only, not here.


def compute_rolling_vol(df: pd.DataFrame, windows=(21, 63, 252)) -> pd.DataFrame:
    """Add annualised rolling volatility columns (``vol_{w}d``).

    Daily std is scaled by sqrt(252) to annualise (there are ~252 trading days a
    year). 21d ~= 1 trading month, 63d ~= 1 quarter, 252d ~= 1 year.
    """
    df = df.copy()
    for w in windows:
        df[f"vol_{w}d"] = df["log_return"].rolling(w).std() * np.sqrt(252)
    return df


def tag_structural_breaks(df: pd.DataFrame) -> pd.DataFrame:
    """Label each row by Vodacom-takeover era.

    The June 2026 Vodacom takeover is a structural break: the return-generating
    process is not guaranteed to be stationary across it, so downstream analysis
    can slice on ``era`` to treat pre/post separately.
    """
    df = df.copy()
    df["era"] = "pre_announcement"
    df.loc[df["date"] >= pd.Timestamp("2025-12-04"), "era"] = "announcement_to_close"
    df.loc[df["date"] > pd.Timestamp("2026-06-30"), "era"] = "post_takeover"
    return df
