"""
Data loading layer for the SAFCOM dashboard.

Keeping *loading* separate from *modelling* is a deliberate software-engineering
choice: the rest of the app never reads a file path directly, it just calls
``load_safcom()``. If the data source changes (CSV -> database -> live API), only
this one module changes and everything downstream keeps working.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# ``__file__`` is .../safcom_dashboard/modules/data_loader.py
# parents[1] climbs two levels up to .../safcom_dashboard, then into /data.
DATA_DIR = Path(__file__).parents[1] / "data"


def load_safcom() -> pd.DataFrame:
    """Load the cleaned SAFCOM price series and derive return columns.

    Returns
    -------
    pd.DataFrame
        Columns: ``date`` (datetime), ``close`` (float), ``volume`` (nullable),
        ``source``/``vodacom_era`` (from the merge step), plus computed
        ``log_return`` and ``abs_return``. Sorted ascending by date with the
        first (NaN-return) row dropped.
    """
    path = DATA_DIR / "safcom_base.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Build it first with:  python data/merge_data.py"
        )

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Recompute returns here (rather than trusting the CSV) so the loader is the
    # single source of truth for what a "return" means across the whole app.
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["abs_return"] = df["log_return"].abs()

    return df.dropna(subset=["log_return"]).reset_index(drop=True)


def load_garch_params() -> dict:
    """Reference GARCH(1,1)-t parameters from the Jeff_project analysis.

    These are the *published* values (arch_output.txt, N=2500). The dashboard
    fits GARCH live on the loaded data and shows its own numbers; this dict is
    kept only as a labelled benchmark to compare the live fit against.
    """
    return {
        "omega": 3.0122e-05,
        "alpha": 0.10,
        "beta": 0.80,
        "nu": 4.9853,
        "aic": -13996.8,
        "n_obs": 2500,
    }
