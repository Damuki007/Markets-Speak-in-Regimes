"""
Market-regime detection via Gaussian Mixture Model (GMM).

Jeff_project's model selection (BIC) landed on k=4 components — see
arch_output.txt "best k=4". We treat those 4 statistical components as the
ground-truth regimes and give them human-readable, volatility-ordered labels.
"""
import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.mixture import GaussianMixture

# Regimes are RE-ORDERED after fitting so index 0 = calmest, 3 = most volatile.
# GMM component numbering is arbitrary, so without this remap the colours/labels
# would jump around between runs.
REGIME_LABELS = {
    0: "Low Volatility / Accumulation",
    1: "Moderate Volatility / Trending",
    2: "High Volatility / Stress",
    3: "Extreme Volatility / Crisis",
}

REGIME_COLORS = {
    0: "#2ecc71",  # green  - calm
    1: "#3498db",  # blue   - normal
    2: "#e67e22",  # orange - stressed
    3: "#e74c3c",  # red    - crisis
}


def fit_gmm(features: np.ndarray, k: int = 4, random_state: int = 42) -> GaussianMixture:
    """Fit a k-component full-covariance GMM.

    ``n_init=5`` restarts the EM algorithm from 5 seeds and keeps the best
    likelihood — GMM is non-convex, so a single init can land in a poor local
    optimum. ``random_state`` fixes the result so the dashboard is reproducible.
    """
    gmm = GaussianMixture(
        n_components=k, covariance_type="full", random_state=random_state, n_init=5
    )
    gmm.fit(features)
    return gmm


def assign_regimes(df: pd.DataFrame, gmm: GaussianMixture, feature_cols=None) -> pd.DataFrame:
    """Predict each row's regime and attach ordered label/colour columns.

    Regimes are remapped so component 0 has the lowest mean ``abs_return``
    (calmest) and 3 the highest (most volatile), then joined back onto the
    original frame. Rows without complete features get regime -1 / "Unclassified".
    """
    if feature_cols is None:
        feature_cols = ["log_return", "abs_return"]

    feats = df[feature_cols].dropna()
    X = feats.values
    labels = gmm.predict(X)

    # rank raw components by mean absolute return (column index 1 == abs_return)
    regime_vol = {
        c: X[labels == c, 1].mean()
        for c in range(gmm.n_components)
        if (labels == c).sum() > 0
    }
    remap = {
        old: new
        for new, (old, _) in enumerate(sorted(regime_vol.items(), key=lambda kv: kv[1]))
    }

    df = df.copy()
    df.loc[feats.index, "regime"] = [remap.get(l, l) for l in labels]
    df["regime"] = df["regime"].fillna(-1).astype(int)
    df["regime_label"] = df["regime"].map(REGIME_LABELS).fillna("Unclassified")
    df["regime_color"] = df["regime"].map(REGIME_COLORS).fillna("#cccccc")
    return df


def detect_regime_episodes(df: pd.DataFrame, pen: float = 16.0, min_size: int = 20,
                           vol_col: str = "vol_21d", date_col: str = "date"):
    """Segment the series into a few clean *volatility episodes* for shaded bands.

    Per-day GMM labels flip too often to shade directly (they would be visual
    confetti). Instead we run Ruptures PELT change-point detection on the
    rolling volatility — the same PELT method the report cites — to find the
    handful of dates where the volatility level structurally shifts. Each
    resulting episode is then labelled with the GMM regime whose mean volatility
    is closest to the episode's mean, so the bands carry the 4-regime taxonomy.

    Returns a list of ``(start_date, end_date, regime, mean_ann_vol)`` spans that
    tile the timeline with no gaps (``end`` = start of the next episode).
    """
    vol = df[vol_col].bfill().to_numpy().reshape(-1, 1)
    z = (vol - vol.mean()) / (vol.std() or 1.0)
    bkps = rpt.Pelt(model="l2", min_size=min_size).fit(z).predict(pen=pen)

    # mean rolling vol per GMM regime → the "centre" of each regime band
    centers = df.groupby("regime")[vol_col].mean().to_dict()
    centers = {k: v for k, v in centers.items() if k >= 0 and pd.notna(v)}

    dates = df[date_col].to_numpy()
    n = len(df)
    episodes, prev = [], 0
    for b in bkps:
        seg_vol = float(df[vol_col].iloc[prev:b].mean())
        regime = min(centers, key=lambda k: abs(centers[k] - seg_vol)) if centers else -1
        end = dates[b] if b < n else dates[-1]
        episodes.append((dates[prev], end, int(regime), seg_vol))
        prev = b
    return episodes
