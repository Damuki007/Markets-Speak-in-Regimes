"""
Safaricom (SCOM.NSE) Risk Dashboard
Run:  streamlit run app.py

Design principle: every headline number is computed LIVE from the loaded data
(GARCH fit, GMM regimes, VaR + Kupiec backtest). Nothing is hardcoded, so the
dashboard stays truthful when the series is extended toward the present.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
from modules.data_loader import load_safcom, load_garch_params
from modules.preprocessor import compute_rolling_vol, tag_structural_breaks
from modules.models import fit_garch, garch_var, kupiec_backtest
from modules.regimes import (
    fit_gmm, assign_regimes, detect_regime_episodes, REGIME_LABELS, REGIME_COLORS,
)

st.set_page_config(
    page_title="SAFCOM Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

VODACOM_EVENTS = [
    ("2025-12-04", "Vodacom Deal Announced", "dash"),
    ("2026-06-30", "Deal Closed (55% stake)", "dot"),
]


def add_event_vlines(fig, enabled: bool = True, **kwargs) -> None:
    """Draw the Vodacom event marker lines on a figure.

    plotly 6.x quirk: ``add_vline`` with an annotation averages the line's x
    position and cannot sum a string/Timestamp, so ``x="2025-12-04"`` throws.
    Passing x as epoch-milliseconds (a plain number) sidesteps that — plotly
    interprets numeric x on a datetime axis as ms-since-epoch, so it still lands
    on the right date.
    """
    if not enabled:
        return
    for date, label, dash in VODACOM_EVENTS:
        x_ms = pd.Timestamp(date).timestamp() * 1000
        fig.add_vline(
            x=x_ms, line_dash=dash, line_color="purple", line_width=1.5,
            annotation_text=label, annotation_position="top left", **kwargs,
        )


# ── Cached compute layer ─────────────────────────────────────────────────────
# @st.cache_data memoises on the function inputs. We prefix DataFrame args with
# "_" so Streamlit skips trying to hash them (a DataFrame isn't cheaply hashable)
# and instead keys on the other args — fine here because the data is static per
# session.
@st.cache_data
def get_data() -> pd.DataFrame:
    df = load_safcom()
    df = compute_rolling_vol(df)
    df = tag_structural_breaks(df)
    return df


@st.cache_data
def get_regimes(_df: pd.DataFrame) -> pd.DataFrame:
    feats = _df[["log_return", "abs_return"]].dropna()
    gmm = fit_gmm(feats.values, k=4)
    return assign_regimes(_df, gmm)


@st.cache_data
def get_garch(_df: pd.DataFrame) -> dict:
    return fit_garch(_df["log_return"])


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("Settings")
show_vodacom = st.sidebar.checkbox("Highlight Vodacom Events", value=True)
confidence_level = st.sidebar.slider("VaR Confidence Level", 0.90, 0.99, 0.95, 0.01)
alpha = 1 - confidence_level
regime_pen = st.sidebar.slider(
    "Regime episode sensitivity", 6, 40, 16, 1,
    help="Ruptures PELT penalty for the shaded regime bands (Tab 3). "
    "Lower = more, shorter episodes; higher = fewer, broader ones.",
)

df = get_data()
df = get_regimes(df)
garch = get_garch(df)

# GARCH conditional volatility, aligned to df's rows (fit uses the same series)
df = df.copy()
df["garch_cond_vol"] = garch["cond_vol"].values
nu = garch["params"].get("nu")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Overview", "🌊 Volatility", "🎯 Regimes", "⚠️ Risk / VaR"]
)

# ── TAB 1: Overview ──────────────────────────────────────────────────────────
with tab1:
    st.header("Safaricom (SCOM.NSE) — Price & Returns Overview")

    latest_close = df["close"].iloc[-1]
    latest_date = df["date"].iloc[-1]
    low_close = df["close"].min()
    low_date = df.loc[df["close"].idxmin(), "date"]
    pct_from_low = (latest_close / low_close - 1) * 100
    ann_vol = df["log_return"].std() * np.sqrt(252)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Latest Close (in data)",
        f"KES {latest_close:.2f}",
        f"+{pct_from_low:.0f}% from {low_date:%b %Y} low (KES {low_close:.2f})",
    )
    c2.metric("Data Period", f"{df['date'].min():%b %Y} – {latest_date:%b %Y}")
    c3.metric("Trading Days", f"{len(df):,}")
    c4.metric("Full-Period Ann. Vol", f"{ann_vol:.1%}")
    st.caption(
        f"Latest close is the most recent row in the loaded data ({latest_date:%Y-%m-%d}). "
        "Run data/fetch_eodhd.py locally + data/merge_data.py to extend toward the present."
    )

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Closing Price (KES)", "Daily Log Return"),
        row_heights=[0.65, 0.35], vertical_spacing=0.08,
    )
    fig.add_trace(
        go.Scatter(x=df["date"], y=df["close"], line=dict(color="#2c3e50", width=1.5), name="Price"),
        row=1, col=1,
    )
    colors = np.where(df["log_return"] >= 0, "#2ecc71", "#e74c3c")
    fig.add_trace(
        go.Bar(x=df["date"], y=df["log_return"], marker_color=colors, name="Return"),
        row=2, col=1,
    )
    add_event_vlines(fig, show_vodacom, row=1, col=1)
    fig.update_yaxes(tickformat=".2%", row=2, col=1)
    fig.update_layout(height=580, showlegend=False, margin=dict(t=40))
    st.plotly_chart(fig, width="stretch")

# ── TAB 2: Volatility ────────────────────────────────────────────────────────
with tab2:
    st.header("Volatility — GARCH(1,1) Student-t (fitted live)")

    ref = load_garch_params()
    a = garch["params"].get("alpha[1]", 0.0)
    b = garch["params"].get("beta[1]", 0.0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("α (ARCH)", f"{a:.2f}", help=f"Short-run shock sensitivity. Reference: {ref['alpha']:.2f}")
    c2.metric("β (GARCH)", f"{b:.2f}", help=f"Volatility persistence. Reference: {ref['beta']:.2f}")
    c3.metric("α + β", f"{garch['persistence']:.2f}", help="Near-integrated (shocks decay slowly) if > 0.95")
    c4.metric("ν (tail dof)", f"{nu:.2f}" if nu else "n/a", help=f"Lower ⇒ heavier tails. Reference: {ref['nu']:.2f}")

    st.info(
        f"GARCH(1,1)-t fitted live on N={garch['n_obs']:,} obs (AIC={garch['aic']:,.1f}, "
        f"BIC={garch['bic']:,.1f}, %-scaled fit). "
        f"α+β={garch['persistence']:.2f} → {'high' if garch['persistence'] > 0.9 else 'moderate'} "
        f"volatility persistence; ν≈{nu:.2f} → fat tails. "
        f"Reference (Jeff_project, N={ref['n_obs']:,}): α={ref['alpha']}, β={ref['beta']}, ν≈{ref['nu']:.2f}."
    )

    fig2 = go.Figure()
    palette = {"vol_21d": ("#3498db", "21-day"), "vol_63d": ("#e67e22", "63-day"), "vol_252d": ("#e74c3c", "252-day")}
    for col, (color, label) in palette.items():
        if col in df.columns:
            fig2.add_trace(go.Scatter(x=df["date"], y=df[col], name=f"{label} Vol", line=dict(color=color, width=1.5)))
    # live GARCH conditional volatility, annualised for comparability with rolling vols
    fig2.add_trace(go.Scatter(
        x=df["date"], y=df["garch_cond_vol"] * np.sqrt(252),
        name="GARCH cond. vol", line=dict(color="#8e44ad", width=1.2, dash="dot"),
    ))
    add_event_vlines(fig2, show_vodacom)
    fig2.update_layout(title="Annualised Rolling & GARCH Volatility", height=430, yaxis_tickformat=".0%", legend=dict(x=0.01, y=0.99))
    st.plotly_chart(fig2, width="stretch")

# ── TAB 3: Regimes ───────────────────────────────────────────────────────────
with tab3:
    st.header("Market Regimes — GMM (k=4) + PELT episodes")
    st.caption(
        "4-component Gaussian Mixture on [log_return, |log_return|], fitted live. "
        "The chart shades the series into volatility episodes via Ruptures PELT "
        "change-point detection, each coloured by its nearest GMM regime."
    )

    regime_counts = df["regime_label"].value_counts()
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.subheader("Regime Distribution")
        for r in range(4):
            label = REGIME_LABELS[r]
            count = int(regime_counts.get(label, 0))
            pct = count / len(df) * 100
            st.markdown(
                f"<span style='color:{REGIME_COLORS[r]}; font-size:18px'>■</span> "
                f"**{label}**  {count:,} days ({pct:.1f}%)",
                unsafe_allow_html=True,
            )
        last_row = df.iloc[-1]
        st.markdown("---")
        st.subheader("Current Signal")
        st.markdown(
            f"<span style='color:{last_row['regime_color']}; font-size:22px'>●</span> "
            f"**{last_row['regime_label']}**",
            unsafe_allow_html=True,
        )
        st.caption(f"As of {df['date'].iloc[-1]:%Y-%m-%d}")

    with col_b:
        episodes = detect_regime_episodes(df, pen=regime_pen)
        fig3 = go.Figure()
        # shaded regime episode bands (below the price line)
        for start, end, reg, _mv in episodes:
            if reg < 0:
                continue
            fig3.add_vrect(
                x0=pd.Timestamp(start).timestamp() * 1000,
                x1=pd.Timestamp(end).timestamp() * 1000,
                fillcolor=REGIME_COLORS[reg], opacity=0.22, line_width=0, layer="below",
            )
        # price line on top of the bands
        fig3.add_trace(go.Scatter(
            x=df["date"], y=df["close"], mode="lines",
            line=dict(color="#1B2C5B", width=1.6), name="SCOM price",
        ))
        # legend proxies so the regime colours get a key
        for r in range(4):
            fig3.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=11, color=REGIME_COLORS[r], symbol="square"),
                name=REGIME_LABELS[r],
            ))
        add_event_vlines(fig3, show_vodacom)
        # bracket the post-takeover era once data extends past the close — this is
        # the out-of-sample regime the thesis predicts should be detectable
        post = df[df["date"] > pd.Timestamp("2026-06-30")]
        if not post.empty:
            fig3.add_vrect(
                x0=pd.Timestamp("2026-06-30").timestamp() * 1000,
                x1=pd.Timestamp(post["date"].max()).timestamp() * 1000,
                line_width=1.5, line_color="#8e44ad", line_dash="dot",
                fillcolor="rgba(0,0,0,0)",
                annotation_text="Post-takeover (out-of-sample)",
                annotation_position="top left",
            )
        fig3.update_layout(
            title="SCOM Price — Shaded by Volatility Regime (PELT episodes)",
            height=440, legend=dict(orientation="h", y=-0.18, x=0),
        )
        st.plotly_chart(fig3, width="stretch")
        if df["date"].max() < pd.Timestamp("2025-12-04"):
            st.info(
                f"Data ends {df['date'].max():%b %Y} — **before** the Vodacom announcement, so "
                "no post-takeover regime is present yet. Run `data/fetch_eodhd.py` then "
                "`data/merge_data.py` locally to extend past Dec 2025; the post-takeover episode "
                "will then shade and annotate automatically — the out-of-sample validation."
            )

# ── TAB 4: Risk / VaR ────────────────────────────────────────────────────────
with tab4:
    st.header("Risk — Value at Risk (VaR) & Kupiec Backtest")
    st.caption(
        f"GARCH-t VaR at {confidence_level:.0%} confidence, backtested LIVE with the "
        "Kupiec unconditional-coverage test on the loaded data."
    )

    # Live GARCH-t VaR (positive loss magnitude) + two comparison VaR lines:
    #   var_garch — GARCH-t conditional VaR (what we backtest)
    #   var_param — Normal parametric VaR off 21d rolling vol
    #   var_hist  — empirical 252d rolling historical VaR
    df["var_garch"] = garch_var(df["garch_cond_vol"], alpha=alpha, dist_nu=nu)
    df["var_param"] = df["vol_21d"] / np.sqrt(252) * abs(norm.ppf(alpha))
    df["var_hist"] = df["log_return"].rolling(252).quantile(alpha).abs()

    # Live Kupiec backtest on the GARCH-t VaR
    kp = kupiec_backtest(df["log_return"], df["var_garch"], alpha=alpha)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Expected Violations ({alpha:.0%})", f"{kp['expected']:.0f}")
    c2.metric(
        "Actual Violations", f"{kp['violations']}",
        delta=f"{kp['violations'] - kp['expected']:+.0f}", delta_color="inverse",
    )
    c3.metric(
        "Violation Rate", f"{kp['violation_rate']:.1%}",
        delta=f"{kp['violation_rate'] - alpha:+.1%} vs target", delta_color="inverse",
    )

    verdict = "PASSES" if kp["model_ok"] else "FAILS"
    msg = (
        f"**Kupiec POF test {verdict}** (LR={kp['lr_stat']:.2f}, p={kp['p_value']:.3f}). "
        f"{kp['violations']} violations vs {kp['expected']:.0f} expected "
        f"({kp['violation_rate']:.1%} vs {alpha:.0%} target) over N={kp['n_obs']:,}. "
    )
    if kp["model_ok"]:
        st.success(msg + "Coverage is statistically consistent with the target — VaR is well-calibrated at this level.")
    else:
        st.warning(
            msg + "The GARCH(1,1)-t VaR is **mis-calibrated** — it underestimates tail risk. "
            "Recommended upgrade: **GJR-GARCH** (asymmetric/leverage vol) + **EVT** tail correction, "
            "and do NOT use current VaR for production position sizing."
        )

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df["date"], y=df["log_return"], name="Daily Return", line=dict(color="#7f8c8d", width=0.7), opacity=0.8))
    fig4.add_trace(go.Scatter(x=df["date"], y=-df["var_garch"], name=f"GARCH-t VaR ({confidence_level:.0%})", line=dict(color="#8e44ad", width=1.5)))
    fig4.add_trace(go.Scatter(x=df["date"], y=-df["var_param"], name=f"Param VaR ({confidence_level:.0%})", line=dict(color="#e74c3c", dash="dash", width=1.2)))
    fig4.add_trace(go.Scatter(x=df["date"], y=-df["var_hist"], name=f"Hist VaR ({confidence_level:.0%})", line=dict(color="#e67e22", dash="dot", width=1.2)))
    # mark the actual violations
    viol = df[df["log_return"] < -df["var_garch"]]
    fig4.add_trace(go.Scatter(x=viol["date"], y=viol["log_return"], mode="markers", name="Violation", marker=dict(color="#c0392b", size=5, symbol="x")))
    add_event_vlines(fig4, show_vodacom)
    fig4.update_layout(title=f"Returns vs VaR ({confidence_level:.0%} Confidence)", height=450, yaxis_tickformat=".2%", legend=dict(x=0.01, y=0.01))
    st.plotly_chart(fig4, width="stretch")
