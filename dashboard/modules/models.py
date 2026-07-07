"""
Volatility & risk models for the SAFCOM dashboard.

Everything here is computed *live* on whatever data is loaded — no hardcoded
coefficients or violation counts. That keeps the dashboard honest as the series
is extended toward the present (e.g. after an EODHD pull).

Convention note: ``arch`` fits more stably when returns are expressed in percent,
so we fit on ``returns * 100`` and divide the resulting conditional volatility
back by 100 to return to plain log-return units. All series this module returns
are already back in log-return units.
"""
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import chi2


def fit_garch(returns: pd.Series, p: int = 1, q: int = 1, dist: str = "studentst") -> dict:
    """Fit a Zero-Mean GARCH(p, q) model with the given error distribution.

    Parameters
    ----------
    returns : pd.Series
        Daily log returns (plain units, NaNs already dropped).
    dist : str
        'studentst' (default, matches Jeff_project) or 'normal'.

    Returns
    -------
    dict
        ``params`` (incl. omega/alpha[1]/beta[1]/nu), ``aic``, ``bic``,
        ``persistence`` (alpha+beta), ``cond_vol`` (Series, log-return units,
        indexed like ``returns``) and ``std_resid`` (Series).
    """
    r = returns.dropna()
    model = arch_model(r * 100, vol="Garch", p=p, q=q, dist=dist, mean="Zero")
    res = model.fit(disp="off")

    params = res.params.to_dict()
    alpha = params.get("alpha[1]", 0.0)
    beta = params.get("beta[1]", 0.0)

    return {
        "params": params,
        "aic": float(res.aic),
        "bic": float(res.bic),
        "n_obs": int(r.shape[0]),
        "persistence": float(alpha + beta),
        # divide by 100 to undo the percent scaling above
        "cond_vol": res.conditional_volatility / 100.0,
        "std_resid": res.std_resid,
    }


def garch_var(cond_vol: pd.Series, alpha: float = 0.05, dist_nu: float | None = None) -> pd.Series:
    """One-day-ahead Value-at-Risk (positive loss magnitude) from GARCH vol.

    VaR_t = q_alpha * sigma_t, where q_alpha is the |quantile| of the innovation
    distribution. If ``dist_nu`` is given we use the Student-t quantile scaled to
    unit variance (matching the fitted GARCH-t); otherwise the Normal quantile.
    Returned as a positive series so a violation is ``return < -VaR``.
    """
    if dist_nu is not None and dist_nu > 2:
        from scipy.stats import t as student_t

        # scale so the t distribution has unit variance, matching sigma_t
        scale = np.sqrt((dist_nu - 2) / dist_nu)
        q = abs(student_t.ppf(alpha, df=dist_nu) * scale)
    else:
        from scipy.stats import norm

        q = abs(norm.ppf(alpha))
    return cond_vol * q


def kupiec_backtest(returns: pd.Series, var_series: pd.Series, alpha: float = 0.05) -> dict:
    """Kupiec unconditional-coverage (POF) test on a VaR series.

    Null hypothesis: the true violation rate equals ``alpha``. A small p-value
    (< 0.05) rejects the model — it is mis-calibrated (usually *underestimating*
    tail risk when the violation rate is too high).
    """
    df = pd.concat([returns.rename("r"), var_series.rename("var")], axis=1).dropna()
    n = int(df.shape[0])
    violations = int((df["r"] < -df["var"]).sum())
    p_hat = violations / n if n else 0.0

    # Likelihood-ratio POF statistic (guard the log(0) edge cases)
    if violations in (0, n):
        lr_pof = 0.0
    else:
        lr_pof = -2 * (
            (n - violations) * np.log(1 - alpha) + violations * np.log(alpha)
            - (n - violations) * np.log(1 - p_hat) - violations * np.log(p_hat)
        )
    p_value = float(1 - chi2.cdf(lr_pof, df=1))

    return {
        "n_obs": n,
        "violations": violations,
        "expected": n * alpha,
        "violation_rate": float(p_hat),
        "lr_stat": float(lr_pof),
        "p_value": p_value,
        "model_ok": p_value > 0.05,
    }
