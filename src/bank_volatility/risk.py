from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, t


def standardized_student_t_quantile(alpha: float, nu: float) -> float:
    """Quantile of a Student-t distribution standardized to unit variance."""

    if nu <= 2:
        raise ValueError("Student-t degrees of freedom must exceed 2 for finite variance.")
    return float(t.ppf(alpha, df=nu) * np.sqrt((nu - 2.0) / nu))


def conditional_var(
    conditional_mean: pd.Series,
    sigma_daily: pd.Series,
    volatility_result,
    confidence: float,
) -> pd.Series:
    """One-day parametric VaR as a positive loss threshold."""

    alpha = 1.0 - confidence
    params = volatility_result.params
    if "nu" in params:
        q = standardized_student_t_quantile(alpha, float(params["nu"]))
    else:
        q = float(norm.ppf(alpha))

    mu = pd.Series(conditional_mean).reindex(sigma_daily.index).fillna(0.0)
    var = -(mu + sigma_daily * q)
    return var.rename(f"VaR_{int(confidence * 100)}")


def kupiec_uc_test(hits: pd.Series, alpha: float) -> Tuple[float, float]:
    hits = pd.Series(hits).dropna().astype(int)
    n = len(hits)
    if n == 0:
        return np.nan, np.nan

    x = int(hits.sum())
    phat = np.clip(x / n, 1e-12, 1 - 1e-12)
    alpha = float(np.clip(alpha, 1e-12, 1 - 1e-12))

    ll0 = (n - x) * np.log(1 - alpha) + x * np.log(alpha)
    ll1 = (n - x) * np.log(1 - phat) + x * np.log(phat)
    lr = -2.0 * (ll0 - ll1)
    return float(lr), float(1 - chi2.cdf(lr, df=1))


def christoffersen_ind_test(hits: pd.Series) -> Tuple[float, float]:
    values = pd.Series(hits).dropna().astype(int).to_numpy()
    if len(values) < 2:
        return np.nan, np.nan

    previous, current = values[:-1], values[1:]
    n00 = int(((previous == 0) & (current == 0)).sum())
    n01 = int(((previous == 0) & (current == 1)).sum())
    n10 = int(((previous == 1) & (current == 0)).sum())
    n11 = int(((previous == 1) & (current == 1)).sum())

    p01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    p11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    p = (n01 + n11) / (n00 + n01 + n10 + n11)

    p01, p11, p = [float(np.clip(v, 1e-12, 1 - 1e-12)) for v in (p01, p11, p)]

    ll_ind = n00 * np.log(1 - p01) + n01 * np.log(p01) + n10 * np.log(1 - p11) + n11 * np.log(p11)
    ll_uc = (n00 + n10) * np.log(1 - p) + (n01 + n11) * np.log(p)
    lr = -2.0 * (ll_uc - ll_ind)
    return float(lr), float(1 - chi2.cdf(lr, df=1))


def christoffersen_cc_test(hits: pd.Series, alpha: float) -> Tuple[float, float]:
    lr_uc, _ = kupiec_uc_test(hits, alpha)
    lr_ind, _ = christoffersen_ind_test(hits)
    if np.isnan(lr_uc) or np.isnan(lr_ind):
        return np.nan, np.nan
    lr = lr_uc + lr_ind
    return float(lr), float(1 - chi2.cdf(lr, df=2))


def backtest_var(
    returns: pd.Series,
    var: pd.Series,
    confidence: float,
    test_window: int | None = 252,
) -> dict:
    returns = pd.Series(returns).dropna()
    var = pd.Series(var).reindex(returns.index)

    if test_window is not None and test_window < len(returns):
        returns = returns.iloc[-test_window:]
        var = var.reindex(returns.index)

    hits = (returns < -var).astype(int)
    alpha = 1.0 - confidence
    _, p_uc = kupiec_uc_test(hits, alpha)
    _, p_ind = christoffersen_ind_test(hits)
    _, p_cc = christoffersen_cc_test(hits, alpha)

    return {
        "level": confidence,
        "alpha": alpha,
        "obs_rate": float(hits.mean()),
        "expected": alpha,
        "n_obs": int(len(hits)),
        "n_hits": int(hits.sum()),
        "kupiec_p": p_uc,
        "christ_ind_p": p_ind,
        "christ_cc_p": p_cc,
    }
