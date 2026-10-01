from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.base.tsa_model import ValueWarning

from .config import MeanSpec


def _range_indexed(series: pd.Series) -> pd.Series:
    values = pd.Series(series).dropna().astype(float).to_numpy()
    return pd.Series(values, index=pd.RangeIndex(len(values)), name="log_return")


def fit_arma(returns: pd.Series, spec: MeanSpec):
    """Fit ARMA(p,q) as ARIMA(p,0,q) using the state-space implementation."""

    r = _range_indexed(returns)
    model = ARIMA(
        r,
        order=(spec.p, 0, spec.q),
        trend=spec.trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(
        method="statespace",
        method_kwargs={"method": "lbfgs", "maxiter": 2000, "disp": False},
    )


def search_arma(
    returns: pd.Series,
    p_max: int = 5,
    q_max: int = 5,
    trends=("n", "c"),
    lags=(10, 20, 30),
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Grid-search ARMA models.

    Returns (all_models, white_residual_candidates), each sorted by BIC/AIC.
    Candidate models must pass Ljung-Box residual tests at every requested lag.
    """

    warnings.filterwarnings("ignore", category=ValueWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    r = _range_indexed(returns)
    rows: list[dict] = []

    for trend in trends:
        for p in range(p_max + 1):
            for q in range(q_max + 1):
                try:
                    result = fit_arma(r, MeanSpec(p, q, trend))
                    converged = bool(getattr(result, "mle_retvals", {}).get("converged", True))
                    if not converged:
                        continue

                    residuals = pd.Series(result.resid).dropna()
                    lb = acorr_ljungbox(residuals, lags=list(lags), return_df=True)
                    lb2 = acorr_ljungbox(residuals**2, lags=list(lags), return_df=True)

                    row = {
                        "trend": trend,
                        "p": p,
                        "q": q,
                        "aic": float(result.aic),
                        "bic": float(result.bic),
                    }
                    for lag in lags:
                        row[f"LB_p@{lag}"] = float(lb.loc[lag, "lb_pvalue"])
                        row[f"LB2_p@{lag}"] = float(lb2.loc[lag, "lb_pvalue"])
                    rows.append(row)
                except Exception:
                    continue

    all_models = pd.DataFrame(rows)
    if all_models.empty:
        raise RuntimeError("No ARMA specification converged.")

    all_models = all_models.sort_values(["bic", "aic", "trend", "p", "q"]).reset_index(drop=True)
    mask = np.ones(len(all_models), dtype=bool)
    for lag in lags:
        mask &= all_models[f"LB_p@{lag}"] > alpha

    candidates = all_models.loc[mask].reset_index(drop=True)
    return all_models, candidates


def aligned_fitted_and_residuals(returns: pd.Series, result) -> tuple[pd.Series, pd.Series]:
    """Map state-space fitted values/residuals back to the original return index."""

    r = pd.Series(returns).dropna().astype(float)
    fitted = pd.Series(np.asarray(result.fittedvalues), index=r.index, name="mu_hat")
    residuals = (r - fitted).rename("residual")
    return fitted, residuals
