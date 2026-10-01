from __future__ import annotations

import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller


def adf_test(series: pd.Series, regression: str = "c") -> dict:
    values = pd.Series(series).dropna().astype(float)
    statistic, pvalue, used_lag, nobs, critical_values, icbest = adfuller(
        values, regression=regression, autolag="AIC"
    )
    return {
        "statistic": float(statistic),
        "pvalue": float(pvalue),
        "used_lag": int(used_lag),
        "nobs": int(nobs),
        "critical_values": {k: float(v) for k, v in critical_values.items()},
        "aic_best": float(icbest),
    }


def ljung_box(series: pd.Series, lags=(10, 20, 30)) -> pd.DataFrame:
    return acorr_ljungbox(pd.Series(series).dropna(), lags=list(lags), return_df=True)


def arch_lm(series: pd.Series, nlags: int = 10) -> dict:
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(pd.Series(series).dropna(), nlags=nlags)
    return {
        "lm_statistic": float(lm_stat),
        "lm_pvalue": float(lm_pvalue),
        "f_statistic": float(f_stat),
        "f_pvalue": float(f_pvalue),
    }
