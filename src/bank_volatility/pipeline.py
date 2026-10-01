from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AssetSpec
from .diagnostics import arch_lm, ljung_box
from .mean import aligned_fitted_and_residuals, fit_arma
from .risk import backtest_var, conditional_var
from .volatility import fit_volatility, garch_persistence

TRADING_DAYS = 252


def analyze_asset(
    name: str,
    returns: pd.Series,
    spec: AssetSpec,
    var_levels=(0.95, 0.99),
    test_window: int | None = 252,
) -> dict:
    """Fit the report specification and return models, volatility, diagnostics, and VaR."""

    mean_result = fit_arma(returns, spec.mean)
    mu, residuals = aligned_fitted_and_residuals(returns, mean_result)

    volatility_result, sigma_daily, z = fit_volatility(residuals, spec.volatility)
    sigma_ann = (sigma_daily * np.sqrt(TRADING_DAYS)).rename("sigma_ann")

    var = {
        level: conditional_var(mu, sigma_daily, volatility_result, confidence=level)
        for level in var_levels
    }
    var_backtests = [
        {"asset": name, **backtest_var(returns, var[level], level, test_window=test_window)}
        for level in var_levels
    ]

    diagnostics = {}
    for lag in (10, 20, 30):
        diagnostics[f"LBz_p@{lag}"] = float(ljung_box(z, lags=(lag,)).loc[lag, "lb_pvalue"])
        diagnostics[f"LBz2_p@{lag}"] = float(ljung_box(z**2, lags=(lag,)).loc[lag, "lb_pvalue"])

    return {
        "name": name,
        "returns": returns,
        "mean_result": mean_result,
        "mu": mu,
        "residuals": residuals,
        "arch_lm": arch_lm(residuals),
        "volatility_result": volatility_result,
        "sigma_daily": sigma_daily,
        "sigma_ann": sigma_ann,
        "standardized_residuals": z,
        "diagnostics": diagnostics,
        "var": var,
        "var_backtests": pd.DataFrame(var_backtests),
        "garch_persistence": garch_persistence(volatility_result),
    }


def common_index(results: dict[str, dict]) -> pd.Index:
    index = None
    for result in results.values():
        current = result["sigma_ann"].dropna().index
        index = current if index is None else index.intersection(current)
    return index


def summarize_common_window(results: dict[str, dict]) -> pd.DataFrame:
    index = common_index(results)
    rows = []
    for asset, result in results.items():
        sigma = result["sigma_ann"].reindex(index).dropna()
        params = result["volatility_result"].params
        vol_name = result["volatility_result"].model.volatility.name.upper()
        rows.append(
            {
                "asset": asset,
                "n": len(sigma),
                "mean_ann": float(sigma.mean()),
                "median_ann": float(sigma.median()),
                "p90_ann": float(sigma.quantile(0.90)),
                "p95_ann": float(sigma.quantile(0.95)),
                "max_ann": float(sigma.max()),
                "garch_alpha_plus_beta": result["garch_persistence"],
                "egarch_beta": float(params.get("beta[1]", np.nan)) if "EGARCH" in vol_name else np.nan,
                "egarch_gamma": float(params.get("gamma[1]", np.nan)) if "EGARCH" in vol_name else np.nan,
                "nu": float(params.get("nu", np.nan)),
            }
        )
    return pd.DataFrame(rows).set_index("asset")
