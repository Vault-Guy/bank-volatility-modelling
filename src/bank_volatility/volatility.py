from __future__ import annotations

import pandas as pd

from .config import VolSpec


def _arch_model():
    try:
        from arch import arch_model
    except ImportError as exc:
        raise ImportError("The `arch` package is required. Install it with `pip install arch`.") from exc
    return arch_model


def fit_volatility(residuals: pd.Series, spec: VolSpec, scale_to_percent: bool = True):
    """Fit a GARCH-family model to conditional-mean residuals."""

    arch_model = _arch_model()
    x = pd.Series(residuals).dropna().astype(float)
    x_in = 100.0 * x if scale_to_percent else x

    result = arch_model(
        x_in,
        mean="Zero",
        vol=spec.vol,
        p=spec.p,
        o=spec.o,
        q=spec.q,
        dist=spec.dist,
        rescale=False,
    ).fit(disp="off")

    sigma = pd.Series(result.conditional_volatility, index=x.index, name="sigma")
    if scale_to_percent:
        sigma = sigma / 100.0
    z = pd.Series(result.std_resid, index=x.index, name="z")
    return result, sigma.rename("sigma_daily"), z


def search_symmetric_garch(
    residuals: pd.Series,
    p_max: int = 3,
    q_max: int = 3,
    distributions=("normal", "t"),
) -> pd.DataFrame:
    """Grid-search symmetric GARCH(p,q) models and rank them by BIC/AIC."""

    rows: list[dict] = []
    for dist in distributions:
        for p in range(1, p_max + 1):
            for q in range(1, q_max + 1):
                try:
                    result, _, _ = fit_volatility(
                        residuals,
                        VolSpec(vol="GARCH", p=p, o=0, q=q, dist=dist),
                    )
                    rows.append(
                        {
                            "p": p,
                            "q": q,
                            "dist": dist,
                            "aic": float(result.aic),
                            "bic": float(result.bic),
                            "loglikelihood": float(result.loglikelihood),
                            "converged": bool(result.convergence_flag == 0),
                        }
                    )
                except Exception:
                    continue

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No GARCH specification converged.")
    return frame.sort_values(["bic", "aic"]).reset_index(drop=True)


def garch_persistence(result) -> float | None:
    """Return alpha+beta for standard GARCH models; not for EGARCH."""

    model_name = result.model.volatility.name.upper()
    if "EGARCH" in model_name:
        return None
    params = result.params
    alpha = sum(float(v) for k, v in params.items() if str(k).startswith("alpha"))
    beta = sum(float(v) for k, v in params.items() if str(k).startswith("beta"))
    return alpha + beta
