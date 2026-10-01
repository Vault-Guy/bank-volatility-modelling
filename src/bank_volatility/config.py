from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MeanSpec:
    """ARMA(p, q) specification, represented as ARIMA(p, 0, q)."""

    p: int
    q: int
    trend: str = "n"


@dataclass(frozen=True)
class VolSpec:
    """GARCH-family specification."""

    vol: str = "GARCH"
    p: int = 1
    o: int = 0
    q: int = 1
    dist: str = "t"


@dataclass(frozen=True)
class AssetSpec:
    mean: MeanSpec
    volatility: VolSpec


# Final report specifications from the executed coursework notebook.
ASSET_SPECS: dict[str, AssetSpec] = {
    "SBER": AssetSpec(MeanSpec(2, 4), VolSpec("GARCH", 1, 0, 1, "t")),
    "VTBR": AssetSpec(MeanSpec(3, 3), VolSpec("GARCH", 1, 0, 1, "t")),
    "T": AssetSpec(MeanSpec(1, 1), VolSpec("EGARCH", 1, 1, 1, "t")),
    "SVCB": AssetSpec(MeanSpec(0, 0), VolSpec("GARCH", 1, 0, 1, "t")),
    "MOEXFN": AssetSpec(MeanSpec(2, 5), VolSpec("GARCH", 1, 0, 1, "t")),
}
