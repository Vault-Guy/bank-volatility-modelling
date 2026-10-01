from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bank_volatility.config import ASSET_SPECS
from bank_volatility.data import close_prices, compute_log_returns, load_finam_export
from bank_volatility.diagnostics import adf_test, ljung_box
from bank_volatility.mean import search_arma
from bank_volatility.pipeline import analyze_asset
from bank_volatility.volatility import search_symmetric_garch


def main() -> None:
    parser = argparse.ArgumentParser(description="ARMA-GARCH/EGARCH bank-volatility analysis")
    parser.add_argument("--ticker", required=True, choices=sorted(ASSET_SPECS))
    parser.add_argument("--file", required=True, help="Path to a Finam-style daily market-data export")
    parser.add_argument("--search", action="store_true", help="Repeat ARMA and symmetric-GARCH grid searches")
    args = parser.parse_args()

    frame = load_finam_export(args.file)
    prices = close_prices(frame)
    returns = compute_log_returns(prices)
    spec = ASSET_SPECS[args.ticker]

    print(f"{args.ticker}: {len(prices)} prices / {len(returns)} returns")
    print("\nADF prices (constant + trend)")
    print(json.dumps(adf_test(prices, regression="ct"), indent=2))
    print("\nADF returns (constant)")
    print(json.dumps(adf_test(returns, regression="c"), indent=2))

    print("\nLjung-Box: returns")
    print(ljung_box(returns).to_string())
    print("\nLjung-Box: squared returns")
    print(ljung_box(returns**2).to_string())

    if args.search:
        arma_all, arma_candidates = search_arma(returns, q_max=10 if args.ticker == "MOEXFN" else 5)
        print("\nTop ARMA models by BIC")
        print(arma_all.head(10).to_string(index=False))
        print("\nTop ARMA candidates with white residuals")
        print(arma_candidates.head(10).to_string(index=False))

    result = analyze_asset(args.ticker, returns, spec)

    print("\nSelected mean model")
    print(result["mean_result"].summary())
    print("\nARCH-LM on mean-model residuals")
    print(json.dumps(result["arch_lm"], indent=2))

    if args.search:
        print("\nTop symmetric GARCH candidates")
        print(search_symmetric_garch(result["residuals"]).head(10).to_string(index=False))

    print("\nSelected volatility model")
    print(result["volatility_result"].summary())
    print("\nStandardized-residual diagnostics")
    print(json.dumps(result["diagnostics"], indent=2))
    print("\nVaR backtests")
    print(result["var_backtests"].to_string(index=False))


if __name__ == "__main__":
    main()
