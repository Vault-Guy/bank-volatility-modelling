# Bank Volatility Modelling with ARMA-GARCH and EGARCH

Econometric study of daily returns, conditional volatility, and downside risk for major Russian banking equities and the MOEX Financials sector index.

**Assets:** SBER · VTBR · T · SVCB · MOEXFN  
**Methods:** ADF · ACF/PACF · Ljung-Box · ARMA · ARCH-LM · GARCH/EGARCH · Student-t innovations · VaR · Kupiec/Christoffersen backtests  
**Stack:** Python · pandas · NumPy · SciPy · statsmodels · arch · matplotlib

## Research questions

1. Are price levels non-stationary while log-returns are stationary?
2. Is there exploitable serial structure in conditional mean returns?
3. Do returns exhibit volatility clustering and ARCH effects?
4. Which ARMA and GARCH-family specifications best describe each instrument?
5. How persistent are volatility shocks across banks and the sector index?
6. How do 95% and 99% one-day VaR profiles compare across instruments?

## Main findings

The analysis finds the standard stylized facts of financial returns across the sample: stationary log-returns, heavy tails, and pronounced volatility clustering.

| Instrument | Conditional mean | Conditional volatility | Main finding |
|---|---|---|---|
| **SBER** | ARMA(2,4) | GARCH(1,1)-t | High volatility persistence |
| **VTBR** | ARMA(3,3) | GARCH(1,1)-t | Very persistent volatility shocks and the highest stress-volatility quantiles on the common window |
| **T** | ARMA(1,1) | EGARCH(1,1)-t | Statistically significant negative asymmetry / leverage effect |
| **SVCB** | ARMA(0,0) | GARCH(1,1)-t | Little conditional-mean structure, but clear volatility clustering |
| **MOEXFN** | ARMA(2,5) | GARCH(1,1)-t | Lowest average risk profile among the studied series |

On the **530-observation common comparison window**, median annualized conditional volatility was about **21.0% for SBER**, **31.9% for VTBR**, **32.7% for T**, **30.4% for SVCB**, and **21.3% for MOEXFN**. The 95th percentile of conditional volatility was highest for VTBR (~60.3%).

For T-Bank, EGARCH(1,1)-t improves AIC/BIC relative to symmetric GARCH(1,1)-t, and the estimated asymmetry parameter is negative and statistically significant, indicating that negative return shocks increase future volatility more than positive shocks of the same magnitude.

## Repository structure

```text
.
├── data/
│   ├── README.md
│   └── raw/                  # raw Finam exports (not tracked)
├── docs/
│   └── report_summary.md
├── notebooks/
│   └── 01_bank_volatility_analysis.ipynb
├── results/
│   ├── common_window_volatility.csv
│   ├── diagnostics.csv
│   ├── model_specifications.csv
│   └── var_backtest.csv
├── scripts/
│   └── run_analysis.py
├── src/bank_volatility/
│   ├── config.py
│   ├── data.py
│   ├── diagnostics.py
│   ├── mean.py
│   ├── pipeline.py
│   ├── plotting.py
│   ├── risk.py
│   └── volatility.py
└── tests/
```

## Data

The original study used daily market data exported from **Finam**. Raw market data are not redistributed in this repository.

Place local exports into `data/raw/`. The loader accepts common Finam schemas such as:

```text
<TICKER>;<PER>;<DATE>;<TIME>;<OPEN>;<HIGH>;<LOW>;<CLOSE>;<VOL>
```

The source notebook used files named approximately:

```text
SBER_140901_260219_merged.txt
VTBR_140901_260219_merged.txt
T_140901_260219_merged.txt
SVCB_140901_260219_merged.txt
MOEXFN_140901_260219_merged.txt
```

See [`data/README.md`](data/README.md) for details.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

or install the package in editable mode:

```bash
pip install -e .
```

## Run the analysis

For one instrument:

```bash
python scripts/run_analysis.py \
  --ticker SBER \
  --file data/raw/SBER_140901_260219_merged.txt
```

To repeat ARMA/GARCH model searches rather than only fit the report specification:

```bash
python scripts/run_analysis.py \
  --ticker SBER \
  --file data/raw/SBER_140901_260219_merged.txt \
  --search
```

## Methodology

The reusable pipeline mirrors the original research workflow while removing repeated instrument-specific code:

1. load and clean daily prices;
2. compute continuously compounded returns;
3. test price/return stationarity with ADF;
4. inspect ACF/PACF and Ljung-Box diagnostics;
5. select ARMA(p,q) specifications by AIC/BIC subject to residual diagnostics;
6. test ARMA residuals for ARCH effects;
7. fit GARCH-family volatility models with Student-t innovations;
8. diagnose standardized residuals and squared standardized residuals;
9. estimate conditional 95% and 99% one-day VaR;
10. backtest VaR using Kupiec unconditional-coverage and Christoffersen independence/conditional-coverage tests;
11. compare conditional-volatility distributions on a common date window.

A compact notebook is provided in [`notebooks/01_bank_volatility_analysis.ipynb`](notebooks/01_bank_volatility_analysis.ipynb), while the implementation lives in `src/bank_volatility/`.

## Reproducibility and refactoring

The original Colab notebook was exploratory and repeated the same analysis block separately for each asset. This repository refactors that code into reusable functions and explicit model specifications. The CSV files in `results/` are exported from the executed original analysis and are included so the principal reported outputs remain visible even without redistributing the raw market data.

One methodological correction is made explicit in the refactored presentation: for **EGARCH**, the GARCH-style quantity `alpha + beta` is not a comparable persistence measure. The T-Bank specification is therefore reported using its EGARCH `beta` coefficient and asymmetry parameter rather than a misleading `alpha + beta` comparison.

## Author

**Nikita Lukianenko**

Quantitative research interests: time-series analysis, volatility modelling, statistical arbitrage, portfolio construction, and risk analytics.
