# English research summary

> This is an English-language summary of the full analytical report. The [original report](../reports/original_report_ru.pdf) is written in Russian.

## Scope

The study analyses daily closing prices and log-returns of four Russian banking equities — **SBER, VTBR, T, SVCB** — together with the **MOEX Financials Index (MOEXFN)**. The longest series run from September 2014 to February 2026; T-Bank and Sovcombank are analysed from their respective listing histories.

## Empirical workflow

The original analysis follows a standard financial-econometrics workflow:

1. compare price levels with log-returns;
2. test stationarity with the Augmented Dickey-Fuller test;
3. inspect ACF/PACF and Ljung-Box statistics;
4. fit low-order ARMA models for the conditional mean;
5. test ARMA residuals for ARCH effects;
6. fit GARCH-family models with Student-t innovations;
7. diagnose standardized residuals;
8. compare conditional volatility across instruments;
9. estimate 95% and 99% conditional one-day VaR;
10. backtest VaR using Kupiec and Christoffersen tests.

## Selected models

| Asset | Mean model | Volatility model |
|---|---|---|
| SBER | ARMA(2,4) | GARCH(1,1)-t |
| VTBR | ARMA(3,3) | GARCH(1,1)-t |
| T | ARMA(1,1) | EGARCH(1,1)-t |
| SVCB | ARMA(0,0) | GARCH(1,1)-t |
| MOEXFN | ARMA(2,5) | GARCH(1,1)-t |

## Key observations

- Price levels are non-stationary while log-returns are stationary.
- Squared returns show substantial serial dependence, motivating conditional-volatility models.
- Student-t innovations fit the heavy-tailed return distributions better than Gaussian innovations in the selected GARCH searches.
- Volatility persistence is high for the long-history instruments.
- T-Bank displays statistically significant asymmetric volatility: its EGARCH `gamma[1]` is negative, so negative return shocks have a larger effect on future volatility than positive shocks of equal magnitude.
- On the 530-observation common comparison window, VTBR has the highest 95th-percentile conditional volatility, while SBER and MOEXFN are substantially less volatile.
- 99% VaR backtests in the original notebook do not reject correct unconditional coverage for any of the five instruments at the 5% level; 95% SBER VaR is more conservative than expected in the final 252-observation backtest window.

## Important comparability caveat

The full-history volatility summaries use different sample lengths because T-Bank and Sovcombank have shorter listing histories. Direct cross-asset comparisons are therefore made on the common date intersection in `results/common_window_volatility.csv`.

Also, `alpha + beta` is a meaningful persistence summary for the standard GARCH(1,1) models but **not** a directly comparable EGARCH persistence measure. For T-Bank the repository therefore reports the EGARCH `beta` and `gamma` coefficients separately.
