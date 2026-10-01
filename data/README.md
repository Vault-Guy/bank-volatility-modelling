# Data

The project was developed using daily market data exported from **Finam**.

Raw data are intentionally excluded from version control. Put local files in:

```text
data/raw/
```

## Expected schema

The loader accepts Finam-style text/CSV exports with columns equivalent to:

```text
TICKER, PER, DATE, TIME, OPEN, HIGH, LOW, CLOSE, VOL
```

Angle brackets such as `<DATE>` and `<CLOSE>` are automatically removed. Delimiters are detected from comma, semicolon, tab, or pipe-separated files.

Both `YYMMDD` and `YYYYMMDD` date formats are supported.

## Instruments used in the original analysis

- `SBER` — Sberbank
- `VTBR` — VTB
- `T` — T-Bank
- `SVCB` — Sovcombank
- `MOEXFN` — MOEX Financials Index

The original notebook used file names approximately matching:

```text
SBER_140901_260219_merged.txt
VTBR_140901_260219_merged.txt
T_140901_260219_merged.txt
SVCB_140901_260219_merged.txt
MOEXFN_140901_260219_merged.txt
```
