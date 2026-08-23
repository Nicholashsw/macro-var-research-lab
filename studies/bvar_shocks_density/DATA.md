# Data — Study 2

All series are public. Quarterly, 2000Q1–2025Q1.

## Endogenous block (FRED)

| Variable | Series | Transform |
|---|---|---|
| Real GDP growth | `GDPC1` | YoY % |
| CPI inflation | `CPIAUCSL` | YoY % |
| Real effective exchange rate | `RBUSBIS` | YoY % |
| 10-year Treasury yield | `GS10` | level |
| Policy rate | `FEDFUNDS` | level |

## Constructed shocks

Each shock is the standardised AR(1) surprise (z-scored on the training window only; see
`src/shock_builder.py`) of:

| Shock | Underlying series | Public source |
|---|---|---|
| Oil | WTI spot price | FRED `WTISPLC` / EIA |
| Equity wealth | S&P 100 index | S&P / FRED proxies |
| Consumption | PCE quantity index, SA | BEA / FRED `PCECC96` |
| Investment | Private fixed investment, 2017$ SAAR | BEA / FRED `FPIC1` |
| Credit impulse | Domestic nonfinancial sector debt, SAAR (differenced) | Fed Z.1 / FRED `TCMDO` |
| Fiscal impulse | Cyclically adjusted primary balance | CBO |

`shock_builder.py` reads these from a set of spreadsheet exports named in its `SOURCES` map. Those
exports are not in the repository; rebuild them from the sources above with the same column layout
(date in column A, value in column B) or point the map at your own files. The scripts look for them
in this study's `data/` directory by default (gitignored); set `BVAR_DATA_DIR` to read them from
somewhere else. The G7 extension (`fetch_g7.py`) pulls its panel directly from FRED at run time.

## Outputs committed

`results/*.csv` are the executed evaluation tables the paper reports; `figures/` are the paper
figures.
