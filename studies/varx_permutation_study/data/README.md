# Data

All source data is public and keyless. Nothing in this folder is committed;
run `python src/fetch_data.py` to populate it.

## Sources

- **FRED** (Federal Reserve Bank of St. Louis), keyless CSV endpoint. Fifteen
  series build the panel: GDPC1, PCEPILFE, FEDFUNDS, AHETPI, PAYEMS, GS10,
  TB3MS, PERMIT, WTISPLC, CPIAUCSL, GPDIC1, TNWBSHNO, TWEXBMTH, DTWEXBGS, GCEC1.
- **NY Fed Holston-Laubach-Williams** one-sided natural-rate estimates, from the
  current-estimates workbook. Column K, US one-sided r-star.
- **ALFRED** archival vintages, fetched on demand by `varx_realtime.py` with the
  `vintage_date` parameter and cached under `data/vintages`. Revised national
  accounts only: GDPC1, PCEPILFE, GPDIC1, GCEC1, TNWBSHNO.
- **Kaenzig (2021)** oil supply news shock, from the `dkaenzig/oilsupplynews`
  repository, vintage 2025M12, "Monthly" sheet.
- **Bauer and Swanson (2023)** monetary policy surprises, from the FRBSF Center
  for Monetary Research workbook, "Monthly (update 2023)" sheet.

The Ramey-Zubairy military news shock is discussed as a fiscal comparison but not
fetched, because the series ends in 2015 and carries almost no variation in the
post-1990 forecasting window.

Nicholas Hong | Built for educational and research purposes. Not financial advice.
