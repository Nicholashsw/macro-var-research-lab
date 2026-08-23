# A Permutation Study of VARX Forecasting Models

Do exogenous shocks help a vector autoregression forecast the US macroeconomy, and if so, when? This repository runs that question to the ground on public data: a large permutation study of vector autoregressions with exogenous shocks (VARX), a real-time replication on archival vintages, a head-to-head against externally identified shocks, and a full density evaluation. Everything is rebuilt from public sources with no API keys, and every result is reported as it came out, including the ones that went nowhere.

The companion paper is in [`paper/varx_permutation_study.pdf`](paper/varx_permutation_study.pdf). An interactive console for browsing the leaderboard and forecasts is in [`dashboard/dashboard.html`](dashboard/dashboard.html) (self-contained, open it in a browser).

## The question and the design

The core is a three-variable VAR: GDP growth, core PCE inflation, and a policy stance defined as the real fed funds rate minus the Holston-Laubach-Williams natural rate. Around that core the study crosses 16 endogenous variable sets, 12 exogenous shock subsets drawn from real oil price, private investment, household net worth, and government spending growth, three shock treatments (raw, purged of contemporaneous GDP, purged of the full endogenous vector), and three information-criterion lag rules. That is 1,632 specification cells and 1,014 unique models, each evaluated in an expanding-window walk-forward from 2009Q4 at horizons of one, four, and eight quarters, under both an unconditional and a realized exogenous path. All growth rates are 400 times the log difference, which sidesteps the MA(3) residual autocorrelation that year-over-year transforms introduce, and every model carries 2020Q2 and 2020Q3 dummies.

## What the study found

**Shocks give a small point-forecast edge that is mostly COVID and mostly noise.** The best GDP specification at one quarter cuts RMSE about 12 percent against the best pure VAR, and stacking all four shocks helps rather than overfits, because each shock enters with only two lags and the dummies absorb 2020. But only 4.0 percent of specifications beat the core VAR at the 5 percent level in Diebold-Mariano tests, which is what noise alone delivers, and the ex-COVID gains are far smaller. The leaderboard ordering is not a discovery, it is one member of a large statistically equivalent set.

**Purging is an identification tool, not a forecasting tool.** Residualizing the shocks costs accuracy for GDP, inflation, and the dollar, and pays only in the interest rate equation. BIC is the better lag rule for three of four targets, while the persistent stance variable rewards the longer AIC lags and earns its largest gain, an out-of-sample R-squared of 0.40, at the eight-quarter horizon.

**The edge survives revisions.** A real-time replication on ALFRED vintages, rebuilding the panel from the data available roughly 45 days after each quarter closed, leaves the GDP shock advantage almost intact, from 16.1 percent to 15.9 percent on the full window and 8.1 to 8.0 ex-COVID. The one fragile choice is GDP purging, which degrades 35 percent out of vintage because it projects on a still-unrevised GDP number.

**Externally identified shocks do not beat the cheap proxies.** Swapping the growth-rate proxies for Kaenzig's oil supply news shock and Bauer-Swanson's FOMC surprises does not improve forecasts and often worsens them, even when the identified series is handed a full-sample look-ahead the proxies lack. The proxy forecast encompasses the identified one in 22 of 24 cells, leak-free combinations of the two are 0.2 to 0.8 percent worse than the better component alone, and the more heavily a series is orthogonalized toward clean identification the worse it forecasts. Identification strips exactly the predictable variation a point forecaster needs.

**Shocks earn nothing in the predictive distribution either.** Under the naive plug-in density, holding future shocks at their mean, the all-four-shock model looks sharp but is badly overconfident: its nominal 90 percent GDP interval covers only 68 percent of outcomes and the COVID quarter drives its log score to minus thirteen. Pricing in future-shock uncertainty restores coverage to 91 percent and removes the catastrophe, but it widens the intervals, so the honest density is no sharper than the pure VAR's. A Bayesian predictive check shows the pure VAR's own residual under-coverage is estimation noise, closed by parameter uncertainty rather than fat tails, with only a mild long-horizon inflation shortfall left over.

The short version: for reduced-form macro forecasting the cheap growth-rate proxies are the right instrument, not a compromise, and exogenous shocks earn their keep in neither the mean nor the tails.

## Layout

```
src/            core library, analysis modules, runners, figure scripts
paper/          LaTeX source and compiled PDF
notebooks/      executed walk-through of the main study
figures/        paper and repository figures
dashboard/      self-contained interactive console
results/        comparison tables (parquet) and summary statistics (json)
data/           populated by src/fetch_data.py, not committed
```

The analysis modules build on each other rather than duplicating code. `varx_lab.py` holds the estimation, lag selection, forecasting, and metric machinery. `varx_realtime.py` adds the ALFRED replication, `varx_identified.py` and `varx_monetary.py` the identified-shock comparisons, `varx_combine.py` the forecast-encompassing test across both, `varx_density.py` the log-score and calibration evaluation, and `varx_bayes_density.py` the Bayesian decomposition of interval coverage.

## Reproducing

```
pip install -r requirements.txt
cd src
python fetch_data.py         # public sources, no API keys -> ../data/
python run_study.py          # main walk-forward, writes results and dashboard payload
python run_realtime.py       # ALFRED real-time replication
python varx_identified.py    # oil identification
python varx_monetary.py      # monetary identification
python varx_combine.py       # forecast combination and encompassing
python varx_density.py       # density forecasts
python varx_bayes_density.py # parameter uncertainty
```

`fetch_data.py` resolves `../data/` from its own location, so the panel always lands in the study's
`data/` directory. The analysis scripts read that panel and write their intermediates
(`results.parquet`, `payload.json`) and their `*_compare.parquet` / `*_stats.json` outputs into the
current working directory, so running them from `src/` as above keeps a re-run away from the
committed copies in `results/`; copy the ones you want to keep across yourself.

The figure scripts (`make_*.py`) regenerate the plots, and `build_dashboard.py` inlines the study
payload into the standalone console. On a 2026 laptop the full walk-forward (1,014 models, 63,918
result rows) takes about 35 seconds; the extensions run in seconds against cached data.

`results/id_stats.json` and `results/mp_stats.json` were regenerated from the same public data on
2026-08-23; the earlier committed copies were empty skeletons. The identified-shock figures
`figures/fig_id.png` and `figures/fig_mp.png` are the originals and are not byte-identical to a
re-run, because FRED vintages later than the original run carry national-accounts revisions.

## Data

Everything is public and keyless: FRED for the macro panel, the NY Fed for the Holston-Laubach-Williams natural rate, ALFRED for archival vintages, the `dkaenzig/oilsupplynews` repository for the oil supply news shock, and the FRBSF Center for Monetary Research for the Bauer-Swanson surprises. See [`data/README.md`](data/README.md) for the exact series and endpoints.

## Notes

This is an educational and methodological study built entirely on public data. It reconstructs standard tools from scratch, a pure-numpy VAR and VARX estimator, a Klein-style forecasting routine, Diebold-Mariano and forecast-encompassing tests, a companion-form predictive covariance, and a Normal-inverse-Wishart posterior, so the mechanics are visible rather than hidden behind a library call.

---

Nicholas Hong | Built for educational and research purposes. Not financial advice.
