# VAR / BVAR / Local Projections — Curated Paper Pack

24 open-access PDFs covering the foundations, modern frontier, and applied
methodology of vector autoregressions, Bayesian VARs, structural identification,
and local projections. Suggested reading order at the bottom.

================================================================
TIER 1 — FOUNDATIONS (must read)
================================================================

Sims_1980_MacroeconomicsAndReality.pdf
    Sims, C. (1980). Econometrica 48(1).
    The paper that started multivariate macro time series. Critiques structural
    macro models, introduces reduced-form VARs and recursive identification.

Litterman_1986_BayesianVAR.pdf
    Litterman, R. (1986). JBES 4(1).
    Origin of the Minnesota prior. Documents how shrinkage toward random walk
    improves macro forecasts.

BlanchardQuah_1989_DemandSupplyDisturbances.pdf
    Blanchard, O. & Quah, D. (1989). AER 79(4).
    Long-run identification: demand shocks have no permanent effect on output,
    supply shocks do. The classic alternative to Cholesky.

StockWatson_2001_VectorAutoregressions.pdf
    Stock, J. & Watson, M. (2001). JEP 15(4).
    The cleanest textbook-level introduction. Read first if you want one paper
    covering reduced-form, recursive, and structural VARs with examples.

================================================================
TIER 2 — MODERN TOOLKIT (Local Projections)
================================================================

Jorda_Taylor_2024_LocalProjections_Review.pdf
    Jorda, O. & Taylor, A. (2024). FRBSF WP / JEL forthcoming.
    NOTE: this is the modern review article that covers the original Jorda 2005
    methodology in depth. The 2005 AER paper itself is paywalled by AEA.

PlagborgMoller_Wolf_2021_LP_VAR.pdf
    Plagborg-Moller, M. & Wolf, C. (2021). Econometrica 89(2).
    Settles the LP-vs-VAR debate: same population estimand, finite-sample
    bias-variance trade-off. Essential modern reference.

PlagborgMoller_etal_2025_LP_VAR_Primer.pdf
    Plagborg-Moller, Montiel Olea, Qian, Wolf (2025). NBER Macro Annual.
    Practical primer for applied macroeconomists. Lag selection, control choice,
    bias correction, CI construction.

MontielOlea_PlagborgMoller_2021_LP_Inference.pdf
    Montiel Olea, J. & Plagborg-Moller, M. (2021). Econometrica 89(4).
    Lag-augmented local projections with normal critical values. Robust to
    persistence and long horizons.

StockWatson_2018_ExternalInstruments.pdf
    Stock, J. & Watson, M. (2018). Economic Journal 128.
    LP-IV framework. Combining local projections with external instruments.
    Current state of the art for medium-horizon causal effects.

Ramey_2016_MacroeconomicShocks.pdf
    Ramey, V. (2016). Handbook of Macroeconomics Vol 2.
    Encyclopaedic survey of shock identification: monetary, fiscal, technology,
    uncertainty. Single best reference for structural macro.

Ramey_2022_MacroShocksPostscript.pdf
    Ramey, V. (2022). Postscript / updates to the 2016 handbook chapter.

================================================================
TIER 3 — IDENTIFICATION FRONTIER
================================================================

Uhlig_2005_SignRestrictions.pdf
    Uhlig, H. (2005). JME 52.
    Sign restrictions identification. Robust alternative to zero restrictions.

RubioRamirez_Waggoner_Zha_2010_SVAR_Identification.pdf
    Rubio-Ramirez, J., Waggoner, D., Zha, T. (2010). REStud 77.
    Formal theory of identification in SVARs. Algorithms for sign-restricted
    models. Technical but the standard reference.

GertlerKaradi_2015_MonetaryPolicySurprises.pdf
    Gertler, M. & Karadi, P. (2015). AEJ:Macro 7(1).
    High-frequency FOMC surprises as monetary instruments. The canonical modern
    monetary-shock paper. Read this for interview signal.

NakamuraSteinsson_2018_InformationEffect.pdf
    Nakamura, E. & Steinsson, J. (2018). QJE 133(3).
    Refinement of high-frequency identification. Distinguishes monetary policy
    news from Fed information shocks. Important nuance.

================================================================
TIER 4 — BAYESIAN AND LARGE SYSTEMS
================================================================

Banbura_etal_2010_LargeBayesianVARs.pdf
    Banbura, M., Giannone, D., Reichlin, L. (2010). JAE 25(1).
    Shows Minnesota prior with appropriate tightness scales to 20+ variables.
    Foundational for large BVAR practice at central banks.

Bernanke_etal_2005_FAVAR.pdf
    Bernanke, B., Boivin, J., Eliasz, P. (2005). QJE 120(1).
    FAVAR introduction. Compresses 100+ series into a handful of factors plus
    observed VAR variables.

Primiceri_2005_TVP_SVAR.pdf
    Primiceri, G. (2005). REStud 72.
    TVP-VAR with stochastic volatility. Heavy machinery for time-varying
    relationships. Used widely at central banks.

Giannone_Lenza_Primiceri_2015_PriorSelection.pdf
    Giannone, D., Lenza, M., Primiceri, G. (2015). REStat 97(2).
    Hierarchical priors: data-driven hyperparameter selection. The pragmatic
    answer to "how tight should the Minnesota prior be?"

StockWatson_2002_DiffusionIndexes_DFM.pdf
    Stock, J. & Watson, M. (2002). JBES 20(2).
    Dynamic Factor Models for forecasting. Compressed-information approach
    distinct from FAVAR. Workhorse at central banks for nowcasting.

================================================================
TIER 5 — REGIME SWITCHING AND SG-RELEVANT MACRO
================================================================

Hamilton_2005_RegimeSwitchingSurvey.pdf
    Hamilton, J. (2005). Palgrave survey.
    NOTE: this is Hamilton's later survey of Markov-switching models. The
    foundational 1989 Econometrica paper is paywalled by JSTOR.

Rey_2015_DilemmaTrilemma.pdf
    Rey, H. (2015). NBER WP 21162.
    Global financial cycle hypothesis. VIX-driven capital flows constrain small
    open economy monetary policy. Highly relevant for Singapore.

GilchristZakrajsek_2012_CreditSpreads_EBP.pdf
    Gilchrist, S. & Zakrajsek, E. (2012). AER 102(4).
    Excess bond premium (EBP) construction. The credit-spread variable to include
    in any modern macro VAR for leading-indicator content.

================================================================
TIER 6 — PRACTICAL / COMPUTATIONAL
================================================================

Kuschnig_Vashold_2021_BVAR_R_Package.pdf
    Kuschnig, N. & Vashold, L. (2021). JSS 100(14).
    Companion paper to R's BVAR package. Cleanest practical implementation of
    hierarchical Minnesota prior.

================================================================
SUGGESTED READING PATHS
================================================================

Path A: Quant interview prep (8-10 hours)
    Stock-Watson 2001 -> Sims 1980 -> Jorda-Taylor 2024 -> Plagborg-Moller-Wolf 2021
    -> Gertler-Karadi 2015 -> Ramey 2016 (sections 1-3).
    Covers reduced-form, structural identification, local projections, modern
    instruments. Sufficient signal for screens at BAM, Citadel, Point72.

Path B: independent macro research forecasting work (12-15 hours)
    Litterman 1986 -> Banbura-Giannone-Reichlin 2010 -> Giannone-Lenza-Primiceri 2015
    -> Stock-Watson 2002 DFM -> Kuschnig-Vashold 2021 -> Plagborg-Moller-Wolf 2021.
    Builds the BVAR forecasting arm to pair with existing LP work.

Path C: FYP regime overlay (4-6 hours)
    Hamilton 2005 -> Primiceri 2005 -> Rey 2015.
    Keep this module light: regime indicator drives factor return conditioning.

Path D: Full structural depth (25-30 hours)
    All of Tier 1 + Tier 3 + Ramey 2016 + Rubio-Ramirez-Waggoner-Zha 2010
    + Nakamura-Steinsson 2018 + Stock-Watson 2018.
    For aspiring macro researcher or macro strategy long-horizon role.

================================================================
NOT INCLUDED (HARD PAYWALLED, NO OPEN VERSION AVAILABLE)
================================================================

Jorda 2005 (AER) — covered by Jorda-Taylor 2024 review which includes the
    original methodology and modern extensions.
Hamilton 1989 (Econometrica) — covered by Hamilton 2005 survey.
Kim & Roubini 2000 (JME) — small open economy SVAR; identification approach is
    summarized in Ramey 2016 and Stock-Watson 2018.
Mertens & Ravn 2013 (AER) — proxy SVAR; same instrument approach covered in
    detail in Stock-Watson 2018 and Gertler-Karadi 2015.
Carriero, Clark, Marcellino 2019 (J Econometrics) — large BVAR with SV;
    covered in part by Banbura et al 2010 and Giannone-Lenza-Primiceri 2015.

