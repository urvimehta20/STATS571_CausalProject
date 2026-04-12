# Causal Discovery for Nonstationary Financial Time Series

## Introduction

This project implements and evaluates **causal discovery** on observational financial and macroeconomic time series, centered on the framework of **CD-NOTS**: constraint-based discovery allowing **nonstationarity**, **lags**, and **nonlinear** dependence. The motivating fact is that many series violate assumptions behind simpler methods (stationarity, no lag structure, linear Gaussian structure), which can distort both graphs and any downstream estimands.

We distinguish two layers:

1. **Structure learning** (which variables appear adjacent or directed in a graph under CI-based rules and context).
2. **Treatment-style estimands** after a graph is fixed, using **regression adjustment** on an explicit covariate set.

The repository therefore contains: (i) a custom **CD-NOTS**-style pipeline (simulations, PCMCI benchmark, case-study edge tables), (ii) **causal-learn**’s **CD-NOD** on the same raw CSVs with a **context index**, and (iii) **OLS with HAC standard errors** linking the directed subgraph to reported coefficients.

**Research questions.** (a) Can we **export** and **reuse** discovered structure (directed vs. undirected) for transparent regression estimands rather than figures alone?

## Dataset

We use two public datasets aligned with feasible case studies:

1. **Fama–French + Apple (daily, 2000–2022)**  
   Five Fama–French daily factors (`Mkt_RF`, `SMB`, `HML`, `RMW`, `CMA`, `RF` as available) and Apple returns (`AAPL_RET`), joined on trading dates. File: `data/raw/famafrench_apple_daily.csv`.

2. **Macroeconomic panel (monthly, 2000–2023)**  
   Unemployment, CPI, and PPI for six countries (US, Canada, Japan, France, UK, Italy). CPI/PPI in **month-over-month percent change** where applicable. File: `data/raw/macro_countries_monthly.csv`.

## Potential outcomes and identification:

**Observational units.** Index trading days by \(i = 1,\ldots,n\) for the daily panel, or country–months for macro. There is **no randomized assignment** of factor or macro realizations (contrast with NSW in HW2). Thus the **difference in means** \(\hat{\tau} = \bar{Y}_1 - \bar{Y}_0\) from HW1/HW2 is **not** automatically unbiased for a causal effect; randomization inference and Neyman-style variance formulas apply to **experiments**, not to passive factor returns.

**Hypothetical interventions.** For a focal “treatment” variable \(Z_i\) (e.g. daily `RMW`) and outcome \(Y_i\) (e.g. `Mkt_RF`), one can still write **potential outcomes** \(Y_i(z)\) under a notional intervention that sets \(Z\) to level \(z\), and define unit-level effects \(Y_i(z) - Y_i(z')\). For continuous \(Z\), the regression coefficient on \(Z\) in a linear model is a **linear approximation** to a dose-response slope, not the same object as \(\hat{\tau}\) for a binary experiment unless the structural model is linear and correctly specified.

**Adjustment set \(L\).** Under a **causal DAG** assumed to hold, if \(L\) **blocks all backdoor paths** from \(Z\) to \(Y\), then a conditional mean model \(E[Y \mid Z, L]\) identifies the effect of \(Z\) on \(Y\) under standard assumptions (HW2’s regression ideas are the same *conditioning* logic, with \(X\) playing the role of confounders). Here \(L\) is built as **parents(\(Z\))** in the **directed** part of a **CD-NOD** output, intersected with observed columns—explicitly a **heuristic** because the learned graph is only an estimate and may be a **PDAG**.

**SUTVA / consistency.** We need a stable link between observed \((Y_i, Z_i, L_i)\) and the hypothetical \(Y_i(z)\): e.g. factor definitions and data construction do not change discretely mid-sample; no “hidden versions” of what `RMW` means. **Violations** include structural breaks, unmodeled common shocks hitting all factors, and misspecification of lags—analogous to spillovers or multiple treatment versions discussed in HW1.

**Faithfulness and Markov properties.** CI tests used in discovery assume **faithfulness** (independences in data match \(d\)-separation in some generating graph) and **sufficient conditioning** for nonstationarity (time or country **context** in CD-NOD). These can fail under measurement error, hidden confounding, or finite-sample CI errors.

## Methods

### CD-NOTS implementation (custom pipeline)

1. **Stage 1 (construction):** Build lagged variables up to max lag \(L\) and add a time-index node \(T\).  
2. **Stage 2 (skeleton):** Conditional independence testing; remove edges when independence is not rejected (default \(\alpha = 0.05\)).  
3. **Stage 3 (orientation):** Temporal priors (\(T \to X\) for nonstationary nodes), lag directions, v-structures where supported.  
4. **Stage 4 (closure):** Meek-style rules to orient additional edges.

**CI interface:** ParCorr, KCIT, RCoT, CMIknn naming under one API. 

**Baseline:** PCMCI on matched synthetic settings.

### Causal-learn CD-NOD on case-study CSVs

- **API:** `causallearn.search.ConstraintBased.CDNOD`; driver `discovery2/run_causal_learn.py` calls `cdnod(data, c_indx, alpha)`.  
- **Context \(c\_indx\):** For Fama–French + Apple, **integer time order** (row index after `dropna`); appended as node `context_time`. For macro, **integer country code**; single-country runs use constant context; pooled `--country all` uses `context_country`.  
- **Outputs:** For each run tag (`cdnod_famafrench`, `cdnod_macro_US`, `cdnod_macro_all`): `*_nodes.csv`, `*_edges.csv`, `*_directed_edges.csv` (`from`, `to`), `*_undirected_edges.csv` (`node_a`, `node_b`), plus DOT/PNG (Graphviz `dot` or NetworkX fallback).

### Graph-guided regression (post-discovery)

`experiments/lecture13_graph_adjustment.py` reads `cdnod_<tag>_directed_edges.csv`, sets \(L =\) **parents(\(Z\))** in that digraph (restricted to data columns), and estimates  

\[
Y_i = \beta_0 + \beta_Z Z_i + \beta_L^\top L_i + \varepsilon_i
\]

by **OLS** with **HAC** (Newey–West style) covariance—appropriate when \(\varepsilon_i\) may be serially correlated, unlike the i.i.d.-motivated variance formulas in HW1/HW2. When **parents(\(Z\))** is empty in the directed subgraph (e.g. `SMB` in one run), users may pass **`--extra-controls`** so \(L\) is still economically meaningful; when parents exist (e.g. **\(Z=\)** `RMW`, **\(Y=\)** `Mkt_RF` gives \(L=\{\texttt{SMB},\texttt{HML}\}\)), \(L\) is **graph-derived only**.

## Results

The repository produces:

- **Simulation outputs:** Metrics (e.g. F1, precision, recall, SHD, runtime) across node/sample grids: `results/tables/simulation_metrics.csv`, `results/tables/simulation_summary.csv`.  
- **Benchmark:** CD-NOTS vs PCMCI: `results/tables/benchmark_pcmci.csv`.  
- **Custom CD-NOTS case studies:** `results/tables/case_famafrench_apple_edges.csv`, `results/tables/case_macro_edges.csv`.  
- **Causal-learn CD-NOD artifacts:** `discovery2/outputs/cdnod_*` (CSVs + figures).  
- **Illustrative graph-guided tables:** e.g. `results/tables/lecture13_adjust_famafrench_RMW_Mkt_RF.csv`, `results/tables/lecture13_adjust_famafrench_SMB_HML.csv`.

**Qualitative takeaway:** Performance improves with sample size; CI choice matters; allowing **nonstationarity** and **context** is important for financial/macro series. 

Regression coefficients are **assumption-dependent** (correct DAG, sufficient \(L\), linearity); they connect **estimated structure** to **numeric estimands** also, connects covariate adjustment to \(\hat{\tau}_{\mathrm{sta}}\)—but without experimental identification.

## Assumption diagnostics and sensitivity

Stress-tests aligned with both the paper and course material:

- Vary **CI method** (ParCorr, KCIT, RCoT, CMIknn), **max lag \(L\)**, and **\(\alpha\)**.  
- **Pooled vs. segmented** time windows.  
- **Causal-learn:** \(\alpha\) in `cdnod`, **US-only vs. pooled** macro context.  
- **HAC lag length** in graph-guided regressions.

## Limitations, conclusion, and future work

**Limitations.** Stage 4 is a Meek-style approximation, not the full module-change statistic. Public data omit the paper’s Bloomberg case. **Discovery \(\neq\)** truth: directed edges are algorithm outputs; undirected edges are **not** used in the adjustment step. **No randomization** implies coefficients are **not** \(\hat{\tau}\) from a CRD without additional structural assumptions.

**Conclusion.** The repo gives a **reproducible** path from **CD-NOTS-style** code through **causal-learn CD-NOD** on the same raw files to **explicit CSVs** and **HAC-adjusted** regressions—bridging **graph discovery** and **treatment-effect language** from STATS 571/671.

**Future work.** Richer kernel CIs; full module-change orientation; rolling-window **stability** analyses; broader asset and country coverage.
