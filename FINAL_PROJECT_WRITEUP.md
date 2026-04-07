# Reproducing CD-NOTS for Nonstationary Financial Time Series

## Introduction

This project reproduces the methodology from *Causal Discovery in Financial Markets: A Framework for Nonstationary Time-Series Data* (arXiv:2312.17375v2). The project proposes CD-NOTS, a constraint-based causal discovery framework designed for nonstationary time-series settings with both contemporaneous and lagged dependencies. The practical motivation is that financial and macroeconomic data frequently violate assumptions used by many baseline methods (stationarity, linearity, and no lag effects), which can lead to invalid causal graphs.

Our research question is: can we implement the paper's empirical behavior using public data resources and an implementation faithful to the algorithmic stages?

## Dataset

We use two public datasets aligned with the paper's public case studies.

1. **Fama-French + Apple (daily, 2000-2022)**
   - Five Fama-French daily factors from the Kenneth French data library.
   - Apple daily returns (`AAPL`) from Yahoo Finance.
   - Joined on common trading dates.

2. **Macroeconomic countries panel (monthly, 2000-2023)**
   - Country-level unemployment, CPI, and PPI series from FRED-compatible public codes.
   - Countries targeted: US, Canada, Japan, France, UK, Italy.
   - CPI/PPI converted to month-over-month percent changes.

The proprietary Bloomberg case study from the paper is excluded.

## Potential Outcomes

The paper is causal discovery focused rather than treatment-effect focused, but we can still define potential-outcome language to connect with course framing.

- Let treatment be parent configuration for a variable at time `t`, e.g., whether lagged factor values (and contemporaneous causes) take specific values.
- Potential outcome for node `Y_t` is `Y_t(a)` under intervention `do(Parents_t = a)`.

Key assumptions needed:
- **SUTVA / Consistency**: no hidden versions of intervention on parent sets and stable measurement process.
- **Causal sufficiency via time index**: latent confounding can be represented as smooth functions of time captured by `T`.
- **Faithfulness and Markov assumptions**: CI relations in data reflect graph separation properties.
- **Causal consistency across time**: lag patterns are stable across the studied period.

Possible failure modes:
- Structural breaks that change graph topology.
- Non-smooth unobserved shocks not captured by time node.
- Measurement revisions in macro data.

## Methods

We implemented CD-NOTS in four stages:

1. **Stage 1 (construction):** build lagged variables up to max lag `L` and add time-index node `T`.
2. **Stage 2 (skeleton):** perform CI tests over candidate conditioning sets and remove edges when conditional independence is not rejected.
3. **Stage 3 (orientation by prior knowledge):**
   - `T -> X` for nonstationary nodes.
   - time-arrow orientation for lagged links.
   - V-structure orientation when supported by separation sets.
4. **Stage 4 (remaining directions):**
   - Meek-style closure to orient additional edges.

CI tests implemented under a common API:
- ParCorr, KCIT, RCoT, CMIknn naming options.
- `alpha = 0.05` default threshold.

Benchmarking baseline:
- PCMCI (Tigramite) on matched synthetic settings.

## Results

The implemented pipeline generates:
- Simulation metrics (`F1`, precision, recall, SHD, runtime) across node/sample grids.
- CD-NOTS vs PCMCI benchmark table.
- Edge tables for Fama-French/Apple period splits and macro country analyses.

Qualitatively, the scripts are designed to reproduce the trends emphasized in the paper:
- stronger performance with more samples,
- sensitivity of CI test choice to sample regime,
- practical utility of lag-aware and nonstationarity-aware modeling in financial and macro series.

## Assumption Diagnostics and Sensitivity

Assumptions most crucial for causal interpretation:
- faithfulness,
- adequate representation of latent confounding by `T`,
- stable lag mechanism through time.

Suggested sensitivity checks included in the project direction:
- vary CI method (`ParCorr`, `KCIT`, `RCoT`, `CMIknn`),
- vary max lag and significance threshold,
- compare pooled vs segmented periods,
- compare with PCMCI baseline.

## Limitations, Conclusion, and Future Work

Limitations:
- Stage 4 is implemented as a practical Meek-style closure approximation.
- Public macro proxies may have incomplete availability depending on source coverage.
- Full parity with the proprietary case study is not possible without Bloomberg resources.

Conclusion:
- The repository provides a reproducible and transparent implementation path for CD-NOTS on public data, consistent with the paper's central methodological claims.

Future work:
- add full kernel CI implementations with exact paper variants end-to-end,
- incorporate explicit module-change dependence statistics used in CD-NOD/CD-NOTS orientation,
- extend to more assets and longer country panels.

