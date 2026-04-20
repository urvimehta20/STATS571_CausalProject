# STATS 571 Final Project Write-Up

## Title

**Causal Discovery for Nonstationary Financial Time Series**

## Authors

Deepan Islam  
Urvi Mehta  
Santosh Desai

## Summary

This version of the report is written in the notation and style used in class. The main shift is
that each case study now states the causal objects directly:

- the unit of analysis,
- the treatment or exposure `Z`,
- the outcome `Y`,
- the observed covariates `X`,
- the assignment space `Omega`,
- and the role of consistency, SUTVA, exchangeability, and overlap.

The report also removes the old lecture-based naming and uses `class_oriented` instead. Tables and
figures are placed next to the discussion they support.

## Case Definitions

### Financial case

- Unit: trading day `t` after preprocessing and lagging.
- `Z_t = SMB_t`.
- `Y_t = HML_t`.
- `X_t`: observed factor returns, risk-free rate, time context, and lagged history available at
  day `t`.
- `L_t`: graph-derived adjustment set. In the saved run, `L_t = {RMW_t}`.
- `Omega_F`: feasible SMB exposure paths over the sample window.

The class-style estimand is a dose-response contrast
\[
\tau_F(z,z') = E\{Y_t(z) - Y_t(z')\}.
\]

Consistency means observed `HML_t` equals `Y_t(Z_t)`. SUTVA means that once the chosen lag
structure is fixed, the potential outcome at time `t` does not depend on unmodeled assignments
outside that encoded history. Exchangeability is the claim that the relevant potential outcomes are
independent of treatment given `X_t`. Overlap means there is enough SMB variation within strata of
`X_t` to avoid pure extrapolation.

### Macro case

- Unit: US month `t` after preprocessing.
- `Z_t = unemployment_t`.
- `Y_t = Delta cpi_t`, stored back into the `cpi` column after differencing.
- `X_t`: observed macro covariates, time context, PPI, and lagged history available at month `t`.
- `L_t`: no clean graph-derived set in the saved run because the graph stays partially undirected.
- `Omega_M`: feasible unemployment paths over the sample window.

The macro estimand is
\[
\tau_M(z,z') = E\{Y_t(z) - Y_t(z')\}.
\]

The same lecture assumptions apply, but they are harder to defend. The sample is smaller, the
series are more strongly trended, and the graph ambiguity near unemployment and CPI weakens the
backdoor argument.

## Course Framing

As discussed in Lecture 13 titled: *Recap: Confounding and Causal DAGs*, the graph only helps if
it blocks noncausal paths. As discussed in Lecture 15 titled: *Methods for adjustment beyond
stratification*, the downstream analysis only becomes causal after stating the estimand and the
identification assumptions.

The saved regressions are best read as outcome regressions:
\[
E(Y_t \mid Z_t, L_t) = \beta_0 + \beta_Z Z_t + \beta_L^\top L_t.
\]

That matches Lecture 16 titled: *Recap: Outcome regression / model standardization*. It does not
match Lecture 18 titled: *Recap: Augmented Inverse Probability Weighting*, because there is no
propensity score model and no doubly robust correction.

## What the Code Supports

The code follows the broad CD-NOTS structure, but not the full paper implementation. The main gaps
are still the same:

- `kcit`, `rcot`, and `cmiknn` fall back to partial correlation,
- stage 4 is approximate,
- and the saved benchmark does not reproduce the paper's claim against PCMCI.

The graph-guided adjustment script is now named `class_oriented_graph_adjustment.py`, and the saved
regression outputs now use `class_oriented_adjust_...csv`.

## Main Results

- The financial case remains the strongest result.
- `class_oriented_adjust_famafrench_SMB_HML.csv` gives an SMB on HML coefficient of `0.271` with
  standard error `0.034`, using `L = {RMW}`.
- The graph falsification report passes only `6/13` implied independence checks.
- The SMB to HML edge appears in `8/57` rolling windows, while the coefficient is significant in
  `31/57`.

- The macro case is weaker.
- `class_oriented_adjust_macro_US_unemployment_cpi.csv` gives an unemployment on `Delta cpi`
  coefficient of `-0.074` with standard error `0.028`.
- The script also raises an ambiguity warning because unemployment and CPI remain undirected
  neighbors.

- The quick simulations are useful only as a scaffold.
- Their identical F1 scores across CI labels are exactly what we should expect once the code audit
  shows that the non-ParCorr methods still fall back to ParCorr.
- The saved benchmark favors PCMCI, not CD-NOTS.

## Bottom Line

The report now presents the project as a course-style causal analysis built on top of a reproducible
discovery pipeline. That is a legitimate final-project claim. A stronger claim about scientific
replication would require real KCIT, RCoT, and CMIknn backends, a paper-faithful stage 4, and a
clearer sensitivity analysis for hidden bias.
