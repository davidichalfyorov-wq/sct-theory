---
id: OP-33
title: "Unified cross-program comparison table"
domain: [predictions, theory]
difficulty: medium
status: partial
deep-research-tier: A
blocks: []
blocked-by: []
roadmap-tasks: [COMP-1]
papers: ["1205.3637", "0805.2909", "1401.4793", "hep-th/0505113", "hep-th/0508202", "1205.0971", "1311.2898", "1107.2403", "1110.5249", "gr-qc/0407052", "1806.05407", "1503.06472", "1306.3512", "2005.09550", "0812.2214", "1407.3002", "1806.02406", "gr-qc/0606032"]
date-opened: 2026-03-31
date-updated: 2026-03-31
progress: "28/54 cells filled (52%), 3 discriminating axes identified, 1 universal axis"
---

# OP-33: Unified cross-program comparison table

## 1. Statement

Produce a systematic, quantitative comparison of SCT with the
major competing quantum gravity programs: Loop Quantum Gravity (LQG),
Asymptotic Safety (AS), Causal Dynamical Triangulations (CDT),
string theory, and Infinite Derivative Gravity (IDG). The comparison
must cover at least nine quantitative axes and provide specific
numerical predictions (not qualitative assessments) wherever possible.

## 2. Context

SCT has now computed predictions along several axes: spectral
dimension, logarithmic BH entropy correction, GW speed, PPN
parameters, UV behavior of the graviton propagator, and matter
coupling structure. Each of these predictions has counterparts in
other quantum gravity programs, but no systematic comparison exists.

Individual SCT-vs-X comparisons have been made informally during
the development process, but they are scattered across different
documents and do not use a uniform framework. The COMP-1 task in the
roadmap calls for a living comparison table that is updated as new
results become available.

The value of this comparison is threefold: (1) it clarifies where
SCT makes distinctive predictions that could distinguish it from
competitors; (2) it identifies areas of agreement that reflect
universal features of quantum gravity; (3) it highlights where
experimental data could discriminate between programs.

## 3. Known Results

Individual SCT results available for comparison:

- **Spectral dimension:** d_S(UV) ~ 2 (ML definition, NT-3).
- **BH entropy log correction:** c_log = 37/24 (conditional, MT-1).
- **GW speed:** c_T = c exactly (NT-4c).
- **PPN parameters:** gamma = 1 + exponentially small corrections;
  indistinguishable from GR (PPN-1, LT-3d).
- **UV propagator:** G_TT ~ 1/(k^2 Pi_TT), entire-function dressing.
  Not 1/k^4 (Stelle) and not 1/k^2 (GR).
- **Matter coupling:** SM content fixes alpha_C = 13/120 (parameter-free).
  N_s = 4, N_D = 22.5, N_v = 12.
- **Singularity resolution:** not proven. Gap G1 blocks the analysis.
- **Lambda_cc:** no prediction. SCT does not address the cosmological
  constant problem.
- **Inflation:** conditional (INF-1). Scalaron mass problem open.
- **Dispersion:** not computed (OP-27).
- **Form factors:** F_1, F_2 entire functions (NT-1, NT-1b). Known
  in closed form.

Competitor predictions (from literature, not computed in this project):

- **LQG:** d_S(UV) ~ 2 (Modesto 2009), c_log = -1/2 (Meissner 2004),
  c_T = c (standard), singularity resolved (bounce).
- **AS:** d_S(UV) = 2 exactly (Lauscher-Reuter 2005),
  c_log program-dependent, UV fixed point for G and Lambda.
- **CDT:** d_S(UV) ~ 1.80 +/- 0.25 (lattice, Ambjorn et al. 2005),
  matter coupling not computed.
- **String theory:** d_S varies by compactification, c_log depends on
  charges, extended objects modify UV behavior.
- **IDG:** G_TT ~ e^{-k^2/M^2}/k^2 (ghost-free by construction),
  d_S = 2 in UV, singularity resolved.

## 4. Failed Approaches

No systematic comparison has been attempted. The difficulty is
primarily one of scope: each competing program has its own conventions,
approximation schemes, and levels of rigor, making direct numerical
comparison non-trivial.

## 5. Success Criteria

- A comparison table with at least nine quantitative axes:
  (1) d_S(UV), (2) c_log, (3) singularity resolution,
  (4) inflation (n_s, r), (5) dispersion relation,
  (6) PPN parameters, (7) UV propagator behavior,
  (8) Lambda_cc prediction, (9) matter coupling structure.
- Each entry must cite a specific paper with the relevant
  computation. Entries marked "not computed" where applicable.
- Clear identification of axes where SCT makes distinctive
  (discriminating) predictions versus axes where all programs agree.
- Assessment of which experimental data (current or planned) could
  discriminate between programs on each axis.
- The table must be maintainable: structured as a machine-readable
  document (YAML or Markdown table) that can be updated as new
  results emerge.

## 6. Suggested Directions

1. Literature survey: for each axis, identify the best available
   computation in each program. Prioritize recent (post-2020)
   results with the highest truncation order or lattice volume.

2. Normalization: express all predictions in common units and
   conventions. For d_S, use the heat kernel definition. For c_log,
   use the coefficient of log(A/l_P^2) in S_BH. For PPN, use the
   standard PPN gauge.

3. Discriminating axes: identify axes where programs make
   incompatible predictions. Candidates include: c_log (SCT: 37/24,
   LQG: -1/2), UV propagator behavior (SCT: entire, IDG: Gaussian,
   AS: power-law), and matter coupling (SCT: fixed by SM, AS:
   universal fixed point).

4. Experimental forecast: for each discriminating axis, estimate the
   experimental precision needed to distinguish between programs.
   Map this onto planned experiments (LISA, ET, DESI, Euclid, CMB-S4).

## 7. References

1. Carlip, S. (2017). "Dimension and dimensional reduction in quantum
   gravity." Class. Quant. Grav. 34, 193001. arXiv:1705.05417.
2. Codello, A., Percacci, R. and Rachwal, L. (2008). "The
   renormalization group and Weyl invariance." arXiv:0805.2909.
3. Modesto, L. and Shapiro, I. L. (2016). "Superrenormalizable
   quantum gravity with complex ghosts." Phys. Lett. B 755, 279.
   arXiv:1512.07600.
4. Biswas, T., Mazumdar, A. and Siegel, W. (2006). "Bouncing
   universes in string-inspired gravity." JCAP 0603, 009.
   arXiv:0508194.
5. Dona, P., Eichhorn, A. and Percacci, R. (2014). "Matter matters
   in asymptotically safe quantum gravity." Phys. Rev. D 89, 084035.
   arXiv:1401.4793.

## 8. Connections

- Depends on **OP-30** (running constants), **OP-31** (form factor
  comparison), and **OP-32** (spectral dimension comparison) for
  specific entries in the table.
- Related to all prediction problems (OP-26 through OP-29): each
  prediction feeds into the comparison table.
- The COMP-1 roadmap task is iterative: the table should be updated
  each time a new SCT result is obtained.
- This is the highest-impact deliverable for external communication
  of the SCT program's status.
