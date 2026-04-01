---
id: OP-43
title: "N-normalization divergence of A_E"
domain: [theory, numerics]
difficulty: medium
status: open
deep-research-tier: B
blocks: []
blocked-by: []
roadmap-tasks: []
papers: [1212.0631]
date-opened: 2026-03-31
date-updated: 2026-03-31
---

# OP-43: N-normalization divergence of A_E

## 1. Statement

The CJ amplitude coefficient A_E, defined by CJ = A_E E^2 V_4, scales
as N^{1.24 +/- 0.05} with the number of sprinkled points N. Since CJ
itself converges to a well-defined function of the curvature (the
bridge formula is verified), the divergence of A_E indicates that the
current normalization convention is incorrect. Find the proper
N-normalization that gives a finite continuum limit for A_E, or prove
that no such normalization exists.

## 2. Context

The CJ observable is constructed as a sum over N elements of the
causal set. Each element contributes a local kurtosis excess that
depends on the chain-length distribution in its neighborhood. The
sum is then divided by a normalization factor (currently N itself)
to give an intensive quantity.

If the individual contributions were independent and identically
distributed, the sum would scale as N and the mean as 1, giving a
finite A_E. However, the contributions are correlated: nearby elements
share chains, and the correlation structure introduces non-trivial
N-scaling.

The observed scaling A_E proportional to N^{1.24} means that the sum
grows faster than N. Equivalently, the mean contribution per element
grows as N^{0.24}. This divergence must be absorbed by a proper
normalization to define a continuum limit.

## 3. Known Results

- **A_E scaling:** Measured at N = 1000, 2000, 5000, 10000 on pp-wave
  backgrounds. Log-log fit gives exponent 1.24 +/- 0.05.
- **sigma_0 scaling:** sigma_0 / N^{1/4} = 0.299 (CV = 0.48%). The
  fluctuation amplitude scales as N^{1/4}, which is CONVERGENT after
  dividing by N^{1/4}.
- **CJ bridge formula:** CJ = [8/(3 x 9!)] (8 pi / 15) E^2 V_4.
  This formula is verified numerically (agreement within 2% at
  N = 5000). The formula does not contain N explicitly, suggesting
  the continuum limit exists.
- **Reconciliation:** The bridge formula gives CJ as an integral over
  the diamond, not as a sum over elements. The passage from sum to
  integral involves a factor of rho = N / V_4 (the sprinkling
  density). If A_E is measured as (sum / N) / E^2, but the correct
  normalization is (sum / V_4) / E^2, then A_E(correct) =
  A_E(measured) x (N / V_4) = A_E(measured) x rho. Since
  V_4 proportional to T^4 is fixed and N = rho V_4 varies, the
  discrepancy is A_E proportional to N^{0.24} vs the expected N^0.
- **Dimensional analysis:** [CJ] = dimensionless (kurtosis excess).
  [E^2] = length^{-4}. [V_4] = length^4. So CJ / (E^2 V_4) is
  dimensionless and should be N-independent in the continuum limit.

## 4. Failed Approaches

1. **Simple N-rescaling.** Dividing A_E by N^{0.24} gives a constant,
   but 0.24 is a fitted exponent with no theoretical justification.
   A normalization that depends on a fitted parameter is not predictive.

2. **Volume-density normalization.** Replacing N with rho V_4 (where
   rho is the sprinkling density and V_4 is the diamond volume) does
   not help because rho V_4 = N by definition. The problem is not in
   the dimensional analysis but in the scaling of the sum.

3. **Per-chain normalization.** Dividing each element's contribution by
   the number of chains passing through it. This removes the N-dependence
   but also removes the curvature signal: the per-chain kurtosis excess
   is approximately zero at all N. The curvature signal is in the
   COLLECTIVE behavior of many chains, not in individual chains.

4. **Variance normalization.** Dividing CJ by sigma_0 (the standard
   deviation). Since sigma_0 proportional to N^{1/4}, this gives
   CJ / sigma_0 proportional to N^{1.24 - 0.25} = N^{0.99}, still
   divergent.

5. **Logarithmic normalization.** CJ / (N log N) was tested.
   At N = 1000-10000, the ratio decreases slightly, suggesting
   over-normalization. CJ / (N log(N)^{0.3}) gives a better fit but
   the exponent 0.3 has no theoretical basis.

## 5. Success Criteria

- Identify the correct normalization function g(N) such that
  CJ / (g(N) E^2 V_4) converges to a finite, non-zero constant
  as N -> infinity.
- Derive g(N) from the combinatorial structure of the CJ sum (not
  as a fitted power law).
- Verify that A_E(normalized) is N-independent to within 5% over
  the range N = 1000 to 20000.
- The normalized A_E must be consistent with the bridge formula
  coefficient 8/(3 x 9!) x (8 pi / 15) = 3.67 x 10^{-5}.

## 6. Suggested Directions

1. **Connection to OP-34.** If CJ proportional to N^alpha with
   alpha = 8/9, then A_E = CJ / (E^2 V_4) proportional to N^{8/9}.
   The normalization is then g(N) = N^{8/9}. But the measured A_E
   exponent is 1.24, not 8/9 = 0.889. The discrepancy (1.24 - 0.889 =
   0.35) may come from the N-dependence of the stratification.

2. **Stratification-corrected normalization.** The number of effective
   strata grows with N (more elements per bin at larger N). If the
   effective number of contributing strata scales as N^gamma, then
   the total CJ sum scales as N^{alpha + gamma}. Measuring gamma
   independently (from population counts) would determine g(N) =
   N^{alpha + gamma}.

3. **Poisson fluctuation subtraction.** Part of the N^{1.24} scaling
   may come from Poisson noise in the sprinkling. The noise contributes
   CJ_noise proportional to N (central limit theorem) to the sum.
   Subtracting CJ_noise before normalizing may reduce the effective
   exponent. Measure CJ_noise on flat spacetime (where CJ = 0 on
   average) and subtract it from curved-spacetime results.

4. **Intensive vs extensive distinction.** Define CJ_intensive =
   CJ / N_bulk where N_bulk is the number of elements in the bulk
   strata (excluding boundary). If the boundary-bulk ratio depends on
   N, this changes the effective normalization. Measure N_bulk / N as
   a function of N.

5. **Exact small-N enumeration.** For N <= 30, enumerate all sprinklings
   and compute A_E exactly. Plot A_E(N) and look for the asymptotic
   form. Exact results may reveal whether g(N) is a power, logarithm,
   or more complex function.

## 7. References

- Roy, M., Sinha, D. and Surya, S. (2013). "Discrete geometry of a
  small causal diamond." arXiv:1212.0631.
- Bollobas, B. and Brightwell, G. (1991). "The width of random graph
  orders." *Rand. Struct. Alg.* 2, 37-49.
- Dowker, F. and Glaser, L. (2013). "Causal set d'Alembertians for
  various dimensions." arXiv:1305.2588.

## 8. Connections

- **OP-34 (N-scaling exponent):** The A_E divergence is a direct
  consequence of the N-scaling of CJ. If OP-34 determines alpha
  exactly, the divergence exponent 1.24 - alpha gives the
  stratification scaling gamma.
- **OP-35 (stratification measure):** The stratification procedure
  affects the effective number of contributing terms and thus the
  N-normalization. A proof that the measure converges (OP-35) would
  constrain gamma.
- **OP-40 (b_eff):** sigma_0 / N^{1/4} = 0.299 is finite.
  Understanding why sigma_0 converges (OP-40) while A_E diverges
  would clarify the normalization structure.
- **OP-42 (A_E universality):** The universality test compares
  A_E across geometries at fixed N. If A_E diverges, the ratio
  A_E(Sch) / A_E(ppw) might acquire N-dependent corrections,
  but current data shows the ratio is N-independent (0.508 at all
  tested N).
