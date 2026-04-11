# GCP Experiment: Large-N Weyl Coefficient Extraction

## Goal
Extract the discrete a₂ Seeley-DeWitt coefficient (Weyl correction to
causal diamond volume) by going to large enough N that the boundary
effect (~ τ⁻²) becomes subdominant to the interior correction (~ τ⁴).

## Scientific Background

The volume of a causal diamond of proper time τ in d=4:
```
V(τ) = V₀τ⁴ [1 + a₁·R·τ² + a₂·(c₁R² + c₂Ric² + c₃K)·τ⁴ + ...]
```

For pp-wave (R=0, Ric=0, K=8ε²):
```
V(τ) = V₀τ⁴ [1 + a₂c₃·K·τ⁴ + ...]
```

At N=2000-3000, local data shows δV/V ~ τ⁻² (boundary-dominated).
Theory predicts at large N: boundary effect ~ N^{-1/4}, interior ~ const.
At N >> 20000, interior τ⁴ correction should dominate at large τ.

## Key Property: CRN Valid for PP-wave

det(g) = 1 for pp-wave in (t,x,y,z) coordinates. Therefore:
- Physical volume = coordinate volume
- CRN (fixed coordinates) = proper sprinkling comparison
- No density artifact

## Experiment Design

### Parameters
- N = [5000, 10000, 20000, 50000]
- eps = [0.5, 1.0] (weak field; K=2.0 and K=8.0; estimated SNR ~ 1.6 and 6.4 at N=50000)
- T = 1.0 (causal diamond half-size)
- M = 20 per (N, eps) (CRN paired, variance is low)
- Fine tau bins: 25 bins from 0.03 to 0.45
- Bonferroni: 4 N values × 2 eps = 8 tests, alpha = 0.01/8 = 0.00125

### For each trial:
1. Sprinkle N points into 4D causal diamond (sorted by time)
2. Build flat causal matrix C_flat (O(N²) — sparse at ~5% density)
3. Build pp-wave causal matrix C_ppw (same coordinates, different condition)
4. Build synthetic lapse matrix C_syn (same g_tt, no wave components)
5. Compute C²_flat, C²_ppw, C²_syn (interval volumes) — sparse @ sparse
6. For each tau bin: compute mean V for flat, ppw, syn among
   pairs causal in ALL THREE matrices
7. Compute residual ratio: (V_ppw - V_syn) / V_flat at each tau

### Key output:
- Profile δV_residual/V_flat vs τ for each N
- Fit to model: δV/V = A_boundary/τ² + A_interior·τ⁴
- Extract A_interior / K = a₂c₃ at each N
- Check if a₂c₃ stabilizes at large N

### Success criteria (PRE-REGISTERED):
- If A_interior > 0 with p < 0.01 at N ≥ 20000 → τ⁴ behavior detected
- If a₂c₃ stable (CV < 30%) across N=20000 and N=50000 → coefficient extracted
- Compare with Gibbons-Solodukhin continuum value

### Failure criteria:
- If A_interior not significant at N=50000 → boundary dominates at all scales
- If a₂c₃ changes sign or CV > 50% → not converged

## Computational Requirements

### Memory:
- N=5000: C is 5000² × 8 bytes = 200 MB (dense). Feasible.
- N=10000: 800 MB. Feasible with 16 GB RAM.
- N=20000: 3.2 GB. Feasible with 32 GB RAM.
- N=50000: 20 GB. Needs 64 GB RAM or sparse representation.

### Sparse alternative (for N=50000):
- Causal density ~4.5% → ~112M nonzeros
- scipy.sparse CSR: ~1.5 GB for C
- C² = C@C: fill increases to ~30-50% → ~10 GB. Tight.
- Alternative: compute C²[i,j] on-the-fly for sampled pairs only

### CPU time (estimate per trial):
- N=5000: C construction ~1s, C² ~2s, binning ~1s → ~5s/trial
- N=10000: ~5s, ~15s, ~5s → ~25s/trial
- N=20000: ~20s, ~120s, ~20s → ~160s/trial
- N=50000: ~120s, ~2000s, ~120s → ~40min/trial (sparse needed)

### Total time:
- N=5000: 20 trials × 5s = 2 min
- N=10000: 20 × 25s = 8 min
- N=20000: 20 × 160s = 53 min
- N=50000: 20 × 40min = 13 hours (sparse, or GPU)

### Recommended GCP VM:
- n2-highmem-16 (16 vCPU, 128 GB RAM) for N ≤ 20000
- n2-highmem-32 + GPU for N = 50000
- Estimated total: ~2-15 hours depending on max N

## Also run: N-sweep for scaling behavior (Path 3)

At pp-wave eps=10, run the lapse-separated residual at:
N = [500, 1000, 2000, 5000, 10000, 20000]

Check:
- Does |residual| scale with N?
- Does the profile SHAPE change (from τ⁻² toward τ⁴)?
- At what N does the crossover happen?

## Null controls (MANDATORY):
1. Conformal (eps=5): must give exact 0 at ALL N
2. Flat vs flat (eps=0): must give exact 0
3. Random DAG comparison at N=10000 (geometry-specificity)

## Memory management (N=50000):
- Use SPARSE matrices (scipy.sparse.csr_matrix)
- Process ONE metric at a time: build C, compute C², extract stats, delete
- Peak memory: ~10 GB (one sparse C + one sparse C²)
- Sequential: flat → ppwave → synthetic_lapse (not simultaneous)

## Synthetic lapse density artifact:
- det(g) for synthetic lapse = (1-eps*f/2) ≠ 1
- BUT: <f> = <x²-y²> = 0 by x↔y symmetry of causal diamond
- Average density correction = 0 at O(eps). Residual artifact = O(eps²).
- At eps=0.5: O(eps²) = O(0.25) — negligible compared to O(eps) signal.

## Analysis strategy for a₂ extraction:
- Do NOT fit the full τ range (boundary effect ~ τ⁻² dominates at small τ)
- Focus on LARGE-τ tail (τ > 0.25) where boundary effect is minimal
- Fit large-τ points to δV/V = Bτ⁴
- Check if B converges across N values
- ALSO fit two-regime model to full range: δV/V = A·τ^α + B·τ⁴ to see crossover

## R1 Objection: Nonlinear lapse subtraction
- Causal condition depends nonlinearly on metric components
- Subtracting synthetic lapse from ppwave removes lapse AT LEADING ORDER (O(ε))
- Residual ~ ε·(Δt·Δz + Δz²/2) terms = wave components (off-diagonal)
- At O(ε²): interaction between lapse and wave components exists
- Addressed by using small ε (0.5) where O(ε²) effects are ~0.25 relative
- The residual is NOT exact volume comparison, it's "pair-count difference
  from wave components" — physically meaningful but different from a₂

## Power estimate at N=50000, eps=1.0, τ=0.3:
- Density ρ ≈ 10⁶/volume ≈ 10⁶
- Mean V at τ=0.3: ρ × C₄ × τ⁴ ≈ 8000
- Expected interior δV: a₂c₃ × K × τ⁴ × V ≈ 0.005 × 8 × 0.008 × 8000 ≈ 2.6
- Number of pairs at τ~0.3: ~6M
- SE of mean δV: √Var(V)/√6M ≈ 90/2450 ≈ 0.04
- SNR ≈ 2.6/0.04 ≈ 65 → EASILY detectable if interior correction exists
- Even N=10000 should work: SNR ~ 65/5 ≈ 13

## Additional: Path 2 N-scaling
- At each N, compute 2neigh_var CRN delta
- If d(2neigh_var) INCREASES with N → local stat IS curvature-sensitive
- If d(2neigh_var) DECREASES with N → finite-size effect, nonlocality supported

## Continuum comparison: COMPUTED (2026-03-26)

CRITICAL FINDING: K = 0 for pp-wave (VSI spacetime).
All scalar curvature invariants vanish identically.
The volume correction comes from Bel-Robinson superenergy W = E² + B².

From Wang (2019, arXiv:1904.01034), eq. (79), vacuum d=4:
  δV/V₀ = (2032 E² + 720 B²) / (37800 × 16) × τ⁴

For pp-wave: E² = B² = ε²/2 (verified by Riemann computation).
  δV/V₀ = 1376 / 604800 × ε² × τ⁴ = 2.275 × 10⁻³ × ε² × τ⁴

TARGET COEFFICIENT: a₂_W = 0.002275 (per ε² τ⁴)

At ε=1.0, τ=0.3: δV/V₀ = 1.85 × 10⁻⁵
At N=50000, V(τ=0.3) ~ 8000: δV ~ 0.15 (detectable with millions of pairs)

NOTE: This is measurement of BEL-ROBINSON SUPERENERGY, not Kretschner.
First-ever on discrete spacetimes if successful.

## Anti-bias protocol:
- Pre-registered success/failure criteria above
- Adversarial proxy check (degree stats + extended) at each N
- Bonferroni correction: alpha = 0.01/8 = 0.00125
- Report ALL results including negatives
- 2 consecutive adversarial reviews must be clean before release
