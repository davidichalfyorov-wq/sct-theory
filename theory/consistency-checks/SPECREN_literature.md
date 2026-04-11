# SPEC-REN: Literature Agent Report — Spectral Renormalizability

## 1. Complete Paper Catalog

### 1A. Core van Suijlekom Program (YM Renormalization)

| # | arXiv | Authors | Title | Year | Key Result |
|---|-------|---------|-------|------|------------|
| 1 | 1101.4804 | van Suijlekom | Renormalization of the spectral action for the YM system | 2011 | **Superrenormalizability** of full YM spectral action; counterterms absorbed by shift f -> f + delta_f |
| 2 | 1104.5199 | van Suijlekom | Renormalization of the asymptotically expanded YM spectral action | 2011 | Explicit one-loop computation via zeta function regularization; counterterm = delta_Z * int F^2 |
| 3 | 1112.4690 | van Suijlekom | Renormalizability conditions for almost-commutative manifolds | 2011 | Graph-theoretical conditions (Krajewski diagrams) for renormalizability of spectral action on M x F |
| 4 | 1204.4070 | van Suijlekom | Renormalizability conditions for almost-commutative geometries | 2012 | Extension: SM spectral action is renormalizable as higher-derivative gauge theory |

### 1B. van Nuland--van Suijlekom Program (One-Loop, Cocycles, Ward Identities)

| # | arXiv | Authors | Title | Year | Key Result |
|---|-------|---------|-------|------|------------|
| 5 | 2104.09899 | van Nuland, van Suijlekom | Cyclic cocycles in the spectral action | 2021 | Spectral action = sum of Chern-Simons + Yang-Mills forms at all orders; entire odd (b,B)-cocycle |
| 6 | 2107.08485 | van Nuland, van Suijlekom | One-loop corrections to the spectral action | 2021 | **One-loop renormalizability** in Gomis-Weinberg sense; counterterms have same CS-YM form |
| 7 | 2209.10094 | van Nuland, van Suijlekom | Cyclic cocycles and one-loop corrections (review) | 2022 | Intelligible review combining 2104.09899 and 2107.08485 |

### 1C. Power Counting and Multi-Loop

| # | arXiv | Authors | Title | Year | Key Result |
|---|-------|---------|-------|------|------------|
| 8 | 2512.14581 | Hekkelman, van Nuland, Reimann | Power counting in the spectral action matrix model | 2025 | omega(G) = U + (p/d)(E_fi - V_fi); planar dominance; UV/IR mixing at L = d+2 |
| 9 | 2404.16338 | Hekkelman, McDonald, van Nuland | MOIs, pseudodifferential calculus, asymptotic expansions | 2024 | Perturbative expansion of spectral action for regular s-summable spectral triples |

### 1D. BV/BRST for Spectral Action

| # | arXiv | Authors | Title | Year | Key Result |
|---|-------|---------|-------|------|------------|
| 10 | 1604.00046 | Iseppi, van Suijlekom | NCG and the BV formalism: application to a matrix model | 2016 | BV spectral triple for U(2) matrix model; ghost fields from NCG |
| 11 | 2410.11823 | Iseppi | The BV construction for finite spectral triples | 2024 | BV and BRST complexes = Hochschild complex of coalgebra; full NCG-native BV |

### 1E. Hopf Algebra Renormalization (van Suijlekom)

| # | arXiv | Authors | Title | Year | Key Result |
|---|-------|---------|-------|------|------------|
| 12 | hep-th/0610137 | van Suijlekom | Renormalization of gauge fields: a Hopf algebra approach | 2006 | Ward/Slavnov-Taylor identities generate Hopf ideal; compatible with Connes-Kreimer |
| 13 | 0807.0999 | van Suijlekom | Structure of renormalization Hopf algebras for gauge theories I | 2008 | Classical master equation in BV-algebra implies Hopf ideals; applied to YM |
| 14 | 1007.4678 | van Suijlekom | Renormalization Hopf algebras for gauge theories and BRST | 2010 | Coaction on coupling constants and fields; Slavnov-Taylor as Hopf ideals |

### 1F. Spectral Action and Gravity

| # | arXiv | Authors | Title | Year | Key Result |
|---|-------|---------|-------|------|------------|
| 15 | 2001.05975 | Mistry, Pinzul, Rachwal | Spectral action approach to higher derivative gravity | 2020 | Full 6-derivative gravitational action from spectral action; rigid coefficient structure |
| 16 | 0705.1786 | Chamseddine, Connes | QG boundary terms from spectral action | 2007 | Spectral action predicts GH-Y boundary term with correct sign and coefficient |
| 17 | 1312.2235 | Kurkov, Lizzi, Vassilevich | High energy bosons do not propagate | 2013 | Quadratic spectral action has no positive derivative powers at high momentum; 2-pt function vanishes at short distance |
| 18 | 1108.3749 | Iochum, Levy, Vassilevich | Spectral action beyond the weak-field approximation | 2011 | Covariant computation to order 2 in gauge perturbation; UV decay as 1/p^4 |
| 19 | 1410.7999 | Alkofer, Saueressig, Zanusso | Spectral dimensions from the spectral action | 2014 | D_S(T) = 0 for all spins beyond EFT; supports non-propagation at high energy |
| 20 | 2511.05909 | Chamseddine | Hearing the Shape of the Universe (review) | 2025 | Comprehensive review; matrix-form spectral action; Feynman rules preserving NCG structure |

### 1G. Related / Foundational

| # | arXiv | Authors | Title | Year | Key Result |
|---|-------|---------|-------|------|------------|
| 21 | hep-th/9603053 | Connes | Gravity coupled with matter and NCG | 1996 | Foundation: spectral action principle; gauge group as normal subgroup of Diff |
| 22 | gr-qc/9612034 | Landi, Rovelli | GR in terms of Dirac eigenvalues | 1996 | Dirac eigenvalues as diffeomorphism-invariant observables for GR |
| 23 | 1103.0478 | Andrianov, Kurkov, Lizzi | Spectral action from anomalies | 2011 | Bosonic spectral action from fermionic scale anomaly cancellation |
| 24 | 2212.06533 | van Nuland | C*-algebraic results (PhD thesis) | 2022 | "Spectral Action vs Renormalization" section marked **in progress** |


## 2. The van Suijlekom Proof Mechanism (for YM)

### 2A. Architecture

The proof proceeds in three steps:

**Step 1: Higher-Derivative Interpretation.**
The spectral action S[A] = Tr(f(D_A/Lambda)) on flat M^4 has the asymptotic expansion:

S[A] ~ sum_{m>=0} Lambda^{4-m} f_{4-m} int_M a_m(x, D_A^2)

where a_m are Seeley-DeWitt invariants, which are gauge-invariant polynomials in F_{mu nu} and its covariant derivatives. The quadratic part takes the form:

S_0[A] ~ -(1/4) int tr F_hat phi_Lambda(Delta) F_hat

where phi_Lambda(Delta) = sum_k Lambda^{-2k} f_{-2k} c_k Delta^k. This is a **higher-derivative gauge theory**.

**Step 2: Power Counting -> Superrenormalizability.**
The propagator behaves as |p|^{-(n-2)} at high momentum (for expansion truncated at order n). The superficial degree of divergence is:

omega = (4-n)(L-1) + 4 - (E + E_tilde)

For n >= 8: omega < 0 when L >= 2. Therefore **all graphs are finite at loop order >= 2**. Only one-loop graphs with E + E_tilde <= 4 can diverge. The theory is **superrenormalizable**.

**Step 3: BRST Invariance -> Spectral Absorption.**
The one-loop effective action Gamma_1 is BRST-invariant: s(Gamma_1) = 0. By BRST cohomology results (Dixon, Barnich-Brandt-Henneaux), the only BRST-closed functional of order <= 4 in the fields is:

delta_Z int F_{mu nu} F^{mu nu}

This can be absorbed by a **shift of the spectral function**:

f_0 -> (1 + delta_Z) f_0

with all higher derivatives f^{(2k)}(0) unchanged. Renormalization of the YM spectral action is accomplished by a simple shift f -> f + delta_f.

### 2B. The van Nuland-van Suijlekom Extension (2021)

The key innovation is expressing the spectral action expansion using **noncommutative integrals** and **divided differences**:

**Expansion.** The spectral action for perturbations D -> D + V decomposes as:

S_D[V] = sum_k (int_{psi_{2k-1}} CS_{2k-1}(A) + (1/2k) int_{phi_{2k}} F^k)

where CS_{2k-1} are generalized Chern-Simons forms and F = dA + A^2 is the curvature. The brackets (noncommutative integrals) are expressed via divided differences of f':

(1/2) <Q,Q> = (1/2) sum_{k,l} Q_{kl} Q_{lk} f'[lambda_k, lambda_l]

**Propagator.** The gauge propagator is:

G_{kl} = 1/f'[lambda_k, lambda_l]

(the **inverse of the first divided difference**). This is bounded -- a key regularization property.

**Ward Identity.** The fundamental identity:

(z-D)^{-1} a - a(z-D)^{-1} = (z-D)^{-1} [D,a] (z-D)^{-1}

extends to a **quantum Ward identity** for one-loop 1PI functions:

<<V_1,...,aV_j,...,V_n>>^{1L} - <<V_1,...,V_{j-1}a,...,V_n>>^{1L} = <<V_1,...,[D,a],...,V_n>>^{1L}

**Result.** The divergent part of the one-loop quantum effective action has the **same CS-YM form** as the classical spectral action:

sum_n (1/n) <<V,...,V>>_infty^{1L} = sum_k (int_{tilde_psi_{2k-1}} CS_{2k-1}(A) + (1/2k) int_{tilde_phi_{2k}} F^k)

One-loop renormalization is realized by the transformation phi -> phi - tilde_phi, psi -> psi - tilde_psi in the space of noncommutative integrals. **The quantum spectral action is again a spectral action.**


## 3. Precisely WHERE the Proof Breaks for Gravity

### 3A. The Five Obstructions

**Obstruction 1: Inner Fluctuations vs. Metric Fluctuations.**
In the YM case, the gauge field A enters through inner fluctuations: D -> D + V where V = a_j [D, b_j]. The eigenvalues lambda_k of the background D are **fixed** — only V is quantized. For gravity, the metric g enters through the Dirac operator D[g] itself. The eigenvalues lambda_k(g) **depend on g**, which is the dynamical variable. There is no fixed background spectrum to define divided differences against.

- YM: Propagator = 1/f'[lambda_k, lambda_l] with fixed lambda's
- Gravity: lambda_k = lambda_k(g_0 + h) varies with the fluctuation h

This is the **most fundamental obstruction**. The entire divided-difference framework presupposes a fixed spectral background.

**Obstruction 2: Non-Polynomial Dependence.**
The YM spectral action, after asymptotic expansion, is **polynomial** in A (through the Seeley-DeWitt coefficients a_m, which are polynomial gauge invariants in F and its covariant derivatives). The gravitational spectral action is **non-polynomial** in the metric: a_m(D[g]^2) involves arbitrary powers of the Riemann tensor and its covariant derivatives. The Seeley-DeWitt coefficients a_{2n} grow factorially (Gevrey-1 asymptotic series, as established in MR-6). This means:

- YM: Finitely many vertices (for truncated expansion)
- Gravity: Infinitely many curvature structures at each order; factorial growth

**Obstruction 3: Diffeomorphism Group is Non-Compact.**
The YM proof uses BRST cohomology results that rely on the compact gauge group SU(N). The BRST-closed functionals of a given order are classified by results of Dixon-Dubois-Violette-Barnich-Brandt-Henneaux. For gravity, the symmetry group is the diffeomorphism group Diff(M), which is **infinite-dimensional and non-compact**. The BRST cohomology of gravity is substantially richer:

- YM BRST-closed at order 4: Only delta_Z int F^2
- Gravity BRST-closed at order 4: int R^2, int R_{mu nu}^2, int C^2, int GB, int R (multiple independent invariants)

At higher orders (order 6, 8, ...), the number of independent gravitational counterterms **grows without bound** (the Molien series gives 3, 8, ... parity-even invariants at quartic, sextic Weyl order). Stelle's R + R^2 gravity is renormalizable precisely because R^2 and C^2 (or equivalently R^2 and R_{mu nu}^2) span the full set of independent dimension-4 gravitational counterterms. But at dimension 6 and higher, the number of independent counterterms exceeds the number of spectral parameters.

**Obstruction 4: No Natural "Inner Fluctuation" for the Metric.**
In NCG, the gauge field arises as inner fluctuation: D -> D + A + JAJ* for elements of the algebra. The gravitational degree of freedom (the metric) is encoded in D itself, not as an inner fluctuation. There is no NCG-native way to write metric fluctuations as "inner fluctuations" of a fixed background:

- YM: V = a_j [D, b_j] for a_j, b_j in the algebra A
- Gravity: h_{mu nu} is not of the form a [D, b] in the commutative spectral triple

This means the entire background field method of 2107.08485, which splits D -> D + V with V an algebra element, does **not** apply to metric perturbations.

**Obstruction 5: The Counterterm Structure Problem.**
Even if one could define a spectral perturbation theory for gravity, the counterterms would need to be of the form Tr(g(D^2/Lambda^2)) for some shifted spectral function g. The spectral action at dimension 2n generates the Seeley-DeWitt coefficient a_{2n}, which is a **specific linear combination** of curvature invariants determined by the spin representation. The question is whether ALL counterterms at ALL loop orders respect this specific linear combination.

At one loop (dimension 4): The counterterm structure of pure gravity is int(alpha R^2 + beta C^2), which can be absorbed by shifting f_0 and f_2 (two spectral parameters vs. two gravitational parameters). This WORKS.

At two loops (dimension 6): The counterterms involve three independent R^3-type invariants, but the spectral action at this order introduces one new parameter f_{-2}. The ratio is 3:1 — spectral renormalizability **already fails** at two loops for generic counterterms.

At three loops (dimension 8): The situation worsens to approximately 8:1 (as established by the FUND program: two independent quartic Weyl invariants, reduced from three by the chirality theorem, vs. one new spectral parameter f_{-4}).

### 3B. Where Each van Suijlekom Step Fails for Gravity

| Step | YM | Gravity | Status |
|------|-----|---------|--------|
| Higher-derivative interpretation | phi_Lambda(Delta) regulates the propagator | The gravitational kinetic operator is not of the simple form P = phi(Delta); it contains non-trivial curvature couplings | **Partially OK** (Stelle theory IS a higher-derivative gravity) |
| Power counting -> superrenormalizability | omega < 0 for L >= 2 | Gravity is NOT superrenormalizable even with higher derivatives; it is at best multiplicatively renormalizable (Stelle) or super-renormalizable with N >= 6 derivatives | **Fails** for 4-derivative; **partially OK** for 6+ derivatives |
| BRST cohomology -> single counterterm form | delta_Z int F^2 only | Multiple independent curvature invariants at each order | **Fails** at dimension >= 6 |
| Spectral absorption f -> f + delta_f | f_0 -> (1+delta_Z)f_0 | Would need one spectral parameter per independent curvature invariant per order | **Fails** at L >= 2 |
| Ward identity for divided differences | Based on (z-D)^{-1} | D depends on the dynamical variable g | **Fails** (no fixed spectrum) |


## 4. What Tools Exist for the Gravity Extension

### 4A. Background Field Method for Spectral Action
**Partially available.** One can expand D[g_0 + h] around a background g_0 and compute the one-loop determinant. This is what Mistry-Pinzul-Rachwal (2001.05975) do at the classical level. The one-loop effective action is:

W[g_0] = (1/2) ln det (P_{g_0} P^{-1})

where P_{g_0} is the kinetic operator on the graviton field h_{mu nu}. This gives the standard heat kernel counterterms. At one loop, the counterterms are a_4(P_{g_0}), which **are** of spectral form (they come from the Seeley-DeWitt expansion). But this is not specific to the spectral action -- it holds for ANY covariant gravitational action.

### 4B. Higher-Derivative Gravity (Stelle, Asorey-Lopez-Shapiro)
**Well-developed.** The spectral action naturally generates a higher-derivative gravity theory. Stelle (1977) proved that 4-derivative gravity (R + R^2 + C^2) is multiplicatively renormalizable. Asorey-Lopez-Shapiro (1997) proved that 6-derivative gravity is super-renormalizable (finite above 3 loops). The spectral action with cutoff at a_6 gives a specific 6-derivative theory with rigid coefficients.

**Key point:** The spectral action's rigid coefficient structure may be MORE restrictive than needed for renormalizability, which means that generic counterterms may NOT be of spectral form.

### 4C. Iseppi's BV Construction for Finite Spectral Triples
**Available for finite geometries.** Iseppi (2410.11823) has shown that the full BV construction (ghost/anti-ghost fields, BRST complex) can be formulated within NCG for finite spectral triples. The classical BV and BRST complexes coincide with the Hochschild complex of a coalgebra. This is a promising direction but has not been extended to the gravitational sector.

### 4D. Connes-Kreimer Hopf Algebra
**Available for gauge theories.** Van Suijlekom (0807.0999) showed that the Connes-Kreimer renormalization Hopf algebra is compatible with BRST/Slavnov-Taylor identities for YM theories. The classical master equation in the BV algebra implies Hopf ideals. This framework could in principle be extended to gravity, but the non-compactness of Diff(M) and the non-polynomial structure of the gravitational action are major obstacles.

### 4E. Barvinsky-Vilkovisky Covariant Perturbation Theory
**Standard tool for quantum gravity.** The Barvinsky-Vilkovisky technique computes the one-loop effective action covariantly using the heat kernel. It gives the standard results for counterterms in higher-derivative gravity. This overlaps with but is distinct from the spectral action approach.

### 4F. Chamseddine's Matrix Form (2511.05909, CIS 2020)
**Available for gauge + Higgs sector on flat background.** Chamseddine-Iliopoulos-van Suijlekom (2020) derived matrix-level Feynman rules that preserve the NCG structure (keeping internal indices untraced). This gives ribbon-graph structure to the perturbation theory. However, gravity is turned OFF (flat background). Extending to curved backgrounds is listed as "a natural next step" but has not been done.

### 4G. Non-Propagation at High Energy (Kurkov-Lizzi-Vassilevich)
**Relevant physical input.** The result that "high energy bosons do not propagate" (1312.2235) — the spectral action's quadratic form has no positive derivative powers at high momentum — suggests a natural UV regularization. Alkofer-Saueressig-Zanusso (1410.7999) confirm D_S = 0 at high energy for all spins. This may provide the physical mechanism for UV finiteness but does not constitute a proof.

### 4H. Chirality Theorem (SCT A8, project-internal)
**Proven in this project.** The chirality theorem establishes that tr(a_8) = c(p^2 + q^2) on Ricci-flat backgrounds, reducing the quartic Weyl invariants from 3 to an effective 1:1 ratio. This is relevant because it reduces the dimension of the counterterm space at three loops. However, it does not eliminate the obstruction entirely.


## 5. Assessment: Is the Proof ACHIEVABLE with Current Methods?

### 5A. What IS provable:

1. **One-loop spectral renormalizability of the gauge sector:** PROVEN (van Nuland-van Suijlekom 2021). The counterterms for the inner-fluctuation sector (gauge + Higgs) at one loop are of spectral form.

2. **Superrenormalizability of the YM spectral action:** PROVEN (van Suijlekom 2011). Only one-loop divergences exist.

3. **One-loop spectral form for pure gravity:** ACHIEVABLE with existing tools. At one loop, the gravitational counterterms are a_4 of the graviton kinetic operator, which IS expressible in terms of spectral data. The two independent counterterms (R^2 and C^2 on-shell) can be absorbed by shifting f_0 and f_2.

4. **Power counting for the spectral action matrix model:** PROVEN (Hekkelman-van Nuland-Reimann 2025). The degree of divergence is omega = U + (p/d)(E_fi - V_fi), with planar dominance.

### 5B. What is NOT provable with current methods:

1. **Multi-loop spectral renormalizability of gravity:** NOT achievable. At L >= 2, the number of independent curvature invariants exceeds the number of spectral parameters. No known mechanism (algebraic, cohomological, or spectral) reduces this mismatch.

2. **Background-independent spectral quantization of gravity:** NOT available. The divided-difference framework requires a fixed spectral background. No one has proposed a satisfactory way to quantize the metric sector within the spectral framework.

3. **All-orders Ward identity for diffeomorphisms:** NOT derived. The Ward identities of 2107.08485 are for inner fluctuations (gauge transformations). The analogous identity for diffeomorphisms would require a fundamentally different structure.

### 5C. Overall Assessment

**The proof of spectral renormalizability for the full gravitational spectral action is NOT achievable with current methods.** The obstruction is structural: it lies in the mismatch between the one-parameter-per-order structure of the spectral function and the multi-parameter structure of gravitational counterterms at dimension >= 6.

**However**, there are three scenarios where a modified statement might be provable:

**(i) Spectral renormalizability of the gauge + Higgs sector on a fixed gravitational background:** This is essentially what van Nuland-van Suijlekom proved at one loop. Extending to all orders for the gauge sector (keeping gravity classical) is the most tractable open problem.

**(ii) One-loop spectral renormalizability of the full theory (gravity + gauge):** At one loop, the counterterms for both gravity and gauge are of spectral form. This is provable with existing tools (background field method + heat kernel) but requires careful treatment of the graviton-gauge mixing.

**(iii) Spectral renormalizability in the SCT sense:** If the spectral action's non-perturbative form Tr(f(D^2/Lambda^2)) somehow constrains the counterterms beyond what the asymptotic expansion suggests (e.g., through entireness properties of the form factors), then the obstruction might be evaded. This is speculative but aligns with the SCT program's findings on entire form factors (NT-2).


## 6. The Minimal Axiom

### 6A. Candidates for the Minimal Additional Axiom

**Candidate A: Spectral BRST Identity for Diffeomorphisms.**
A divided-difference Ward identity of the form:

<<h_1,...,xi h_j,...,h_n>>^{1L} - <<h_1,...,h_{j-1} xi,...,h_n>>^{1L} = <<h_1,...,L_xi g,...,h_n>>^{1L}

where xi is a vector field and L_xi g is the Lie derivative, would be the natural gravitational analog of the van Nuland-van Suijlekom Ward identity. **No one has derived this.**

**Candidate B: Non-Perturbative Spectral Closure.**
Axiom: The non-perturbative spectral action Tr(f(D^2/Lambda^2)) is closed under renormalization as a FUNCTIONAL of D (not just via the asymptotic expansion). This means:

Tr(f(D^2/Lambda^2)) + counterterms = Tr(g_L(D^2/Lambda^2))

for some shifted spectral function g_L at each loop order L. This is stronger than what the Seeley-DeWitt expansion suggests and would require the entire form factors to play a role.

**Candidate C: Chirality-Enhanced Spectral Closure.**
Combining the chirality theorem (tr(a_n) = f(p) + f(q)) with the spectral action's specific coefficient structure, the counterterms at each order might be constrained to have the same chiral 1:1 ratio. If true, this reduces the counterterm space at each order and might bring it within the spectral function's parameter count.

**Candidate D: Fakeon Projection.**
If the ghost poles of the gravitational propagator are treated as fakeons (as in the SCT framework's MR-2 unitarity analysis), the counterterm structure might be modified. The fakeon prescription changes the Feynman rules and could reduce the number of independent counterterms. This is speculative.

### 6B. Assessment of Each Candidate

| Candidate | Plausibility | Difficulty | Status |
|-----------|-------------|------------|--------|
| A: Spectral diff Ward identity | Medium | Very high (no tools exist) | Open |
| B: Non-perturbative closure | Low-Medium | Extremely high (non-perturbative QG) | Open |
| C: Chirality-enhanced closure | Medium-High | High (need full a_8 computation + chirality at all orders) | Partially supported by A8 theorem |
| D: Fakeon projection | Medium | High (need fakeon Feynman rules in spectral framework) | Supported by MR-2 |


## 7. Red Flags and No-Go Results

### 7A. Known Obstructions

1. **Goroff-Sagnotti (1985):** Pure Einstein gravity has a non-vanishing two-loop counterterm proportional to C^3 (Goroff-Sagnotti invariant). This is NOT of spectral form unless f_{-2} is tuned. For the spectral action, this counterterm can in principle be absorbed because a_6 IS generated by the spectral action. But the coefficient is fixed by f, so there is a non-trivial condition.

2. **Three-loop overdetermination (FUND program):** At dimension 8, the number of independent Weyl invariants (2 after chirality theorem) exceeds the number of new spectral parameters (1). This is a structural obstruction to all-orders spectral renormalizability, assuming the Seeley-DeWitt expansion is the correct framework.

3. **UV/IR mixing (Hekkelman-van Nuland-Reimann 2025):** For modes with f'(lambda_i) = 0, the degree of divergence increases. This suggests that the spectral action matrix model may have UV/IR mixing analogous to non-commutative scalar field theory, which could spoil renormalizability.

4. **Van Nuland thesis (2212.06533):** The section "The Spectral Action vs Renormalization" is listed as **"in progress"** (empty in the published thesis). This suggests that even the experts consider this an open problem without a clear path.

### 7B. No-Go Indicators

- **No one** has attempted, let alone proven, spectral renormalizability for the gravitational sector. The literature search across arXiv and web searches found ZERO papers claiming this result.
- The van Suijlekom program explicitly works on **flat backgrounds** (no gravitational dynamics). The extensions to almost-commutative geometries still treat gravity classically.
- The Chamseddine-Iliopoulos-van Suijlekom matrix form (2020) explicitly turns gravity OFF.
- The power counting paper (2512.14581) works in the **matrix model** formulation, which does not directly describe gravitational dynamics.

### 7C. Positive Indicators

- The one-loop result (2107.08485) is **entirely spectral** and works for arbitrary noncommutative geometries, not just YM.
- The chirality theorem reduces the three-loop obstruction from 3:1 to 2:1.
- The non-propagation result (1312.2235) suggests a natural UV cutoff.
- The spectral action's rigid coefficient structure might be a feature, not a bug: if the counterterms respect the same rigidity, spectral renormalizability follows automatically.
- The Connes-Kreimer Hopf algebra framework (0807.0999) provides algebraic tools for encoding Ward identities as Hopf ideals, which could potentially be extended to gravity.


## 8. Summary for the Derivation Agent

### The Question Restated
Does S = Tr(f(D^2/Lambda^2)) belong to a "renormalizably closed" class for the gravitational sector?

### The Literature Answer
**UNKNOWN, with strong evidence against (at the perturbative level).**

- **At one loop:** YES (provable with current methods)
- **At two loops:** UNKNOWN (depends on whether the R^3 counterterms respect the spectral action's coefficient ratios; the Goroff-Sagnotti counterterm IS generated by the spectral action, so this may work)
- **At three loops and beyond:** ALMOST CERTAINLY NO in the perturbative Seeley-DeWitt framework (2:1 mismatch after chirality theorem; 8:1 at sextic level)

### The Path Forward
The ONLY way to prove all-orders spectral renormalizability for gravity is to go **beyond** the Seeley-DeWitt asymptotic expansion and use the **non-perturbative** properties of the spectral action. This requires:

1. Proving that the full non-perturbative spectral action Tr(f(D^2)) is closed under renormalization as a functional of D
2. This likely requires the entire form factor structure (NT-2) and the spectral Ward identity for diffeomorphisms
3. The minimal axiom is some form of **"spectral BRST closure for diffeomorphisms"** — an identity that constrains all counterterms to have the spectral form Tr(g(D^2))

### Critical Input for the Derivation Agent
- The divided-difference Ward identity is the key tool. For gravity, one needs a version where the eigenvalues depend on the dynamical field.
- The chirality theorem (proven in this project) is essential: it reduces the counterterm mismatch.
- The fakeon prescription (MR-2) may play a role: it changes the counterterm structure.
- The Hopf algebra framework (van Suijlekom 2008) provides the algebraic setting for encoding gravitational Ward identities.
- The power counting formula omega = U + (p/d)(E_fi - V_fi) with planar dominance is the state-of-the-art for multi-loop behavior.
