# SCT Verified Results Registry

Machine-readable registry of all established results in Spectral Causal Theory.
Each entry is uniquely identified and cross-referenced to the originating task,
verification method, and confidence level.

**Confidence levels:**
- **proven** --- formally verified (Lean 4) or analytically exact with closed-form proof
- **verified** --- confirmed by 8-layer verification pipeline (100+ digit numerics, dual derivation, triple CAS, property fuzzing)
- **established** --- confirmed by multi-stage verification pipeline with adversarial review
- **conditional** --- result holds under stated assumptions; assumptions not yet fully proven

**Verification methods:**
- `Lean4` --- Lean 4 formal proof (PhysLean/Mathlib4, Aristotle, or SciLean backend)
- `mpmath-100` --- 100-digit mpmath numerical verification
- `dual-derivation` --- two independent derivation methods agree
- `triple-CAS` --- SymPy x GiNaC x mpmath cross-check (12+ digit agreement)
- `literature` --- cross-checked against 3+ independent published sources
- `pytest` --- automated regression test suite
- `numerical-MC` --- Monte Carlo numerical simulation
- `adversarial` --- adversarial verification chain with independent re-derivation

---

## NT-1: Dirac (Spin-1/2) Form Factors

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-001 | $E = -R/4$ (Lichnerowicz endomorphism for Dirac) | `Lean4`, `literature` | proven | NT-1, 4 independent verifications |
| VR-002 | $\hat{P} = E + R/6 = -R/12$ | `mpmath-100`, `dual-derivation` | verified | NT-1 |
| VR-003 | $h_C^{(1/2)}(x) = \frac{3\varphi - 1}{6x} + \frac{2(\varphi - 1)}{x^2}$ | `mpmath-100`, `triple-CAS`, `literature` | verified | NT-1, 274+ checks |
| VR-004 | $h_R^{(1/2)}(x) = \frac{3\varphi + 2}{36x} + \frac{5(\varphi - 1)}{6x^2}$ | `mpmath-100`, `triple-CAS`, `literature` | verified | NT-1, 274+ checks |
| VR-005 | $\beta_W^{(1/2)} = h_C^{(1/2)}(0) = 1/20$ | `mpmath-100`, `Lean4` | proven | NT-1 |
| VR-006 | $\beta_R^{(1/2)} = h_R^{(1/2)}(0) = 0$ (conformal invariance) | `mpmath-100`, `Lean4` | proven | NT-1 |
| VR-007 | $x \cdot h_C^{(1/2)}(x \to \infty) = 1/6$ | `mpmath-100`, `dual-derivation` | verified | NT-1 |
| VR-008 | $x \cdot h_R^{(1/2)}(x \to \infty) = 1/18$ | `mpmath-100`, `dual-derivation` | verified | NT-1 |
| VR-009 | $a_4(\text{Dirac}) = -18\,C^2 + 11\,E_4$ (fourth SD coefficient) | `literature`, `dual-derivation` | verified | NT-1, 3 sources |

---

## NT-1b Phase 1: Scalar (Spin-0) Form Factors

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-010 | $h_C^{(0)}(x) = \frac{1}{12x} + \frac{\varphi - 1}{2x^2}$ | `mpmath-100`, `triple-CAS`, `literature` | verified | NT-1b P1, 142/142 checks |
| VR-011 | $h_R^{(0)}(x;\xi) = f_{R,\mathrm{bis}} + \xi\, f_{RU} + \xi^2\, f_U$ | `mpmath-100`, `triple-CAS` | verified | NT-1b P1, 142/142 checks |
| VR-012 | $\beta_W^{(0)} = 1/120$ ($\xi$-independent) | `mpmath-100`, `Lean4` | proven | NT-1b P1 |
| VR-013 | $\beta_R^{(0)}(\xi) = \tfrac{1}{2}(\xi - 1/6)^2$ | `mpmath-100`, `Lean4` | proven | NT-1b P1 |
| VR-014 | $\beta_R^{(0)}(\xi = 1/6) = 0$ (conformal scalar) | `Lean4` | proven | NT-1b P1 |

---

## NT-1b Phase 2: Vector (Spin-1) Form Factors

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-015 | $h_C^{(1)}(x) = \frac{\varphi}{4} + \frac{6\varphi - 5}{6x} + \frac{\varphi - 1}{x^2}$ | `mpmath-100`, `triple-CAS`, `literature` | verified | NT-1b P2, 1519 checks |
| VR-016 | $h_R^{(1)}(x) = -\frac{\varphi}{48} + \frac{11 - 6\varphi}{72x} + \frac{5(\varphi - 1)}{12x^2}$ | `mpmath-100`, `triple-CAS`, `literature` | verified | NT-1b P2, 1519 checks |
| VR-017 | $\beta_W^{(1)} = 1/10$ (gauge + 2 FP ghosts) | `mpmath-100`, `Lean4` | proven | NT-1b P2 |
| VR-018 | $\beta_R^{(1)} = 0$ (Maxwell conformal in $d=4$) | `mpmath-100`, `Lean4` | proven | NT-1b P2 |
| VR-019 | Ghost subtraction: 2 FP ghosts (not 1) | `literature` | established | NT-1b P2, Parker-Toms, CZ, Vassilevich |

---

## NT-1b Phase 3: Combined Standard Model

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-020 | $\alpha_C = 13/120$ (total Weyl${}^2$, $\xi$-independent, parameter-free) | `mpmath-100`, `Lean4`, `literature` | proven | NT-1b P3, 354/354 checks |
| VR-021 | $\alpha_R(\xi) = 2(\xi - 1/6)^2$ (total $R^2$) | `mpmath-100`, `Lean4` | proven | NT-1b P3 |
| VR-022 | $c_1/c_2 = -1/3 + 120(\xi - 1/6)^2/13$ | `mpmath-100`, `Lean4` | proven | NT-1b P3 |
| VR-023 | $3c_1 + c_2 = 6(\xi - 1/6)^2$ (scalar decoupling at $\xi = 1/6$) | `Lean4`, `mpmath-100` | proven | NT-1b P3 |
| VR-024 | $c_2 = 2\alpha_C = 13/60$ | `Lean4` | proven | NT-1b P3 |
| VR-025 | $x \cdot \alpha_C(x \to \infty) = -89/12$ (UV asymptotic) | `mpmath-100`, `dual-derivation` | verified | NT-1b P3 |
| VR-026 | $F_1(0) = 13/(1920\pi^2) \approx 6.860 \times 10^{-4}$ | `mpmath-100` | verified | NT-1b P3 |
| VR-027 | $F_2(0, \xi = 1/6) = 0$ (exact) | `mpmath-100`, `Lean4` | proven | NT-1b P3 |
| VR-028 | SM counting: $N_s = 4$, $N_D = 22.5$, $N_v = 12$ | `literature` | established | CPR 0805.2909 |

---

## NT-2: Entire-Function Property

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-029 | $\varphi(x) = e^{-x/4}\sqrt{\pi/x}\;\mathrm{erfi}(\sqrt{x}/2)$ is entire | `dual-derivation`, `mpmath-100` | verified | NT-2, 63/63 checks |
| VR-030 | $\varphi(0) = 1$ | `mpmath-100`, `Lean4` | proven | NT-2 |
| VR-031 | $\varphi'(0) = -1/6$ | `mpmath-100`, `Lean4` | proven | NT-2 |
| VR-032 | Taylor coefficients: $a_n = (-1)^n\, n!/(2n+1)!$ | `mpmath-100`, rational arithmetic | verified | NT-2, checked $n = 0\ldots5$ |
| VR-033 | $a_0 = 1$, $a_1 = -1/6$, $a_2 = 1/60$, $a_3 = -1/840$ | `mpmath-100` | verified | NT-2 |
| VR-034 | $F_1(z)$ is entire (ghost-freedom in Weyl sector) | `dual-derivation`, `mpmath-100` | verified | NT-2 |
| VR-035 | $F_2(z, \xi)$ is entire for all $\xi$ (ghost-freedom in $R^2$ sector) | `dual-derivation`, `mpmath-100` | verified | NT-2 |
| VR-036 | $\Pi_{\mathrm{entire}}(z) > 0$ for all real $z \ge 0$ | `mpmath-100`, scan $z = 0\ldots100$ | verified | NT-2 |
| VR-037 | $\varphi(x \to \infty) \sim 2/x$ (NOT zero) | `mpmath-100` | verified | NT-2 |

---

## NT-4a: Linearized Field Equations

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-038 | $\Pi_{\mathrm{TT}}(z) = 1 + \tfrac{13}{60}\, z\, \hat{F}_1(z)$ | `mpmath-100`, `dual-derivation` | verified | NT-4a, 88/88 checks |
| VR-039 | $\Pi_s(z, \xi = 1/6) = 1$ for all $z$ (scalar decoupling) | `mpmath-100` | verified | NT-4a |
| VR-040 | Barnes-Rivers projectors: $\mathrm{tr}(P^{(2)}) = 5$, $\mathrm{tr}(P^{(0-s)}) = 1$ | `mpmath-100`, 10 random momenta | verified | NT-4a |
| VR-041 | Gauge invariance: $k^\mu \Pi^{\ldots}_{\mu\nu,\alpha\beta} k^\nu = 0$ | `mpmath-100`, 10 momenta | verified | NT-4a |
| VR-042 | Bianchi identity: $k^\mu G^{\ldots}_{\mu\nu,\alpha\beta} = 0$ | `mpmath-100`, 10 momenta | verified | NT-4a |
| VR-043 | $m_2 = \Lambda\sqrt{60/13} \approx 2.148\,\Lambda$ (spin-2 Yukawa) | `mpmath-100` | verified | NT-4a |
| VR-044 | $m_0(\xi = 0) = \Lambda\sqrt{6} \approx 2.449\,\Lambda$ (scalar Yukawa) | `mpmath-100` | verified | NT-4a |
| VR-045 | $V(r)/V_N(r) = 1 - \tfrac{4}{3}e^{-m_2 r} + \tfrac{1}{3}e^{-m_0 r}$ | `mpmath-100`, `literature` | verified | NT-4a |
| VR-046 | $V(r = 0)$ is finite (singularity resolution) | `mpmath-100` | verified | NT-4a |
| VR-047 | $V(r \to \infty)/V_N(r) \to 1$ (Newtonian recovery) | `mpmath-100` | verified | NT-4a |
| VR-048 | $V(0)/V_N(0) = 0$ (from $1 - 4/3 + 1/3 = 0$) | `Lean4` | proven | NT-4a |
| VR-049 | $c_T = c$ (GW speed equals light speed, exact on FLRW) | `dual-derivation`, `mpmath-100` | verified | NT-4c |

---

## NT-4b: Nonlinear Field Equations

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-050 | Full nonlinear EOM: $(1/\kappa^2)G_{\mu\nu} + 2\alpha_C[F_1 B_{\mu\nu} + \Theta^{(C)}_{\mu\nu}] + \alpha_R[F_2 H_{\mu\nu} + \Theta^{(R)}_{\mu\nu}] = \tfrac{1}{2}T_{\mu\nu}$ | `dual-derivation`, `mpmath-100` | verified | NT-4b, 110 tests |
| VR-051 | Local limit recovers Stelle (1977) gravity | `mpmath-100` | verified | NT-4b |
| VR-052 | Linearized limit recovers NT-4a ($\Pi_{\mathrm{TT}}$, $\Pi_s$ to $< 10^{-25}$) | `mpmath-100` | verified | NT-4b |
| VR-053 | De Sitter is exact solution of nonlinear SCT field equations | `dual-derivation` | verified | NT-4b |
| VR-054 | Bach tensor is traceless; $\mathrm{tr}(H_{\mu\nu}) = -6\Box R$ | `mpmath-100` | verified | NT-4b |

---

## NT-4c: FLRW Cosmology

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-055 | $C_{\mu\nu\rho\sigma} = 0$ on FLRW: entire Weyl sector drops out | `dual-derivation` | verified | NT-4c, 138 tests |
| VR-056 | $c_T = c$ to all orders on FLRW; $|c_T/c - 1| \sim 10^{-123}$ | `mpmath-100` | verified | NT-4c |
| VR-057 | De Sitter is stable under perturbations | `mpmath-100` | verified | NT-4c |
| VR-058 | $w_\Theta = +1$ (stiff matter EOS for nonlocal correction) | `mpmath-100` | verified | NT-4c |
| VR-059 | At $\xi = 1/6$, ALL spectral corrections vanish on FLRW | `mpmath-100` | verified | NT-4c |

---

## NT-3: Spectral Dimension

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-060 | $d_S(\mathrm{IR}) = 4$ for all definitions | `mpmath-100`, `adversarial` | established | NT-3 |
| VR-061 | $d_S$ is definition-dependent in SCT (ML, CMN, ASZ give different UV limits) | `mpmath-100`, `adversarial` | established | NT-3 |
| VR-062 | $P(\sigma) < 0$ for $\sigma < \sigma^* \approx 0.01/\Lambda^2$ (ghost-induced) | `mpmath-100` | verified | NT-3 |
| VR-063 | $\Pi_{\mathrm{TT}}$ saturates at $-83/6$ in Euclidean UV | `mpmath-100` | verified | NT-3 |

---

## MR-2: Unitarity and Stability

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-064 | $\Pi_{\mathrm{TT}}$ has 8 zeros in $|z| \le 100$: 2 real + 3 complex conjugate pairs | `mpmath-100`, `adversarial` | established | MR-2 |
| VR-065 | $z_L \approx -1.2807$ (Lorentzian ghost, timelike), $R_L \approx -0.5378$ | `mpmath-100` | verified | MR-2 |
| VR-066 | $z_0 \approx 2.4148$ (Euclidean pole, spacelike), $R_0 \approx -0.4931$ | `mpmath-100` | verified | MR-2 |
| VR-067 | $\Pi_s(\xi = 1/6) \equiv 1$: scalar sector ghost-free at conformal coupling | `mpmath-100`, `Lean4` | proven | MR-2 |
| VR-068 | No continuum spectral function: $\rho_{\mathrm{TT}}$ is purely delta-function | `dual-derivation` | verified | MR-2 |
| VR-069 | Modified sum rule: $\sum R_n \to -6/83$ (not zero, because $\Pi_{\mathrm{TT}}(\infty) = -83/6$) | `mpmath-100` | verified | MR-2 |

---

## MR-7: Graviton Scattering Amplitudes

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-070 | Tree-level SCT = GR exactly (Modesto-Calcagni field redefinition theorem) | `adversarial`, `literature` | established | MR-7, CERTIFIED |
| VR-071 | One-loop UV divergence degree: $D = 0$ for SCT graviton bubble (vs $D = 4$ for GR) | `mpmath-100`, `dual-derivation` | verified | MR-7 |
| VR-072 | One-loop counterterms: $\delta\alpha_C \cdot C^2 + \delta\alpha_R \cdot R^2$ (already in spectral action) | `dual-derivation` | verified | MR-7 |
| VR-073 | Ward identity: $k^\mu P^{(2)}_{\mu\nu,\alpha\beta} = 0$ to machine precision | `mpmath-100` | verified | MR-7 |
| VR-074 | $\mathrm{Im}[\Sigma] > 0$ (consistent with optical theorem), $C_m = 283/120$ | `mpmath-100` | verified | MR-7 |
| VR-075 | Tree-level breaks down at $a_6$ order: $\sim 1\%$ correction at $\sqrt{s} \sim 1.3\Lambda$ | `mpmath-100` | verified | MR-7 |

---

## CHIRAL-Q: D-Squared Quantization

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-076 | $D = 0$ at every loop order in $D^2$-quantization (algebraic proof: chirality identity) | `mpmath-100`, `Lean4`, `pytest` | proven | CHIRAL-Q, 85/85 tests |
| VR-077 | $D = 0$ unconditionally through two loops (Theorem 6.12) | `dual-derivation`, `adversarial` | established | CHIRAL-Q |
| VR-078 | $D = 0$ at all orders conditional on BV-1..5 (Theorem 6.11) | `dual-derivation` | conditional | CHIRAL-Q, BV-3/4 verified to 1-loop |
| VR-079 | Chirality identity: $\{D, \gamma_5\} = 0 \Rightarrow$ all counterterms block-diagonal | `Lean4` (14 theorems, zero sorry) | proven | CHIRAL-Q |
| VR-080 | Identity holds for $N = 4, 8, 16, 32, 64$ and $L = 1\ldots8$ | `mpmath-100`, `pytest` | verified | CHIRAL-Q |

---

## CL: Commutativity of Limits

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-081 | $\lim_{N\to\infty}[\text{fakeon}(N\;\text{poles})] = \text{fakeon}(\lim_{N\to\infty}[N\;\text{poles}])$ | `dual-derivation`, `adversarial` | established | CL, CERTIFIED |
| VR-082 | Weierstrass bound: $\sum M_n = 5.002 \times 10^{-4}$ (smooth corrections are 0.32% of amplitude) | `mpmath-100` | verified | CL |
| VR-083 | Two-pole dominance: 99.68% of amplitude from $z_L$ and $z_0$ | `mpmath-100` | verified | CL |

---

## GZ: Entire Part of Mittag-Leffler Expansion

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-084 | $g_A(z) = -13/60$ (constant, equal to $-c_2 = -2\alpha_C$) | `mpmath-100`, `adversarial` | established | GZ, CERTIFIED |
| VR-085 | Sum rule: $\sum R_n/z_n = 13/60$ (corollary: $g_A(0) + \sum R_n/z_n = 0$) | `mpmath-100` | verified | GZ |
| VR-086 | $\Pi_{\mathrm{TT}}$ has order $\rho = 1$, genus $p = 1$ | `dual-derivation` | verified | GZ |

---

## PPN-1: Solar System Tests

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-087 | $\gamma_{\mathrm{PPN}} = 1 + \tfrac{2}{3}e^{-m_2 r} + O(e^{-m_0 r})$ | `mpmath-100`, `dual-derivation` | verified | PPN-1, 121 tests |
| VR-088 | At $r = 1$ AU: $|\gamma - 1| \sim e^{-10^{14}} \approx 0$ (indistinguishable from GR) | `mpmath-100` | verified | PPN-1 |
| VR-089 | Eot-Wash bound: $\Lambda \ge 2.38 \times 10^{-3}$ eV (strongest) | `mpmath-100` | verified | PPN-1 |
| VR-090 | Yukawa coupling: $\alpha = -4/3$ (exact, parameter-free) | `dual-derivation` | verified | PPN-1 |
| VR-091 | Mass ratio: $m_2/m_0 = \sqrt{10/13} \approx 0.877$ (parameter-free) | `Lean4` | proven | PPN-1 |
| VR-092 | Short-range deviations at $r \sim 1/\Lambda \sim 38.6\;\mu\text{m}$ (at Eot-Wash boundary) | `mpmath-100` | verified | PPN-1 |

---

## MR-4: Two-Loop Structure

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-093 | $R^3$ counterterm absorbable by $\delta\psi$ (spectral function redefinition) | `dual-derivation` | conditional | MR-4 |
| VR-094 | $G \sim 1/k^2$ correction at two loops | `mpmath-100` | verified | MR-4 |

---

## MR-5: Finiteness

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-095 | $D = 0$ at two loops on-shell (CCC only, unconditional) | `dual-derivation`, `adversarial` | established | MR-5b |
| VR-096 | $D = 0$ at three loops: NO (2 quartic Weyl invariants vs 1 parameter) | `adversarial` | established | MR-5b |
| VR-097 | Spectral moments: $f_{2k} = (k-1)!$ (factorial growth for $\psi = e^{-u}$) | `mpmath-100` | verified | MR-5 |
| VR-098 | Nonperturbative ambiguity: $\sim 1.3 \times 10^{-32}$ (80x larger than optimal truncation) | `mpmath-100` | verified | MR-5 |

---

## MR-3: Causality

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-099 | Macroscopic causality preserved in SCT | `adversarial` | established | MR-3 |
| VR-100 | Microscopic causality violated at scale $\sim 1/\Lambda$ | `adversarial` | established | MR-3 |

---

## INF-1: Spectral Inflation

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-101 | Scalaron mass from one-loop SM: $M = \sqrt{24\pi^2}\, M_P \approx 15.39\, M_P$ (too heavy by factor $\sim 10^6$) | `mpmath-100` | verified | INF-1 |
| VR-102 | Conditional predictions (if $M$ fixed externally): $n_s(N=55) = 0.965$, $r \approx 3.5 \times 10^{-3}$ | `mpmath-100` | conditional | INF-1 |

---

## MT-2: Late-Time Cosmology

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-103 | $\delta H^2/H^2 = 1.28 \times 10^{-64}$ at PPN-1 bound: 64 orders too small for $H_0$ tension | `mpmath-100` | verified | MT-2, 131 tests |
| VR-104 | $w_{\mathrm{eff}} = -1$ to 63 decimal places at late times | `mpmath-100` | verified | MT-2 |
| VR-105 | SCT automatically consistent with ALL late-time cosmological data (corrections vanish) | `dual-derivation` | established | MT-2 |

---

## MT-1: Black Hole Entropy

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-106 | Logarithmic correction to BH entropy: $c_{\log} = 37/24$ | `mpmath-100`, `literature` | verified | MT-1 |

---

## CJ Bridge and Causal Set Results

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-107 | $(d!)^2 \binom{2d}{d}(2d+1) = (2d+1)!$ for all $d \ge 0$ | `Lean4` | proven | CJ Bridge |
| VR-108 | $B(5,5) = \Gamma(5)^2/\Gamma(10) = (4!)^2/9!$ | `Lean4` (Mathlib) | proven | CJ Bridge |
| VR-109 | $\pi^2/45 = (8\pi/15)(\pi/24)$ | `Lean4` | proven | CJ Bridge |
| VR-110 | $I_{\mathrm{bulk}} = (3/10)\, V_4/9!$ (MC-verified, 50 seeds, $0.2999 \pm 0.0003$) | `numerical-MC` | verified | CJ Bridge |
| VR-111 | CJ $= 0$ on de Sitter (50 seeds, $N = 10000$) | `numerical-MC` | verified | CJ Bridge, GTA |
| VR-112 | CJ$/E^2$ is $\epsilon$-independent (CV $= 1.4\%$ for $\epsilon \ge 3$) | `numerical-MC` | verified | CJ Bridge |
| VR-113 | CJ is not a spectral functional: $|r(\mathrm{CJ}, \text{all spectral obs})| < 0.35$ | `numerical-MC` | established | CJ Bridge, 7 routes closed |
| VR-114 | 105 sorry-free Lean 4 theorems in SCTLean | `Lean4` | proven | CJ Bridge + all tasks |

---

## GTA Framework (Causal Set Observables)

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-115 | $b_{\mathrm{eff}} = 5$ (effective screening depth) | `numerical-MC` | established | GTA |
| VR-116 | $\sigma_0 \approx 0.299 \times N^{1/4}$ (CV $= 0.48\%$) | `numerical-MC` | verified | GTA |
| VR-117 | SO(3) Lemma: $E[D_1] = 0$ for any local isotropic vacuum diamond ($l=0 \otimes l=2 \to 0$) | `dual-derivation` | verified | GTA |
| VR-118 | pp-wave: $T^4$ scaling of signal ($\alpha \approx 3.57 \pm 0.21$) | `numerical-MC` | verified | GTA |
| VR-119 | Schwarzschild local $T^4$: ratio$/T^4 = 0.998$ at $T = 0.70$ ($3.5\sigma$, $M=200$) | `numerical-MC` | verified | GTA |
| VR-120 | dS Ricci subtraction = exact zero ($M = 50$, all $T$) | `numerical-MC` | verified | GTA |
| VR-121 | SA (score amplitude) = CJ + Between (law of total covariance, exact decomposition) | `dual-derivation` | proven | GTA |
| VR-122 | Between/SA $= 31\%$, metric-stable (ppw $32\%$, Sch $29\%$) | `numerical-MC` | verified | GTA |
| VR-123 | Magnetic Weyl suppressed: boosted Sch ratio $2.18$ (closer to $E$-only $2.33$ than $E+B$ $3.67$) | `numerical-MC` | verified | GTA |
| VR-124 | Kottler Ricci subtraction match: $0.8\%$ agreement | `numerical-MC` | verified | GTA |

---

## Formal Verification Infrastructure

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-125 | 41/41 canonical identities registered in SCT_IDENTITIES | `Lean4` | proven | All tasks |
| VR-126 | 46 Lean 4 proof files, 44 sorry-free | `Lean4` | proven | All tasks |
| VR-127 | 91 formal verification tests in Lean 4 suite | `Lean4` | proven | All tasks |
| VR-128 | BV canonical transformation formalized in Lean 4 | `Lean4` | proven | CHIRAL-Q |

---

## Properties of the Master Function (Summary)

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-129 | $\varphi(x) = \int_0^1 d\xi\, e^{-\xi(1-\xi)x}$ (parametric representation) | `literature` | proven | Definition |
| VR-130 | $\varphi(z)$ is entire (no poles, no branch cuts in $\mathbb{C}$) | `dual-derivation` | verified | NT-2 |
| VR-131 | $\varphi(0) = 1$, $\varphi'(0) = -1/6$ | `Lean4`, `mpmath-100` | proven | NT-2 |
| VR-132 | $\varphi(x > 0) > 0$ (positive on positive real axis) | `mpmath-100`, scan | verified | NT-2 |
| VR-133 | Growth order $\rho = 1$ (order of $\varphi$ as entire function) | `dual-derivation` | verified | GZ |

---

## Cross-Cutting Results

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-134 | SCT passes all solar system tests trivially ($\Lambda \ge 2.38 \times 10^{-3}$ eV) | `mpmath-100`, `adversarial` | established | PPN-1 |
| VR-135 | SCT passes GW170817 ($c_T = c$ exactly on FLRW) | `mpmath-100` | verified | NT-4c |
| VR-136 | SCT is an effective framework through $L = 2$; UV-completeness not achieved | `adversarial` | established | MR-5, MR-5b, FUND |
| VR-137 | Survival probability of core framework: 87--93% | `adversarial` | established | All tasks, 4196+ tests |

---

## OP-20: De Sitter Conjecture Check (2026-03-31)

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-138 | Refined dS conjecture VIOLATED for SCT scalaron: $g(\phi) \to 0$, $\eta(\phi) \to 0^-$ on plateau | `mpmath-100`, `dual-derivation` | verified | OP-20 |
| VR-139 | $\eta_{\min} = -1/3$ at $\phi = M_{\rm Pl}\sqrt{3/2}\ln 3 \approx 1.346\,M_{\rm Pl}$ (hard ceiling on $c_2$) | `mpmath-100` | verified | OP-20 |
| VR-140 | $\phi_*(c_1=1) = M_{\rm Pl}\sqrt{3/2}\ln(1+2\sqrt{2/3}) \approx 1.186\,M_{\rm Pl}$ (gradient boundary) | `mpmath-100` | verified | OP-20 |
| VR-141 | $M_0$ cancels in $|V'|/V$ and $V''/V$: Swampland check is mass-independent | `dual-derivation` | verified | OP-20 |
| VR-142 | $\phi_{\rm inflection} = M_{\rm Pl}\sqrt{3/2}\ln 2 \approx 0.849\,M_{\rm Pl}$ ($V''$ sign change) | `mpmath-100` | verified | OP-20 |

---

## OP-44: Critical Coupling Resolution (2026-03-31)

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-143 | $\xi = 1/6$ is a PREDICTION of the standard Chamseddine-Connes spectral action (not a free parameter) | `literature` (5 groups) | established | OP-44 |
| VR-144 | Printed $\xi_0 = 1/12$ in CC-convention $\frac{1}{2}|DH|^2$ equals $\xi = 1/6$ in standard $|D\Phi|^2$ convention | `dual-derivation` | verified | OP-44, hep-th/0610241 |
| VR-145 | $\beta_\xi = (\xi - 1/6) \times [\ldots]/(16\pi^2)$: $\xi = 1/6$ is exact one-loop RG fixed point | `literature`, `mpmath-100` | established | OP-44, 1403.4226 |
| VR-146 | Standard Higgs inflation ($\xi \sim 5 \times 10^4$) is incompatible with plain spectral action | `literature` | established | OP-44, 0710.3755 |

---

## OP-04: Parameter Counting Resolution (2026-04-01)

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-147 | Robust SCT core = $a_4$-level only: $\alpha_C$, $c_1/c_2$, PPN $\gamma=\beta=1$, $c_T=c$ on FLRW | `literature`, `dual-derivation` | established | OP-04 |
| VR-148 | All finite-momentum predictions ($h_C$, $h_R$, $m_2$, $m_0$, $V(r)$, $d_S$, QNMs) are $f$-shape-dependent | `literature`, `dual-derivation` | established | OP-04 |
| VR-149 | Entireness of form factors requires $f$ to be entire (not just rapidly decreasing). $e^{-u^{3/2}}$: NOT entire. | `mpmath-100` | verified | OP-04 |
| VR-150 | $f(u) = e^{-u}$ is NOT uniquely selected by \{entireness + unitarity + causality\}; $e^{-u^2}$, $e^{-u^3}$ also admissible | `literature`, `mpmath-100` | established | OP-04 |
| VR-151 | Mass variation across entire cutoffs: $m_2/\Lambda \in [2.02, 2.15]$ for $\alpha = 1, 2, 3$ (5\% range) | `mpmath-100` | verified | OP-04 |
| VR-152 | Power-law $f(u)=(1+u)^{-N}$: $\varphi_N(x) = {}_2F_1(N,1;3/2;-x/4)$, branch cut at $x=-4$, excluded by entireness | `mpmath-100` | verified | OP-04 |
| VR-153 | UV asymptotic $x \cdot \alpha_C(x \to \infty) = -89/12$ is specific to $f = e^{-u}$, not universal | `dual-derivation` | established | OP-04 |

---

## OP-17: Scalaron Mass Resolution (2026-04-01)

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-154 | No known mechanism within standard NCG spectral action produces $M_0 = M_{\rm inf}$ | `literature` (11 papers) | established | OP-17 |
| VR-155 | ALL NCG BSM scalars have conformal coupling $\xi' = 1/6$: contributes $(\xi'-1/6)^2 = 0$ to $\alpha_R$ | `literature`: Eqs.(52) 1005.1188, (129) 1304.8050, (4.10) 1801.00260 | established | OP-17 |
| VR-156 | At NCG-predicted $\xi = 1/6$: $\alpha_R = 0$, $m_0 \to \infty$, scalaron is entirely absent | `mpmath-100`, `dual-derivation` | verified | OP-17, OP-44 |
| VR-157 | Large $\xi \sim 2.25 \times 10^4$ needed for $M_0 = M_{\rm inf}$; conflicts with Eq.(49) of 1005.1188 | `mpmath-100`, `literature` | established | OP-17 |
| VR-158 | Surviving path: sub-Planckian $\Lambda \sim 10^{12}\text{--}10^{13}$ GeV; conflicts with standard GUT interpretation | `literature` | conditional | OP-17 |

---

## OP-30: Running Constants Resolution (2026-04-01)

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-159 | $\beta_{c_2}^{\rm SM} = 283/[60(4\pi)^2] \approx 0.02987$ (universal matter one-loop) | `literature` (CPR 0805.2909) | established | OP-30 |
| VR-160 | $\beta_{c_1}^{\rm SM} = -283/[180(4\pi)^2] \approx -0.00996$ at $\xi = 1/6$ | `literature`, `mpmath-100` | established | OP-30 |
| VR-161 | $c_1/c_2 = -1/3$ exactly preserved along matter-only one-loop trajectory at $\xi = 1/6$ | `dual-derivation`, `mpmath-100` | verified | OP-30 |
| VR-162 | Perturbative AF ratio $(c_1/c_2)_{\rm AF} \approx -0.3257$, 2.3\% from SCT | `literature` (Shapiro hep-th/0412249 Table 2) | established | OP-30 |
| VR-163 | BMS NGFP ratio $(c_1/c_2)_{\rm NGFP} \approx -1.087$, 226\% from SCT | `literature` (BMS 0901.2984 eq.(12)) | established | OP-30 |
| VR-164 | $c_2$ zero crossing at $\mu/\Lambda \approx 7.07 \times 10^{-4}$ (matter-only) | `mpmath-100` | verified | OP-30 |

---

## OP-31: Form Factor Comparison SCT vs AS (2026-04-01)

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-165 | CZ basis $f_C(x) = \frac{1}{2}f_{\rm Ric}(x)$ is IDENTICAL to SCT $h_C^{(0)}(x)$ for scalar Weyl form factor | `literature` (CZ 1203.2034 eq.(5.2),(5.11)) | established | OP-31 |
| VR-166 | SCT and AS matter one-loop nonlocal form factors coincide (same CZ master function) | `literature`, `dual-derivation` | established | OP-31 |
| VR-167 | Graviton loop contribution: $\alpha_C^{\rm grav} = 7/20$, $\alpha_R^{\rm grav} = 1/4$ (absent in SCT) | `literature` (SCM 1006.3808 eq.(28)) | established | OP-31 |
| VR-168 | Full AS one-loop $\alpha_C = 11/24 \approx 0.458$ vs SCT $13/120 \approx 0.108$ (factor 4.23) | `mpmath-100` | verified | OP-31 |
| VR-169 | SCT and AS belong to different UV universality classes: entire-function vs power-law | `literature` (KRS 2111.12365 eqs.(6)-(9)) | established | OP-31 |
| VR-170 | Form factor shape divergence: $z_{10\%} \approx 0.20$, $z_{\mathcal{O}(1)} \approx 1.8$ after IR matching | `literature` (BKS 1904.04845 eq.(12)) | established | OP-31 |

---

## OP-16: Gevrey Loop Expansion (Partial, 2026-04-02)

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-196 | Standard Gevrey-1 means $\Gamma(L+b)$ ($L!$-type), NOT $(2L)!$. Growth $(2L)!$ is Gevrey-2. | `literature` (Le Guillou-Zinn-Justin 1990) | established | OP-16 |
| VR-197 | Lipatov saddle-point gives $\Gamma(L+b)$ for any action with standard $e^{-S/g}$ exponential; entire form factors change $A$ not Gevrey index | `literature` (Lipatov 1977 eqs.(75)-(76)) | established | OP-16 |
| VR-198 | Non-standard SCT UV ($\Pi_{TT} \to$ const) removes renormalon mechanism but doesn't create new factorial source | `literature` (Beneke 1999) | established | OP-16 |
| VR-199 | No explicit Gevrey-class proof exists even for Stelle gravity | `literature` (surveyed 10+ papers) | established | OP-16 |

---

## OP-12: Kramers-Kronig Dispersion Relations (2026-04-02)

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-191 | Standard KK does NOT survive at 2+ loops in SCT; analyticity replaced by average continuation | `literature` (2109.06889, 1801.00915, 2503.01841) | established | OP-12 |
| VR-192 | ALL thresholds containing at least one fakeon frequency are removed from $\mathrm{Im}[\Sigma]$ | `literature` (2109.06889 §2.5 step 5, eqs.(2.16),(2.18)) | established | OP-12 |
| VR-193 | Mixed ghost+matter threshold $E \sim m_{\rm ghost}+m_{\rm phys}$: dead as absorptive cut; may survive as regionwise analyticity boundary | `literature` (2109.06889 eqs.(4.9)-(4.10),(7.8)-(7.13)) | established | OP-12 |
| VR-194 | Spectral optical theorem eq.(2.10) survives with fakeon projection: cut propagators vanish eq.(2.13), PV eq.(2.14) | `literature` (2109.06889) | established | OP-12 |
| VR-195 | For sunset: if even one of three internal lines is fakeon, entire connected cut is contaminated and removed | `dual-derivation` from graph structure + eq.(2.16) | established | OP-12 |

---

## OP-24: Spectral Dimension Definition Dependence (2026-04-02)

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-186 | $d_S$ is probe-dependent: 6 QG programs use 6 structurally different definitions | `literature` (6 programs, 10+ papers) | established | OP-24 |
| VR-187 | Physical fakeon-projected SCT spectral dimension $d_S^{\rm phys} = 4$ at all scales | `dual-derivation`, `literature` | established | OP-24 |
| VR-188 | "All QG predict $d_S \to 2$" is weak: many Euclidean probes give 2, but they measure different quantities | `literature` (Carlip 1705.05417) | established | OP-24 |
| VR-189 | $P(\sigma) < 0$ below $\sigma^*$ is not unique to SCT; Calcagni et al. (1304.7247) show many QG kernels are non-positive | `literature` | established | OP-24 |
| VR-190 | Causal set BBL $d_S$ (1507.00330) uses scalar operator, not tensor $\Pi_{TT}$; not directly transferable to SCT | `literature` | established | OP-24 |

---

## OP-11: IVP Well-Posedness (Partial, 2026-04-01)

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-182 | CMN diffusion-localization requires polynomial $H(\Box)$; fails for entire (§2.6 of 1803.00561) | `literature` | established | OP-11 |
| VR-183 | Even ghost-free IDG: solution fails to exist in $\mathcal{S}'$ for generic sources (Heredia 2112.05397 cond.(87)) | `literature` | established | OP-11 |
| VR-184 | Anselmi-Calcagni classicization (2510.05276): finite kernel $\Rightarrow$ 2 initial conditions; infinite kernel = open | `literature` | established | OP-11 |
| VR-185 | Malgrange-Ehrenpreis theorem: polynomial $P(D)$ only, does not apply to entire operators | `literature` | established | OP-11 |

---

## OP-07: Fakeon Infinite Poles (Partial, 2026-04-01)

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-178 | Anselmi's proof explicitly assumes finite number of thresholds: 1801.00915 after eq.(2.25) | `literature` | established | OP-07 |
| VR-179 | Main gap = type (a): finite iteration over thresholds. CL bound addresses this at 1-loop. | `literature`, `dual-derivation` | established | OP-07 |
| VR-180 | IDG unitarity theorems do NOT transfer to SCT: require no extra poles (SCT has infinitely many) | `literature` | established | OP-07 |
| VR-181 | Kubo-Kugo (2308.09006): unitarity violated for Lee-Wick complex ghosts, but NOT for fakeons | `literature` | established | OP-07 |

---

## OP-15: a₆ Tensor Structure (Partial, 2026-04-01)

| ID | Statement | Method | Confidence | Source |
|----|-----------|--------|------------|--------|
| VR-171 | Full universal $a_6$ for Laplace-type operator: Vassilevich eq.(4.29), Gilkey Thm 4.8.16 | `literature` | established | OP-15 |
| VR-172 | Standard 4D integral basis has 10 (not 8) independent dimension-6 invariants after IBP | `literature` (DF 0706.0691) | established | OP-15 |
| VR-173 | Goroff-Sagnotti $209/2880$ confirmed: GS eq.(1.2) and van de Ven eq.(6.30) | `literature` | established | OP-15 |
| VR-174 | On-shell 4D reduction to single parity-even $C^3$: GS eqs.(3.12)-(3.14) | `literature` | established | OP-15 |
| VR-175 | Two independent quartic Weyl invariants in $d=4$: Moura eqs.(3.5)-(3.6) | `literature` | established | OP-15 |
| VR-176 | Spectral action 6-derivative sector under one coefficient $\mu_3$: Mistry eq.(3.7) | `literature` | established | OP-15 |
| VR-177 | DF on-shell CCC ratios: scalar/Dirac=$-1/2$, vector/Dirac=$-3/2$ (in standard heat-kernel normalization) | `literature`, `mpmath-100` | verified | OP-15 |

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Total verified results | 199 |
| Formally proven (Lean 4) | 30+ |
| Numerically verified (100-digit) | 90+ |
| Dual-derivation confirmed | 40+ |
| Literature cross-checked | 15+ |
| Conditional results | 4 |
| Total automated tests across project | 4196+ |
| Total compiled PDF artifacts | 73/73 |
