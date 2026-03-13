# Axiom 2 -- Causality as Foundation

## 1. Formal Statement

**Axiom 2 (Causal Structure).** The elements of the causal set $\mathcal{C}$ (defined in Axiom 1) are connected by a partial order $\preceq$ that constitutes the fundamental causal relation. This relation satisfies:

**(a) Causal partial order.** $(\mathcal{C}, \preceq)$ is a locally finite partially ordered set (poset):

$$
\begin{aligned}
&\text{(i)}\quad x \preceq x \quad &\forall\, x \in \mathcal{C} \quad &(\text{reflexivity}) \\
&\text{(ii)}\quad x \preceq y \;\wedge\; y \preceq x \;\Rightarrow\; x = y \quad &\forall\, x, y \in \mathcal{C} \quad &(\text{antisymmetry}) \\
&\text{(iii)}\quad x \preceq y \;\wedge\; y \preceq z \;\Rightarrow\; x \preceq z \quad &\forall\, x, y, z \in \mathcal{C} \quad &(\text{transitivity}) \\
&\text{(iv)}\quad |\mathrm{Alex}(x, z)| < \infty \quad &\forall\, x \preceq z \quad &(\text{local finiteness})
\end{aligned}
$$

where $\mathrm{Alex}(x, z) = \{y \in \mathcal{C} : x \preceq y \preceq z\}$ is the Alexandrov interval (causal diamond).

**(b) Lorentz invariance via Poisson sprinkling.** When $\mathcal{C}$ faithfully embeds into a Lorentzian manifold $(\mathcal{M}, g_{\mu\nu})$, the embedding is defined through a Poisson process with density $\rho$:

$$
\mathrm{Prob}(n \text{ elements in region } \mathcal{R}) = \frac{(\rho \cdot \mathrm{Vol}_g(\mathcal{R}))^n}{n!}\, e^{-\rho \cdot \mathrm{Vol}_g(\mathcal{R})}
$$

where $\mathrm{Vol}_g(\mathcal{R}) = \int_{\mathcal{R}} \sqrt{-g}\, d^4x$ is the spacetime volume. The fundamental density is at the Planck scale: $\rho \sim l_P^{-4}$.

**(c) Spectral decoration of causal links.** Each element $x \in \mathcal{C}$ carries a spectral triple $(\mathcal{A}_x, \mathcal{H}_x, D_x)$. The Dirac operator $D_x$ determines local geometry at $x$. The causal relation $x \preceq y$ is compatible with the spectral data:

$$
x \prec y \;\Rightarrow\; d_D(x, y) \geq 0 \quad \text{and} \quad \sigma^2(x, y) \leq 0
$$

where $d_D$ is the Connes spectral distance and $\sigma^2(x, y)$ is the Synge world function reconstructed from spectral data.

---

## 2. Physical Motivation

### 2.1. Causality as More Fundamental than Geometry

In general relativity, the causal structure $J^+(p)$ (causal future of $p$) and the volume element $\sqrt{-g}\,d^4x$ together determine the spacetime metric up to a conformal factor (Malament's theorem, 1977). Hawking, King, and McCarthy (1976) showed that the causal structure alone determines the topology, differentiable structure, and conformal geometry.

SCT elevates this mathematical observation to a physical postulate: **causal order is ontologically prior to geometry.** The metric, curvature, and all continuum structures are emergent, derived quantities.

### 2.2. Lorentz Invariance Without a Background

A major challenge for any discrete approach to spacetime is preserving Lorentz invariance, which is a continuous symmetry. A regular lattice breaks Lorentz invariance by selecting preferred directions. The Poisson sprinkling process resolves this: since a Poisson process on a Lorentzian manifold is invariant under all volume-preserving diffeomorphisms (and Lorentz transformations are volume-preserving), the resulting causal set is statistically Lorentz invariant. No preferred frame is introduced.

### 2.3. Built-in UV Cutoff

The local finiteness condition $|\mathrm{Alex}(x, z)| < \infty$ ensures that spacetime is fundamentally discrete at the Planck scale. This provides a natural ultraviolet cutoff without breaking the symmetries of the theory, unlike a lattice cutoff. The number of elements in a causal diamond is proportional to its spacetime volume in Planck units:

$$
|\mathrm{Alex}(x, z)| \;\approx\; \rho \cdot \mathrm{Vol}_g(\mathrm{Alex}(x, z))
$$

---

## 3. Mathematical Definitions

### 3.1. Causal Relations

Given the partial order $\preceq$, we define the following relations:

- **Causal precedence:** $x \preceq y$ (read "$x$ causally precedes $y$" or "$y$ is in the causal future of $x$").
- **Strict causal precedence:** $x \prec y$ iff $x \preceq y$ and $x \neq y$.
- **Causal link:** $x \prec\!\!\!\cdot\; y$ (read "$x$ is linked to $y$") iff $x \prec y$ and there is no $z \in \mathcal{C}$ with $x \prec z \prec y$. Links are nearest-neighbor causal relations.
- **Spacelike separation:** $x \sim y$ iff neither $x \preceq y$ nor $y \preceq x$.
- **Causal future:** $J^+(x) = \{y \in \mathcal{C} : x \preceq y\}$.
- **Causal past:** $J^-(x) = \{y \in \mathcal{C} : y \preceq x\}$.
- **Causal diamond (Alexandrov interval):** $\mathrm{Alex}(x, y) = J^+(x) \cap J^-(y)$ for $x \preceq y$.

### 3.2. Faithful Embedding

A causal set $(\mathcal{C}, \preceq)$ admits a **faithful embedding** into a Lorentzian manifold $(\mathcal{M}, g_{\mu\nu})$ if there exists an injective map $\Phi: \mathcal{C} \hookrightarrow \mathcal{M}$ such that:

1. **Order-preserving:** $x \preceq y$ in $\mathcal{C}$ $\iff$ $\Phi(x) \in J^-(\Phi(y))$ in $(\mathcal{M}, g)$.
2. **Density condition:** The image $\Phi(\mathcal{C})$ is distributed according to a Poisson process with density $\rho$ on $(\mathcal{M}, g)$.

Not every causal set admits a faithful embedding. Those that do are called **manifold-like** at scale $\rho^{-1/4}$.

### 3.3. Poisson Sprinkling

The **Poisson sprinkling** $\mathfrak{P}(\mathcal{M}, g, \rho)$ is a random process that generates a causal set from a Lorentzian manifold:

1. Draw points from a Poisson process on $(\mathcal{M}, g)$ with intensity measure $\rho \cdot \mathrm{dVol}_g$.
2. Define the partial order $\preceq$ by restricting the causal order of $(\mathcal{M}, g)$ to the sprinkled points.
3. The expected number of elements in a region $\mathcal{R}$ is $\langle N_\mathcal{R} \rangle = \rho \cdot \mathrm{Vol}_g(\mathcal{R})$, with Poisson fluctuations $\delta N_\mathcal{R} \sim \sqrt{\rho \cdot \mathrm{Vol}_g(\mathcal{R})}$.

**Theorem (Bombelli, Lee, Meyer, Sorkin, 1987).** The Poisson sprinkling process on a globally hyperbolic Lorentzian manifold $(\mathcal{M}, g)$ is invariant under all isometries of $(\mathcal{M}, g)$. In particular, in Minkowski space it is Lorentz invariant.

### 3.4. Volume from Counting (Hauptvermutung of Causal Set Theory)

The **Hauptvermutung** (fundamental conjecture) states that the number of elements encodes spacetime volume:

$$
\mathrm{Vol}_g(\mathcal{R}) \;=\; \frac{|\mathcal{C} \cap \mathcal{R}|}{\rho} \;=\; |\mathcal{C} \cap \mathcal{R}| \cdot l_P^4
$$

at the Planck density $\rho = l_P^{-4}$, up to Poisson fluctuations of order $\sqrt{|\mathcal{C} \cap \mathcal{R}|}\, l_P^4$.

### 3.5. Spectral Compatibility Condition

For causally related elements $x \prec y$, the spectral data must be compatible in the following sense. Define the **causal propagator** from spectral data:

$$
K(x, y) = \langle x | \, \mathrm{sgn}(D) \, e^{-\epsilon |D|} \, | y \rangle
$$

where $\epsilon \to 0^+$ is a regulator. The causal relation $x \prec y$ requires:

$$
K(x, y) \neq 0 \quad \text{(causal connectedness through spectral data)}
$$

Elements with $K(x, y) = 0$ are spacelike separated. This provides an intrinsic, spectral characterization of the causal order that does not require an ambient manifold.

### 3.6. Myrheim--Meyer Dimension Estimator

Given a finite causal set, the **Myrheim--Meyer estimator** determines the effective dimension from the order relations. For $N$ elements in a causal diamond in $d$-dimensional Minkowski space, the fraction $f$ of ordered pairs satisfies:

$$
f = \frac{|\{(x, y) : x \prec y\}|}{\binom{N}{2}} \;\approx\; \frac{\Gamma(d+1)\,\Gamma(d/2)}{4\,\Gamma(3d/2)}
$$

This provides a purely order-theoretic definition of dimension that requires no embedding.

---

## 4. Limiting Cases

### 4.1. Classical Limit ($\hbar \to 0$)

As $\hbar \to 0$ (equivalently, as the density $\rho \to \infty$ while holding macroscopic volumes fixed), the causal set $(\mathcal{C}, \preceq)$ approaches the continuum limit. By the Hauptvermutung, the causal set faithfully embeds into a smooth Lorentzian manifold $(\mathcal{M}, g_{\mu\nu})$, and:

$$
(\mathcal{C}, \preceq) \;\xrightarrow{\;\hbar \to 0\;}\; (\mathcal{M}, g_{\mu\nu})
$$

The causal order $\preceq$ reproduces the light-cone structure of $(\mathcal{M}, g)$. The causal diamond volume reproduces the metric volume element. By the Hawking--King--McCarthy--Malament theorem, the full conformal geometry is recovered from causality. Combined with volume information, the full metric $g_{\mu\nu}$ is determined.

### 4.2. Weak Field, $v \ll c$

For a causal set faithfully embedded in a weakly curved spacetime $g_{\mu\nu} = \eta_{\mu\nu} + h_{\mu\nu}$, the causal relations reduce to standard Minkowski light-cone ordering at leading order. The number--volume correspondence gives:

$$
N \approx \rho \cdot V_4 = \rho \int \sqrt{-g}\, d^4x \approx \rho \int \left(1 + \tfrac{1}{2}h\right) d^4x
$$

and corrections to flat-space counting encode the Newtonian potential $\Phi$ through $h_{00} = -2\Phi/c^2$.

### 4.3. Large Distance ($r \to \infty$)

At scales $r \gg l_P$, the discreteness of $\mathcal{C}$ is unresolvable. The Poisson fluctuations in element counting scale as $\delta N / N \sim 1/\sqrt{N} \to 0$. The causal set is indistinguishable from a continuum, and the causal structure reduces to the light-cone structure of flat Minkowski spacetime. Standard special-relativistic causality ($x \preceq y$ iff $y - x$ is future-directed causal) is exactly recovered.

### 4.4. Planck Scale ($E \to E_{\mathrm{Planck}}$)

At the Planck scale, the discrete structure becomes directly relevant:

- **Effective dimension flows to $\sim 2$.** The Myrheim--Meyer estimator applied to small causal diamonds yields $d_{\mathrm{eff}} \approx 2$, consistent with results from causal dynamical triangulations and the asymptotic safety program.
- **Lorentz invariance is exact** (in the statistical sense), not broken by the discreteness due to the Poisson process.
- **Causal links dominate.** At the smallest scales, almost all causal relations are links ($x \prec\!\!\!\cdot\; y$), and the layered structure of the causal set replaces the notion of time evolution.
- **Volume quantization.** Spacetime volume is quantized in units of $l_P^4$, with quantum fluctuations of order $\sqrt{V/l_P^4} \cdot l_P^4$.
