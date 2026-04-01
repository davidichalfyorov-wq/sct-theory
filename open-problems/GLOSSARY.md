# SCT Notation Glossary

Comprehensive notation reference for Spectral Causal Theory.
All symbols are listed with definitions, conventions, and first appearance.

**Unit convention.** Natural units $c = \hbar = 1$ unless otherwise stated.
Metric signature $(-,+,+,+)$ Lorentzian; $(+,+,+,+)$ Euclidean where noted.

---

## Spacetime and Geometry

| Symbol | Meaning | Convention |
|--------|---------|------------|
| $g_{\mu\nu}$ | Spacetime metric | Signature $(-,+,+,+)$ |
| $g$ | Metric determinant | $g = \det(g_{\mu\nu})$ |
| $\eta_{\mu\nu}$ | Minkowski metric | $\mathrm{diag}(-1,+1,+1,+1)$ |
| $\nabla_\mu$ | Covariant derivative | Levi-Civita connection of $g_{\mu\nu}$ |
| $\Gamma^\lambda_{\mu\nu}$ | Christoffel symbols | Torsion-free, metric-compatible |
| $R^\rho{}_{\sigma\mu\nu}$ | Riemann tensor | $[\nabla_\mu, \nabla_\nu] V^\rho = R^\rho{}_{\sigma\mu\nu} V^\sigma$ |
| $R_{\mu\nu}$ | Ricci tensor | $R_{\mu\nu} = R^\lambda{}_{\mu\lambda\nu}$ |
| $R$ | Ricci scalar | $R = g^{\mu\nu} R_{\mu\nu}$ |
| $G_{\mu\nu}$ | Einstein tensor | $G_{\mu\nu} = R_{\mu\nu} - \tfrac{1}{2} g_{\mu\nu} R$ |
| $C_{\mu\nu\rho\sigma}$ | Weyl tensor | Traceless part of Riemann |
| $C^2$ | Weyl-squared invariant | $C^2 = C_{\mu\nu\rho\sigma} C^{\mu\nu\rho\sigma}$ |
| $E_4$ | Euler (Gauss-Bonnet) density | $E_4 = R^2_{\mu\nu\rho\sigma} - 4 R^2_{\mu\nu} + R^2$ |
| $B_{\mu\nu}$ | Bach tensor | $B_{\mu\nu} = \nabla^\rho \nabla^\sigma C_{\mu\rho\nu\sigma} + \tfrac{1}{2} R^{\rho\sigma} C_{\mu\rho\nu\sigma}$ |
| $H_{\mu\nu}$ | $R^2$ variation tensor | $H_{\mu\nu} = -2\nabla_\mu \nabla_\nu R + 2 g_{\mu\nu} \Box R - \tfrac{1}{2} g_{\mu\nu} R^2 + 2 R R_{\mu\nu}$ |
| $\Box$ | d'Alembertian | $\Box = g^{\mu\nu} \nabla_\mu \nabla_\nu$ |
| $\Delta$ | Laplacian (BV sign) | $\Delta = -(g^{\mu\nu} \nabla_\mu \nabla_\nu + E)$ |
| $d$ | Spacetime dimension | $d = 4$ unless stated otherwise |
| $\kappa$ | Gravitational coupling | $\kappa^2 = 16\pi G$ |
| $G$ | Newton constant | $G = \ell_P^2$ in natural units |
| $\ell_P$ | Planck length | $\ell_P = \sqrt{G} \approx 1.616 \times 10^{-35}$ m |
| $M_P$ | Planck mass | $M_P = 1/\sqrt{G} \approx 1.221 \times 10^{19}$ GeV |
| $E_{ij}$ | Electric Weyl tensor | $E_{ij} = C_{0i0j}$ (observer-dependent) |
| $B_{ij}$ | Magnetic Weyl tensor | $B_{ij} = \tfrac{1}{2} \epsilon_{ikl} C^{kl}{}_{0j}$ |
| $E^2$ | Electric Weyl invariant | $E^2 = E_{ij} E^{ij}$ |
| $B^2$ | Magnetic Weyl invariant | $B^2 = B_{ij} B^{ij}$ |
| $q_W$ | Tidal (Weyl) parameter | Dimensionless curvature strength for local diamond |

---

## Spectral Geometry and Noncommutative Geometry

| Symbol | Meaning | Convention |
|--------|---------|------------|
| $(\mathcal{A}, \mathcal{H}, D)$ | Spectral triple | Connes axioms (algebra, Hilbert space, Dirac operator) |
| $\mathcal{A}$ | Involutive unital algebra | Faithfully represented on $\mathcal{H}$ |
| $\mathcal{H}$ | Hilbert space | Separable; carries representation of $\mathcal{A}$ |
| $D$ | Dirac operator | Self-adjoint, compact resolvent, $[D,a]$ bounded for all $a \in \mathcal{A}$ |
| $D^2$ | Dirac-squared (Laplace-type) | $D^2 = -(\Box + E)$ with endomorphism $E$ |
| $\Lambda$ | Spectral cutoff | Mass dimension 1; UV regularization scale |
| $f$ | Cutoff (spectral) function | $f: \mathbb{R}^+ \to \mathbb{R}^+$, rapid decay; default $f(u) = e^{-u}$ |
| $\psi(u)$ | Spectral function | $\psi(u) = f(u)$; exponential: $\psi(u) = e^{-u}$ |
| $S_{\mathrm{sp}}$ | Spectral action | $S_{\mathrm{sp}} = \mathrm{Tr}\, f(D^2/\Lambda^2)$ |
| $\zeta_D(s)$ | Spectral zeta function | $\zeta_D(s) = \mathrm{Tr}(|D|^{-s})$, $\mathrm{Re}(s) > d$ |
| $d_{\mathrm{Connes}}(p,q)$ | Connes distance | $\sup\{|a(p) - a(q)| : \|[D,a]\| \le 1\}$ |
| $d_S(\sigma)$ | Spectral dimension | $d_S = -2\, d\ln P(\sigma)/d\ln\sigma$ |
| $P(\sigma)$ | Return probability | $P(\sigma) = \mathrm{Tr}(e^{-\sigma D^2})$ |
| $E$ | Endomorphism in $D^2$ | Spin-dependent: $E = -R/4$ (Dirac), $E = 0$ (minimal scalar) |
| $\Omega_{\mu\nu}$ | Bundle curvature | $\Omega_{\mu\nu} = [D_\mu, D_\nu]$ on the relevant bundle |
| $\hat{P}$ | Shifted endomorphism | $\hat{P} = E + R/6$; equals $-R/12$ for Dirac |
| $a_n$ | Seeley-DeWitt coefficients | Heat kernel expansion: $\mathrm{Tr}(e^{-t\Delta}) \sim \sum_n a_n t^{(n-d)/2}$ |
| $a_4$ | Fourth SD coefficient | $a_4 = -18 C^2 + 11 E_4 + 0 \cdot R^2$ (massless Dirac, $d=4$) |
| $\gamma_5$ | Chirality matrix | $\{\gamma_5, D\} = 0$ on spin manifold |

---

## Master Function and Parametric Integrals

| Symbol | Meaning | Convention |
|--------|---------|------------|
| $\varphi(x)$ | Master function | $\varphi(x) = \int_0^1 d\xi\, e^{-\xi(1-\xi)x}$ |
| $\varphi(x)$ | Closed form | $\varphi(x) = e^{-x/4}\sqrt{\pi/x}\;\mathrm{erfi}(\sqrt{x}/2)$ |
| $\mathrm{erfi}(z)$ | Imaginary error function | $\mathrm{erfi}(z) = -i\,\mathrm{erf}(iz) = \tfrac{2}{\sqrt{\pi}} \int_0^z e^{t^2} dt$ |
| $a_n^{(\varphi)}$ | Taylor coefficients of $\varphi$ | $a_n = (-1)^n\, n!/(2n+1)!$ |
| $x$ | Dimensionless argument | $x = -\Box/\Lambda^2$ or $x = k^2/\Lambda^2$ depending on context |
| $z$ | Complex spectral variable | $z = k^2/\Lambda^2$ (Euclidean) or $z = -p^2/\Lambda^2$ (Lorentzian) |
| $\Psi_1(s)$, $\Psi_2(s)$ | Spectral moments | $\Psi_j(s) = \int_0^\infty u^j \psi(u) e^{-su} du$ |
| $\Phi_U$, $\Phi_{RU}$, $\Phi_\Omega$ | BV parametric weights | $\Phi_U = 1/2$, $\Phi_{RU} = -\xi(1-\xi)$, $\Phi_\Omega = (1-2\xi)^2/4$ |
| $f_{2k}$ | Spectral moments of $f$ | $f_{2k} = \int_0^\infty u^{k-1} f(u) du$; for $\psi = e^{-u}$: $f_{2k} = (k-1)!$ |

---

## Form Factors and Propagators

| Symbol | Meaning | Convention |
|--------|---------|------------|
| $h_C^{(s)}(x)$ | Weyl form factor, spin $s$ | Coefficient of $C^2$ in one-loop effective action |
| $h_R^{(s)}(x)$ | Ricci form factor, spin $s$ | Coefficient of $R^2$ in one-loop effective action |
| $h_C^{(0)}(x)$ | Scalar Weyl form factor | $\frac{1}{12x} + \frac{\varphi - 1}{2x^2}$ |
| $h_R^{(0)}(x;\xi)$ | Scalar Ricci form factor | $f_{R,\mathrm{bis}}(x) + \xi\, f_{RU}(x) + \xi^2\, f_U(x)$ |
| $h_C^{(1/2)}(x)$ | Dirac Weyl form factor | $\frac{3\varphi - 1}{6x} + \frac{2(\varphi - 1)}{x^2}$ |
| $h_R^{(1/2)}(x)$ | Dirac Ricci form factor | $\frac{3\varphi + 2}{36x} + \frac{5(\varphi - 1)}{6x^2}$ |
| $h_C^{(1)}(x)$ | Vector Weyl form factor | $\frac{\varphi}{4} + \frac{6\varphi - 5}{6x} + \frac{\varphi - 1}{x^2}$ |
| $h_R^{(1)}(x)$ | Vector Ricci form factor | $-\frac{\varphi}{48} + \frac{11 - 6\varphi}{72x} + \frac{5(\varphi - 1)}{12x^2}$ |
| $F_1(z)$ | Total Weyl form factor | $F_1 = \alpha_C(\Box/\Lambda^2)/(16\pi^2)$; $F_1(0) = 13/(1920\pi^2)$ |
| $F_2(z,\xi)$ | Total Ricci form factor | $F_2 = \alpha_R(\Box/\Lambda^2, \xi)/(16\pi^2)$ |
| $\hat{F}_1(z)$ | Normalized shape function | $F_1(z) = \alpha_C \cdot \hat{F}_1(z)/(16\pi^2)$ |
| $\hat{F}_2(z,\xi)$ | Normalized shape function | $F_2(z,\xi) = \alpha_R(\xi) \cdot \hat{F}_2(z,\xi)/(16\pi^2)$ |
| $\Pi_{\mathrm{TT}}(z)$ | Spin-2 propagator dressing | $\Pi_{\mathrm{TT}}(z) = 1 + \tfrac{13}{60}\, z\, \hat{F}_1(z)$ |
| $\Pi_s(z,\xi)$ | Scalar propagator dressing | $\Pi_s(z,\xi) = 1 + 6(\xi - 1/6)^2\, z\, \hat{F}_2(z,\xi)$ |
| $G_{\mathrm{TT}}(k^2)$ | Dressed TT graviton propagator | $G_{\mathrm{TT}} = 1/(k^2 \cdot \Pi_{\mathrm{TT}})$ |
| $\Pi_{\mathrm{entire}}(z)$ | Full propagator product | $\Pi_{\mathrm{entire}}(z) > 0$ for all real $z \ge 0$ |
| $g_A(z)$ | Entire part of ML expansion | $g_A = -13/60 = -c_2 = -2\alpha_C$ (constant) |
| $R_n$ | ML residues | Residue at $n$-th zero of $\Pi_{\mathrm{TT}}$ |
| $z_L$ | Lorentzian ghost pole | $z_L \approx -1.2807$ (timelike, physical ghost) |
| $z_0$ | Euclidean ghost pole | $z_0 \approx 2.4148$ (spacelike) |

### CZ Basis Form Factors (Codello-Zanusso)

| Symbol | Formula | $f(0)$ |
|--------|---------|--------|
| $f_{\mathrm{Ric}}(x)$ | $\frac{1}{6x} + \frac{\varphi - 1}{x^2}$ | $1/60$ |
| $f_R(x)$ | $\frac{\varphi}{32} + \frac{\varphi}{8x} - \frac{7}{48x} - \frac{\varphi - 1}{8x^2}$ | $1/120$ |
| $f_{RU}(x)$ | $-\frac{\varphi}{4} - \frac{\varphi - 1}{2x}$ | $-1/6$ |
| $f_U(x)$ | $\varphi/2$ | $1/2$ |
| $f_\Omega(x)$ | $-(\varphi - 1)/(2x)$ | $1/12$ |
| $f_{R,\mathrm{bis}}(x)$ | $\frac{1}{3} f_{\mathrm{Ric}} + f_R$ | $1/72$ |

---

## Constants and Coefficients

### Beta-Coefficients (Local Limits)

| Symbol | Meaning | Value |
|--------|---------|-------|
| $\beta_W^{(0)}$ | Scalar Weyl beta | $1/120$ |
| $\beta_R^{(0)}(\xi)$ | Scalar Ricci beta | $\tfrac{1}{2}(\xi - 1/6)^2$ |
| $\beta_W^{(1/2)}$ | Dirac Weyl beta | $1/20$ (per 4-component Dirac fermion) |
| $\beta_R^{(1/2)}$ | Dirac Ricci beta | $0$ (conformal invariance) |
| $\beta_W^{(1)}$ | Vector Weyl beta | $1/10$ (gauge + 2 FP ghosts) |
| $\beta_R^{(1)}$ | Vector Ricci beta | $0$ (Maxwell conformal in $d=4$) |

### Combined SM Coefficients

| Symbol | Meaning | Value |
|--------|---------|-------|
| $\alpha_C$ | Total Weyl${}^2$ coefficient | $13/120$ (parameter-free) |
| $\alpha_R(\xi)$ | Total $R^2$ coefficient | $2(\xi - 1/6)^2$ |
| $c_1$ | $R^2$ coefficient ($\{R^2, R_{\mu\nu}^2\}$ basis) | $\alpha_R - \tfrac{2}{3}\alpha_C$ |
| $c_2$ | $R_{\mu\nu}^2$ coefficient | $2\alpha_C = 13/60$ |
| $c_1/c_2$ | Coefficient ratio | $-1/3 + 120(\xi - 1/6)^2/13$ |
| $3c_1 + c_2$ | Scalar mode combination | $6(\xi - 1/6)^2$ (zero at $\xi = 1/6$) |
| $c_{\log}$ | BH entropy log correction | $37/24$ |

### Effective Masses

| Symbol | Meaning | Value |
|--------|---------|-------|
| $m_2$ | Spin-2 Yukawa mass | $\Lambda\sqrt{60/13} \approx 2.148\,\Lambda$ |
| $m_0(\xi = 0)$ | Scalar Yukawa mass (minimal) | $\Lambda\sqrt{6} \approx 2.449\,\Lambda$ |
| $m_0(\xi = 1/6)$ | Scalar Yukawa mass (conformal) | $\infty$ (decoupled) |
| $m_2/m_0$ | Mass ratio (minimal coupling) | $\sqrt{10/13} \approx 0.877$ |

### Modified Newtonian Potential

| Symbol | Meaning | Value |
|--------|---------|-------|
| $V(r)/V_N(r)$ | Potential ratio | $1 - \tfrac{4}{3} e^{-m_2 r} + \tfrac{1}{3} e^{-m_0 r}$ |
| $\alpha$ | Yukawa coupling | $-4/3$ (exact, parameter-free) |

### UV Asymptotics

| Symbol | Meaning | Value |
|--------|---------|-------|
| $x \cdot \alpha_C(x \to \infty)$ | UV asymptotic of total Weyl | $-89/12$ |
| $\varphi(x \to \infty)$ | UV limit of master function | $\sim 2/x$ (NOT zero) |

### SM Multiplicities

| Symbol | Meaning | Value | Convention |
|--------|---------|-------|-----------|
| $N_s$ | Real scalar fields | 4 | Higgs doublet (4 real components) |
| $N_f$ | Weyl fermion count | 45 | $3 \times 15$ Weyl spinors per generation |
| $N_D$ | Dirac-equivalent count | 22.5 | $N_f/2 = 45/2$ |
| $N_v$ | Gauge bosons | 12 | $\mathrm{SU}(3) \times \mathrm{SU}(2) \times \mathrm{U}(1)$: $8+3+1$ |
| $\xi$ | Non-minimal coupling | Free parameter; $\xi = 1/6$ is conformal coupling |

### Physical Constants

| Symbol | Meaning | Value |
|--------|---------|-------|
| $G$ | Newton constant | $6.674 \times 10^{-11}\;\mathrm{m}^3\,\mathrm{kg}^{-1}\,\mathrm{s}^{-2}$ |
| $\ell_P$ | Planck length | $1.616 \times 10^{-35}$ m |
| $M_P$ | Planck mass | $1.221 \times 10^{19}$ GeV |
| $\Lambda$ | Spectral cutoff (SCT) | Free parameter; $\Lambda \ge 2.38 \times 10^{-3}$ eV (Eoet-Wash) |

---

## PPN and Gravitational Tests

| Symbol | Meaning | Convention |
|--------|---------|------------|
| $\gamma_{\mathrm{PPN}}$ | Post-Newtonian light deflection | $\gamma = 1 + \tfrac{2}{3} e^{-m_2 r} + O(e^{-m_0 r})$ |
| $\beta_{\mathrm{PPN}}$ | Post-Newtonian perihelion | Not yet derived (requires NT-4b completion) |
| $c_T$ | Gravitational wave speed | $c_T = c$ exactly; $|c_T/c - 1| \sim 10^{-123}$ |

---

## Causal Sets

| Symbol | Meaning | Convention |
|--------|---------|------------|
| $(\mathcal{C}, \preceq)$ | Causal set | Locally finite partial order |
| $N$ | Number of sprinkled elements | $N \in \mathbb{N}$ |
| $\rho$ | Sprinkling density | $\rho = N/V_d$ (elements per volume) |
| $C_{ij}$ | Causal matrix | $C_{ij} = 1$ if $i \prec j$, else 0 |
| $C^2_{ij}$ | Interval volume matrix | $(C^2)_{ij} = \sum_k C_{ik} C_{kj}$ = number of paths $i \to k \to j$ |
| $L_{ij}$ | Link matrix | $L_{ij} = 1$ if $i \prec j$ and no $k$ with $i \prec k \prec j$ |
| $T$ | Diamond proper time | Proper time across Alexandrov interval |
| $V_4(T)$ | Diamond 4-volume | $V_4 = \pi T^4/24$ (flat Minkowski) |

### Hasse Diagram and Path Statistics

| Symbol | Meaning | Convention |
|--------|---------|------------|
| $H$ | Hasse diagram | DAG of links (transitive reduction of $\preceq$) |
| $p_\downarrow(i)$ | Longest chain below $i$ | Max antichain to $i$ in Hasse diagram |
| $p_\uparrow(i)$ | Longest chain above $i$ | Max antichain from $i$ in Hasse diagram |
| $\kappa$ | Path kurtosis | Excess kurtosis of $(p_\downarrow, p_\uparrow)$ distribution |
| $Y_i$ | Link score | $Y_i = \log_2(p_\downarrow \cdot p_\uparrow + 1)$ |
| $\delta Y_i$ | Link score deviation | $\delta Y_i = Y_i - \bar{Y}$ within stratum |

### CJ Estimator (Stratified Covariance)

| Symbol | Meaning | Convention |
|--------|---------|------------|
| CJ | Stratified covariance estimator | $\mathrm{CJ} = \sum_B w_B \mathrm{Cov}_B(X, \delta Y^2)$ |
| $X_i$ | Depth product | Proxy for geodesic distance within stratum |
| $B$ | Stratum | Partition of elements by $n = $ number of ancestors/descendants |
| $w_B$ | Stratum weight | Proportional to stratum size |
| $A_{\mathrm{align}}$ | Alignment observable | $A_{\mathrm{align}} = \sum_B w_B \mathrm{Cov}_B(X^2, \delta Y^2)$ |

### Sprinkling and Geometry

| Symbol | Meaning | Convention |
|--------|---------|------------|
| $\epsilon$ | Curvature amplitude (pp-wave) | $h_{xx} = \epsilon x_3^2$ in Brinkmann coordinates |
| $r_s$ | Schwarzschild radius | $r_s = 2GM$ |
| $k_d(s)$ | Retarded kernel | $k_d(s) = s^d/d!$ (Benincasa-Dowker) |
| $c_4$ | BD normalization ($d=4$) | $c_4 = 4/\sqrt{6}$ |
| GSF | Geodesic separation function | $\mathrm{GSF}(i,j) = $ proper geodesic distance between sprinkled points |

---

## Unitarity and Ghosts

| Symbol | Meaning | Convention |
|--------|---------|------------|
| $R_n$ | Residue at $n$-th pole | $R_n = \mathrm{Res}_{z=z_n}[1/(z \Pi_{\mathrm{TT}})]$ |
| $R_L$ | Residue at Lorentzian ghost | $R_L \approx -0.5378$ (negative: ghost) |
| $\rho_{\mathrm{TT}}$ | Spectral function | Pure delta-function (no branch cuts; $\Pi_{\mathrm{TT}}$ entire) |
| $C_m$ | Matter self-energy coefficient | $C_m = 283/120$ |
| $N_{\mathrm{eff}}$ | Effective degree count | $N_{\mathrm{eff}} = 143.5$ |
| $\Sigma(p)$ | Self-energy | $\mathrm{Im}[\Sigma] > 0$ (consistent with optical theorem) |

---

## Field Equations

| Symbol | Meaning | Convention |
|--------|---------|------------|
| $T_{\mu\nu}$ | Stress-energy tensor | Standard definition, $\nabla^\mu T_{\mu\nu} = 0$ |
| $\Theta^{(C)}_{\mu\nu}$ | Weyl nonlocal correction | BV $\alpha$-insertion term from $C^2$ variation |
| $\Theta^{(R)}_{\mu\nu}$ | Ricci nonlocal correction | BV $\alpha$-insertion term from $R^2$ variation |
| $F_1(\Box/\Lambda^2)$ | Weyl form factor operator | Acts on $B_{\mu\nu}$ in field equations |
| $F_2(\Box/\Lambda^2)$ | Ricci form factor operator | Acts on $H_{\mu\nu}$ in field equations |
| $P^{(2)}_{\mu\nu,\alpha\beta}$ | Spin-2 Barnes-Rivers projector | $\mathrm{tr}(P^{(2)}) = 5$ |
| $P^{(0-s)}_{\mu\nu,\alpha\beta}$ | Spin-0 scalar Barnes-Rivers projector | $\mathrm{tr}(P^{(0-s)}) = 1$ |

---

## Cosmology (FLRW)

| Symbol | Meaning | Convention |
|--------|---------|------------|
| $H$ | Hubble parameter | $H = \dot{a}/a$ |
| $a(t)$ | Scale factor | FLRW line element: $ds^2 = -dt^2 + a(t)^2 d\mathbf{x}^2$ |
| $\rho$ | Energy density | $H^2 = 8\pi G \rho/3$ (standard Friedmann) |
| $w_{\mathrm{eff}}$ | Effective equation of state | SCT: $w_{\mathrm{eff}} = -1$ to 63 digits at late times |
| $n_s$ | Scalar spectral index | SCT (conditional): $n_s(N=55) = 0.965$ |
| $r$ | Tensor-to-scalar ratio | SCT (conditional): $r(N=55) \approx 3.5 \times 10^{-3}$ |
| $f_{\mathrm{NL}}$ | Non-Gaussianity | SCT (conditional): $f_{\mathrm{NL}} \approx -0.016$ |
| $\delta H^2/H^2$ | SCT Hubble correction | $\sim 10^{-64}$ at $\Lambda = 2.38 \times 10^{-3}$ eV |

---

## Formal Verification (Lean 4)

| Symbol | Meaning | Convention |
|--------|---------|------------|
| `SCTLean` | Lean 4 project | 5 modules: Basic, FormFactors, SpectralAction, Tensors, StandardModel |
| `PhysLean` | Physics formalization library | Lorentz group, SM Lagrangian, FLRW |
| `Mathlib4` | Mathematics library | Riemannian manifolds, spectral theory |

---

## GTA Framework (Causal Set Observables)

| Symbol | Meaning | Convention |
|--------|---------|------------|
| $b_{\mathrm{eff}}$ | Effective screening depth | $b_{\mathrm{eff}} = 5$ (from data) |
| $\sigma_0$ | Noise amplitude | $\sigma_0 \approx 0.299 \times N^{1/4}$ |
| $H_{\mathrm{eff}}$ | Effective Hasse depth | Resolvent-composite depth statistic |
| $\tilde{A}$ | Normalized signal | $\tilde{A} = A/(H_{\mathrm{eff}}^2 T^4 E^2)$ |
| SA | Score amplitude | $\mathrm{SA} = \mathrm{CJ} + \mathrm{Between}$ (law of total covariance) |

---

## Indices and Summation Conventions

| Convention | Description |
|------------|-------------|
| Greek indices $\mu, \nu, \rho, \sigma, \ldots$ | Spacetime indices, $0, 1, 2, 3$ |
| Latin indices $i, j, k, \ldots$ | Spatial indices, $1, 2, 3$ |
| Spin label $s$ | $s = 0$ (scalar), $s = 1/2$ (Dirac), $s = 1$ (vector) |
| Einstein summation | Repeated upper-lower indices summed unless stated otherwise |
| $(s)$ superscript | Spin species label on form factors and beta-coefficients |
| Stratum index $B$ | Partition label in CJ stratification |
| Pole index $n$ | Index over zeros of $\Pi_{\mathrm{TT}}$ or $\Pi_s$ |

---

## Abbreviations

| Abbreviation | Expansion |
|--------------|-----------|
| SCT | Spectral Causal Theory |
| NCG | Noncommutative Geometry |
| BV | Barvinsky-Vilkovisky (heat kernel formalism) |
| CZ | Codello-Zanusso (form factor conventions) |
| CPR | Codello-Percacci-Rachwal (SM counting) |
| SD | Seeley-DeWitt (heat kernel coefficients) |
| ML | Mittag-Leffler (partial fraction expansion) |
| PPN | Parametrized Post-Newtonian |
| SM | Standard Model |
| GR | General Relativity |
| FP | Faddeev-Popov (ghost fields) |
| BR | Barnes-Rivers (momentum-space projectors) |
| BD | Benincasa-Dowker (causal set action) |
| FLRW | Friedmann-Lemaitre-Robertson-Walker |
| GW | Gravitational wave |
| UV | Ultraviolet |
| IR | Infrared |
| EOM | Equations of motion |
| TT | Transverse-traceless |
| CCC | Cubic curvature contractions |
| CRN | Causal random null (control) |
| GTA | Geodesic-Tidal-Alignment (framework) |
| CJ | [Stratified covariance estimator] |
