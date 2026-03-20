# Formal Theorem on the Impossibility of Direct Synthesis in Spectral Causal Theory (FND-1)

**Theorem (The FND-1 No-Go Theorem):** 
There exists no Dirac operator $D$ on a finite, discrete Lorentzian Causal Set $\mathcal{C}$ that simultaneously satisfies the axioms of Noncommutative Geometry (NCG) and reproduces the Einstein-Hilbert gravitational action via the Chamseddine-Connes Spectral Action.

**Proof Structure:**
The proof proceeds by analyzing the necessary and sufficient conditions for both NCG and the Spectral Action, demonstrating that they are mutually exclusive on a discrete graph.

## Part 1: The Discreteness Obstruction (The Loss of Poles)
The Chamseddine-Connes Spectral Action derives the Einstein-Hilbert action ($a_2 = \int R \sqrt{g}$) from the Seeley-DeWitt asymptotic expansion of the heat kernel:
$$ \text{Tr}(e^{-t D^2}) \sim \sum_{k \geq 0} a_{2k} t^{k - d/2} \quad (t \to 0) $$
Alternatively, this requires the spectral zeta function $\zeta_D(s) = \text{Tr}(|D|^{-s})$ to possess a simple pole at $s = d-2$.
On a finite causal set of $N$ elements, any operator $D$ is an $N \times N$ matrix with a discrete, finite spectrum $\{ \lambda_1, \dots, \lambda_N \}$. The zeta function evaluates to a finite sum:
$$ \zeta_D(s) = \sum_{n=1}^N |\lambda_n|^{-s} $$
A finite sum of entire functions is everywhere analytic. Therefore, $\zeta_D(s)$ has **no poles** anywhere in the complex plane. Consequently, all Seeley-DeWitt coefficients, including the curvature scalar $R$, are strictly zero. 
*Conclusion 1: Gravity cannot be extracted from the spectral action on a finite causal set.*

## Part 2: The Continuum Limit Obstruction (The Dual Paradox)
To circumvent Part 1, one must evaluate the Spectral Action in the continuum limit ($N \to \infty$). In this limit, the operator must converge to a continuum differential operator. Two natural candidates exist in Causal Set Theory:

**Candidate A: The Sorkin-Johnston Operator ($D_{SJ}$)**
Derived from the Pauli-Jordan matrix $A = \frac{1}{2}(C - C^T)$, the operator $D_{SJ}$ is self-adjoint, preserving the NCG axioms. However, the Pauli-Jordan function $\Delta(x,y)$ is a solution to the homogeneous wave equation $\Box \Delta = 0$. Its spectrum is strictly "on-shell" (confined to the mass shell). It is mathematically blind to the off-shell bulk geometry required to measure the Ricci scalar. 
*Conclusion 2A: The Sorkin-Johnston operator satisfies NCG but yields zero macroscopic gravity in the continuum limit.*

**Candidate B: The Benincasa-Dowker Operator ($B_{BD}$)**
This operator explicitly measures the discrete D'Alembertian and is proven to converge to $\Box - \frac{1}{2}R$, successfully recovering the Einstein-Hilbert action. However, $B_{BD}$ is a retarded differential operator. It is **not self-adjoint** ($B_{BD} \neq B_{BD}^\dagger$), and its spectrum is completely complex. Plugging it into the Spectral Action $\text{Tr}(f(B_{BD}))$ yields a divergent, unphysical, non-Hermitian action, fundamentally violating the foundational axiom of the Spectral Triple.
*Conclusion 2B: The Benincasa-Dowker operator recovers gravity but destroys the mathematical consistency of Noncommutative Geometry.*

## Final Verdict
The synthesis of a bare Lorentzian Causal Set and the Chamseddine-Connes Spectral Action is mathematically undecidable. The two formalisms cannot be unified at the fundamental discrete level. 

**Theoretical Pivot for SCT:**
The geometries must be formally decoupled.
1. The Causal Set generates the macroscopic topology and cosmological corrections (MT-2).
2. The Noncommutative Standard Model (ALG-1) must be defined on the smooth continuum limit of the causal set, treating the discrete points as a background regulator rather than the primary space of the Spectral Triple.
