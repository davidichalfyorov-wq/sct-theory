---
id: OP-23
title: "Information paradox: island formula and Page curve from spectral principles"
domain: [black-holes, quantum-information]
difficulty: very-hard
status: open
deep-research-tier: D
blocks: []
blocked-by: [OP-02, OP-21]
roadmap-tasks: [LT-2]
papers: ["1905.08762", "1908.10996"]
date-opened: 2026-03-31
date-updated: 2026-03-31
---

# OP-23: Information paradox and island formula

## 1. Statement

Derive the island formula for entanglement entropy and the Page curve
for black hole evaporation from the spectral action principle. Determine
whether the nonlocal structure of SCT provides a resolution of the
AMPS firewall paradox.

The island formula states that the fine-grained entropy of Hawking
radiation is

  S(R) = min ext_I [ A(partial I)/(4 G) + S_bulk(R union I) ]

where I is the "island" region inside the black hole, partial I is its
boundary, and S_bulk is the bulk entanglement entropy. The
extremization is over all possible island surfaces.

The task is to derive this formula from the SCT spectral action and
to determine whether islands arise naturally from the spectral triple
structure.

## 2. Context

The information paradox (Hawking, 1975) asks whether information is
lost when a black hole evaporates. Recent progress (Penington 2019,
Almheiri-Engelhardt-Marolf-Maxfield 2019) shows that the island
formula, when applied to gravitational path integrals, produces a
Page curve consistent with unitarity.

The island formula has been derived in the context of AdS/CFT and
two-dimensional JT gravity. Its derivation relies on the gravitational
path integral (specifically, replica wormholes in the Euclidean path
integral). Whether an analogous derivation exists in SCT depends on:

1. Whether SCT has a well-defined path integral (Postulate 5, OP-02).
2. Whether the spectral action admits replica wormhole saddles.
3. Whether the nonlocal form factors modify the island prescription.

None of these questions has been investigated.

## 3. Known Results

- **BH entropy (MT-1):** S = A/(4G) + 13/(120 pi) + (37/24) ln(A/l_P^2).
  The Wald entropy formula is compatible with the area term in the
  island formula.

- **Nonlocal propagator:** the dressed propagator G_TT has no branch
  cuts (Pi_TT is entire), so the spectral function is a sum of delta
  functions. This means that the bulk entanglement entropy S_bulk has
  a discrete spectrum of entangling modes, unlike the continuous
  spectrum in local QFT.

- **Ghost sector:** the negative-norm ghost at z_L = -1.2807
  contributes negatively to entanglement entropy. This could either
  resolve or exacerbate the information paradox, depending on the
  ghost prescription.

- **Nonlocality scale:** the form factors F_1 and F_2 introduce a
  nonlocality scale ~ 1/Lambda. For stellar-mass black holes
  (r_s ~ km), 1/Lambda ~ 1/meV ~ 0.2 mm. The nonlocality scale is
  macroscopic compared to the Planck length but microscopic compared
  to the horizon.

- **No SCT-specific work exists.** The information paradox has not
  been studied within the spectral action framework.

## 4. Failed Approaches

No investigation has been attempted. The problem is blocked by two
prerequisite open problems:

1. OP-02 (Postulate 5): the island formula requires a gravitational
   path integral, which requires a specified dynamical principle for
   spectral triples.

2. OP-21 (singularity resolution): the interior geometry of the black
   hole determines whether islands can form. If the singularity is
   resolved (de Sitter core), the island geometry differs from the
   standard Penrose diagram.

## 5. Success Criteria

- **Island formula derivation:** obtain the island formula from the
  SCT spectral action, identifying the role of nonlocal form factors
  in the generalized entropy functional.

- **Page curve:** demonstrate that the entanglement entropy of Hawking
  radiation follows a Page curve (increases, then decreases after the
  Page time) for an evaporating SCT black hole.

- **Firewall resolution:** determine whether SCT nonlocality at scale
  1/Lambda provides a smooth horizon experience for infalling observers
  (no firewall), or whether the ghost sector creates a new type of
  firewall.

- **Unitarity:** verify that the total evolution is unitary in the
  fakeon-projected physical Hilbert space.

## 6. Suggested Directions

1. **JT gravity limit.** Dimensionally reduce the SCT spectral action
   to two dimensions and compare with Jackiw-Teitelboim gravity. If
   the 2d limit reproduces JT + nonlocal corrections, apply the
   known island formula machinery.

2. **Replica wormholes in spectral geometry.** In the path integral
   over spectral triples (Postulate 5, V3), replica geometries
   correspond to spectral triples with Z_n orbifold symmetry. Study
   whether the spectral action on Z_n-symmetric spectral triples
   admits saddle points that correspond to replica wormholes.

3. **Entanglement entropy from spectral data.** The von Neumann
   entropy of a subregion can be expressed in terms of the spectral
   zeta function of the reduced Dirac operator. Develop this
   connection and apply it to the black hole case.

4. **Nonlocal smearing of the horizon.** The form factors F_1(Box)
   and F_2(Box) smear the effective metric over a region of size
   ~ 1/Lambda around the horizon. Estimate the effect on Hawking
   radiation: does the smearing modify the Planckian spectrum? Does
   it introduce correlations between early and late Hawking quanta?

5. **Ghost contribution to entanglement.** The ghost pole at z_L has
   negative residue R_L = -0.5378. In the Lee-Wick framework, ghost
   particles contribute negatively to the entangling surface area.
   Compute the ghost correction to the island formula.

## 7. References

1. Penington, G. (2019). "Entanglement wedge reconstruction and the
   information problem." arXiv:1905.08762.
2. Almheiri, A., Engelhardt, N., Marolf, D. and Maxfield, H. (2019).
   "The entropy of bulk quantum fields and the entanglement wedge of
   an evaporating black hole." JHEP 12, 063. arXiv:1905.08255.
3. Almheiri, A., Mahajan, R., Maldacena, J. and Zhao, Y. (2020).
   "The Page curve of Hawking radiation from semiclassical geometry."
   JHEP 03, 149. arXiv:1908.10996.
4. Hawking, S. W. (1975). "Particle creation by black holes." Comm.
   Math. Phys. 43, 199.
5. Almheiri, A., Marolf, D., Polchinski, J. and Sully, J. (2013).
   "Black holes: complementarity vs. firewalls." JHEP 02, 062.
   arXiv:1207.3123.

## 8. Connections

- **Blocked by OP-02 (Postulate 5):** the island formula requires a
  path integral, which requires a dynamical principle.
- **Blocked by OP-21 (singularity resolution):** the island geometry
  depends on the interior structure of the black hole.
- **OP-22 (second law):** the Page curve is consistent with the second
  law only if the generalized entropy is well-defined, which requires
  the GSL (OP-22).
- **OP-07, OP-08 (ghost resolution):** the ghost sector affects both
  the entanglement entropy and the unitarity of Hawking evaporation.
- This is a long-term problem (roadmap LT-2) and is not expected to
  be tractable without significant progress on OP-01, OP-02, and OP-21.
