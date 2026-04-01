# SCT Open Problems: Dependency Graph

```mermaid
graph TD
    subgraph "UV Completeness Path"
        OP02[OP-02: Postulate 5] --> OP03[OP-03: Non-perturbative]
        OP02 --> OP05[OP-05: QFT recovery]
        OP02 --> OP06[OP-06: UV-completeness]
        OP09[OP-09: BV axioms 3,4] --> OP06
        OP10[OP-10: D² vs metric] --> OP06
        OP13[OP-13: Three-loop] --> OP06
        OP14[OP-14: Hidden principle] --> OP06
    end

    subgraph "Unitarity Path"
        OP07[OP-07: Fakeon ∞-poles] --> OP08[OP-08: Kubo-Kugo]
        OP07 --> OP22[OP-22: BH second law]
        OP07 --> OP24[OP-24: Spectral dim]
        OP11[OP-11: IVP entire] --> OP12[OP-12: KK loop]
    end

    subgraph "Black Holes Path"
        OP01[OP-01: Gap G1] --> OP21[OP-21: BH singularity]
        OP21 --> OP23[OP-23: Information paradox]
        OP22 --> OP23
    end

    subgraph "Cosmology Path"
        OP17[OP-17: Scalaron mass] --> OP18[OP-18: Dilaton excluded]
    end

    subgraph "Causal Sets Path"
        OP34[OP-34: N scaling] --> OP36[OP-36: RSY Weyl]
        OP35[OP-35: Stratification] --> OP36
        OP39[OP-39: Exact Sch] --> OP42[OP-42: Universality]
    end

    classDef veryhard fill:#ff6b6b,stroke:#333,color:#fff
    classDef hard fill:#ffa07a,stroke:#333
    classDef medium fill:#98d8c8,stroke:#333
    classDef easy fill:#87ceeb,stroke:#333

    class OP01,OP02,OP03,OP06,OP09,OP10,OP13,OP14,OP19,OP21,OP23,OP36,OP41 veryhard
    class OP05,OP07,OP08,OP11,OP12,OP15,OP16,OP17,OP18,OP22,OP24,OP34,OP35,OP38,OP40,OP42 hard
    class OP04,OP20,OP25,OP27,OP28,OP30,OP31,OP32,OP33,OP37,OP43,OP44,OP46,OP47,OP48,OP50 medium
    class OP45,OP49 easy
```

## Critical paths

1. **UV path:** OP-02 → OP-06, with OP-09/10/13/14 as parallel inputs
2. **BH path:** OP-01 → OP-21 → OP-23, with OP-07 → OP-22 as parallel input
3. **CJ bridge path:** OP-34 + OP-35 → OP-36
4. **Cosmology path:** OP-17 → OP-18 (short, somewhat isolated)

## Unblocked problems (can start immediately)

OP-04, OP-07, OP-13, OP-14, OP-17, OP-20, OP-33, OP-34, OP-37, OP-40,
OP-44, OP-45, OP-46, OP-47, OP-48, OP-49, OP-50
