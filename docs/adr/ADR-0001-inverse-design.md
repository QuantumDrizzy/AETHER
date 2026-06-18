# ADR-0001: Inverse design — target property → structure

**Status:** Accepted
**Date:** 2026-06-17
**Deciders:** Antonio (QuantumDrizzy)

## Context

AETHER is broad on the **forward** direction: given a structure, compute its property
(electronic bands, phonon dispersion, photonic gap, auxetic Poisson ratio, …). The
breadth is essentially complete; adding a 35th forward module would be padding.

The genuinely new capability — and the one with real engineering pull (generative /
inverse materials design is exactly what NVIDIA/Mistral-adjacent and metamaterials
groups care about) — is the **inverse problem**: given a *target property*, find the
*structure* that produces it. This ADR commits AETHER to an inverse-design line and
fixes the discipline so it stays honest.

## Decision

Build inverse design as a first-class line, with two routes per problem and one
non-negotiable validation:

- **Analytic inverse** where the forward law inverts in closed form (fast, exact).
- **Numeric inverse** otherwise: minimise the property mismatch *through the existing
  forward model* (the general pattern; no new physics, reuse what's validated).
- **Validation = round-trip, always.** Take a known structure, read its target
  property, invert, and check the recovered structure (and its actual property) match.
  This is the same calibrate-on-known discipline used across the lab; an inverse design
  is not trusted until it round-trips.

Reuse forward models directly; never hand-roll a second physics.

## Built so far (`research/inverse_design/`)

- `inverse_photonic.py` — target photonic band gap (centre, rel. bandwidth) → (n1,n2,f0);
  analytic (quarter-wave law) + numeric (through the transfer-matrix model). Round-trip + design-from-spec validated.
- `inverse_auxetic.py` — target Poisson ratio (+ h/l) → rib angle θ (quadratic in sinθ; auxetic → re-entrant θ<0).
- `inverse_hbn.py` — target (band gap, conduction-band top) → (Δ, t); validated vs the BZ model.

Spans metamaterials (photonic + mechanical) and electronic structure. 15 tests, all round-trip-validated.

## Options considered

### A — more forward modules
**Rejected.** Breadth is done; this adds little. Inverse design is the new capability.

### B — inverse design, analytic + numeric, round-trip-validated (CHOSEN)
Reuses validated forward models, stays honest, and opens a real direction.

### C — heavy generative/ML inverse design (e.g. learned surrogates, GAN/diffusion over structures)
**Deferred.** Powerful but premature: needs the analytic/numeric base + datasets first.
Revisit once the optimization-based inverse line is solid and a target needs it.

## Consequences

- **Easier:** turn any forward model into a designer; demonstrable, validatable, on-trend.
- **Harder:** numeric inverses can be ill-posed/multi-modal (several structures hit one
  target) — report that honestly (return the family, not a false unique answer).
- **Revisit:** multi-objective targets; constraints (manufacturability); a learned
  surrogate (option C) when warranted.

## Action items

1. [x] Photonic, auxetic, hBN inverse modules (round-trip validated).
2. [x] **Inverse SSH / topology** — target gap + topological phase → (t1,t2); validated 3 ways (bulk gap, winding number, finite-chain edge states).
3. [x] **DRIFT inverse-computronium** (cross-repo, in `DRIFT/drift/inverse_logic.py`) — target Boolean truth-table → QUBO couplings by LP feasibility (Ising-machine compilation); fed to DRIFT's anneal spine. Honest limit: XOR is not 2-local (detected, recovered by composition). The computational sibling of inverse materials.
4. [x] Handle multi-modal inverses honestly (`research/inverse_design/degeneracy.py`) — generate the solution family (photonic: n1 free / contrast locked; SSH: energy scale free; auxetic: aspect free) and forward-verify every member hits the target. Report the manifold, not a false-unique answer.
5. [ ] (Later) optimization over richer parameter spaces; a learned surrogate if a real target demands it.

## References

- Forward models: `research/em_simulation/photonic_crystal.py`, `research/metamaterials/auxetic.py`, `research/electronic_structure/{hbn,ssh}.py`.
- Inverse design / inverse problems in photonics & metamaterials (optimization-based design).
