# ADR-0002: Extract the simulated-annealing / QUBO core into a shared spine

**Status:** Accepted — Option A (standalone `anneal` spine repo). Executed 2026-06-16.
**Date:** 2026-06-15 (accepted 2026-06-16)
**Deciders:** Antonio (QuantumDrizzy)

## Context

AETHER's `research/quantum_annealing/` contains two clearly separable layers:

1. **A domain-agnostic optimizer** — pure-NumPy simulated annealing over a generic
   QUBO `dict[(i, j) -> float]`, with Metropolis updates, geometric cooling, and
   `num_reads` restarts (`annealer.py::MaterialAnnealer._run_numpy_sa`), an optional
   OpenJij SQA backend, and the validation harness that caught a real bug
   (`tests/python/test_annealer.py`: brute-force-optimum + energy/sample
   consistency, 2/2 passing). None of this knows anything about materials.

2. **A materials-specific formulation** — `QUBOFormulator` (z-score/sigmoid
   property scoring + pairwise compatibility), `MaterialSpec`, `DEMO_MATERIALS`,
   `_build_demo_compatibility`. This is AETHER's domain.

This is the same split that motivated **Spectra** (a spectral spine consumed by
AETHER/SUBSTRATE/DRIFT): a recurring mathematical primitive currently living inside
one lab, while other parts of the ecosystem re-implement or will re-implement the
same thing. Simulated annealing / QUBO is a primitive, not a materials feature —
KHAOS, SUBSTRATE, and future combinatorial-optimization work all want it.

The double-counting delta-energy bug (couplings counted twice relative to the
diagonal) was found *here* and fixed once. Without a single owned core, that class
of bug gets re-introduced every time someone re-writes SA.

## Decision

**Extract the domain-agnostic SA/QUBO core into its own spine repo** (working name
`anneal`, `QuantumDrizzy/<name>`, private), mirroring the Spectra pattern: a thin,
validated, dependency-light core with a frozen API, consumed by the labs. AETHER's
`QUBOFormulator` then *builds* a QUBO and hands it to the spine to *solve*; the
materials scoring stays in AETHER.

> Repo creation is gated on Antonio's explicit go (as with Spectra). This ADR is
> the design; no repo is created until approved.

### Proposed spine API (v0.1, frozen once agreed)

```python
# anneal/solver.py
def anneal(qubo: dict[tuple[int,int], float], *,
           backend: str = "sa",          # "sa" (pure numpy) | "sqa" (openjij, optional)
           num_reads: int = 100, seed: int = 42,
           t_init: float = 5.0, t_final: float = 1e-3, sweeps: int = 1000,
           ) -> AnnealResult: ...

# anneal/qubo.py  (generic builders — NOT material-specific)
def qubo_energy(qubo, sample) -> float            # the one true energy convention
def add_cardinality_penalty(qubo, n, k, weight)   # (sum x_i - k)^2 expanded
def add_onehot_penalty(qubo, group, weight)       # exactly-one-of-group
def brute_force_min(qubo, n) -> float             # exact, for validation/small n
```

`AnnealResult` = `{sample, energy, num_reads, timing_ms, method, all_energies}`
(today's `AnnealingResult`, renamed neutral). Backends mirror Spectra's
delegate-when-better philosophy: pure-NumPy SA always works; OpenJij SQA is an
optional accelerator selected by `backend="sqa"` with a clean fallback.

## Options Considered

### Option A: Extract to a shared spine repo (recommended)
| Dimension | Assessment |
|-----------|------------|
| Complexity | Low — the core is ~120 LoC, already isolated and tested |
| Cost | One new repo + AETHER consumes it by path (like `spectra`) |
| Scalability | KHAOS/SUBSTRATE/future combinatorial work reuse one audited core |
| Team familiarity | High — identical pattern to Spectra, just shipped |

**Pros:** one owned, validated SA/QUBO core; the brute-force test travels with it;
the bug-class is fixed once for everyone; coherent with the spine thesis.
**Cons:** another path dependency to keep present; small repo proliferation.

### Option B: Leave it in AETHER, import from there
**Pros:** zero new repos. **Cons:** other labs import a *materials* package to get a
generic solver (wrong dependency direction); AETHER becomes an accidental library;
re-implementation pressure elsewhere remains. Rejected — same anti-pattern Spectra
was created to kill.

### Option C: A generic `optimize` subpackage *inside* AETHER (not a repo)
A middle ground: move the generic core to `aether/optimize/` with no materials
imports, defer the repo. **Pros:** no repo decision now; clean internal boundary.
**Cons:** cross-repo consumers still import AETHER. Acceptable as a **step 0** if
Antonio wants to defer repo creation.

## Trade-off Analysis

The real question is repo-now (A) vs boundary-now-repo-later (C). The code is small
and already clean, so extraction cost is near-zero either way; the deciding factor
is whether a *second* consumer is imminent. KHAOS has QUBO-shaped structure (the
12-qubit feature / penalty framing) and SUBSTRATE does optimization — so a second
consumer is plausible but not committed today. Recommendation: **A** if Antonio
wants the spine lattice complete now (Spectra + anneal), **C** as the conservative
step that needs no new repo and can be promoted to A in one move later.

## Consequences

- **Easier:** one audited SA/QUBO core; the brute-force/energy-consistency tests
  guard it centrally; AETHER shrinks to its actual domain (materials scoring).
- **Harder:** another path dependency (`anneal = { path = "../anneal" }`-style for
  Python via editable install or sys.path, as Spectra is consumed).
- **Revisit:** if a GPU/Rust annealer is ever wanted (tenSORS-adjacent), the spine
  is the place to add a backend, not AETHER.

## Action Items
1. [x] Antonio approved Option **A** (standalone spine repo `anneal`).
2. [x] SA core + `qubo_energy`/`brute_force_min`/penalty builders extracted to
   `anneal`; `QUBOFormulator`/`DEMO_MATERIALS` stay in AETHER.
3. [x] Test suite in the spine (`anneal/tests/`, 10 green): brute-force optimum,
   energy/sample consistency, determinism, SQA fallback, cardinality/one-hot.
4. [x] AETHER `annealer.py` rewired to consume the spine (local SA removed,
   `MaterialAnnealer`/`AnnealingResult` API preserved); `test_annealer` 2/2 and
   full Python suite 24/24 green; demo works end-to-end.
5. [x] v0.1 API frozen in `anneal/docs/ADR-0001-anneal-spine.md`.

**Pending:** `gh repo create QuantumDrizzy/anneal --private` + push (the only
outward step; awaiting explicit go, as with Spectra). Repo is committed locally.
