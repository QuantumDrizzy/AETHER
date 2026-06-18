# AETHER

> A computational-materials and unconventional-computing-substrates lab.

A bare-metal research lab where **physics computes by minimizing energy** — and
every claim is checked against a closed form or a reference. Rust core (types,
SQLite, native GUI) under a Python research layer spanning electronic structure
and topology, lattice dynamics, real-space quantum mechanics, metamaterials,
self-organisation, criticality, and the thermodynamics of computation.
**~150 validation tests, all green.**

Four threads run through the lab:

- **the edge of chaos** — memory capacity, task performance and order all peak at
  the same critical boundary, across unrelated substrates;
- **structure sets the property** — auxetic metamaterials, band gaps and
  topological invariants come from geometry/symmetry, not the base material;
- **bands → phonons → thermodynamics** — the same matrix machinery gives electron
  bands, then phonon dispersion, then a macroscopic heat capacity (Debye T³);
- **information ↔ energy ↔ matter** — Landauer, Maxwell's demon, and the
  thermodynamic price of computation, measured.

## The map

### Electronic structure & topology — `research/electronic_structure/`
Graphene tight-binding (Dirac cones, Fermi velocity, DOS) · hexagonal boron
nitride (gap = 2Δ from sublattice asymmetry) · the full topological set: **SSH**
(1D winding Z + bulk-boundary correspondence), **Haldane** (2D Chern = ±1,
quantum anomalous Hall), **Kane–Mele** (Z₂ quantum spin Hall) · **Anderson
localization** (disorder kills the metal). CPU/GPU k-space solver, validated GPU==CPU.
Plus a **real-space Schrödinger solver** (finite differences) as the k-space
complement: harmonic-oscillator ladder, square-well levels, and barrier
tunnelling — each checked against its closed form.

<p align="center">
  <img src="https://raw.githubusercontent.com/QuantumDrizzy/AETHER/master/figures/haldane_phase_diagram.png" alt="Haldane topological phase diagram" width="600">
</p>

### Lattice dynamics — `research/phonons/`
Phonon dispersion of 1D chains as **band structure for vibrations**: a monatomic
acoustic branch and a diatomic acoustic+optical pair with a phonon band gap
(the vibrational twin of the hBN electronic gap), from the dynamical matrix.
Then the thermodynamic payoff — lattice **heat capacity** summed over the
spectrum: Dulong–Petit at high T, the Debye **T³ law** at low T (matched to the
analytic limit), and Einstein's too-fast exponential freeze-out.

### Metamaterials — `research/metamaterials/`, `research/em_simulation/`
Auxetic honeycomb (negative Poisson's ratio from re-entrant geometry) · 1D
photonic crystal (Bragg band gap from periodic structure) · effective-medium
negative-index metamaterial (SRR + wire) · 1D FDTD (Yee + PML).

### Daemons & self-organisation — `research/daemons/`, `research/active_matter/`, `research/programmable_matter/`, `research/reaction_diffusion/`, `research/cellular_automata/`
Maxwell's demon / Szilard engine (work = k_BT·ln2 × mutual information; Landauer
floor keeps the 2nd law) · Vicsek active matter (flocking transition + phase
diagram) · a target-seeking active swarm that self-assembles and self-repairs ·
programmable matter that reconfigures to a shape and heals under damage ·
Gray–Scott reaction–diffusion (morphogenesis) · Conway's Game of Life.

<p align="center">
  <img src="https://raw.githubusercontent.com/QuantumDrizzy/AETHER/master/figures/gray_scott_pattern.png" alt="Gray-Scott morphogenesis" width="640">
</p>

### Criticality & computation — `research/criticality/`, `research/reservoir_computing/`, `research/neuro/`, `research/fractals/`
**Universal criticality**: Ising, Vicsek and self-reconfiguration share one
order→disorder transition (the curves collapse) · site percolation (geometric
critical threshold p_c ≈ 0.593) · reservoir computing (ESN; NARMA-10 task error
minimised below the edge of chaos, ruined past it) · Hopfield associative memory
(capacity collapse at α_c ≈ 0.138) · Kuramoto synchronization (critical coupling)
· Watts–Strogatz small-world networks · elementary cellular automata (Wolfram
rules 30/90/110: chaos, Sierpiński, universality) · box-counting fractal
dimension (exact on Sierpiński).

<p align="center">
  <img src="https://raw.githubusercontent.com/QuantumDrizzy/AETHER/master/figures/universal_criticality.png" alt="universal criticality" width="640">
</p>

### Inverse design — target property → structure — `research/inverse_design/`
Every other line here is *forward* (structure → property). This one runs it
backwards: name the property you want and recover the structure that produces it,
each design **round-trip-validated through the forward model** it inverts.

- **Photonic band gap** → quarter-wave stack `(n₁, n₂, f₀)` (analytic + numeric).
- **Auxetic Poisson ratio** → honeycomb rib angle (negative ν ⇒ re-entrant geometry).
- **hBN/gapped honeycomb** → sublattice mass and hopping `(Δ, t)`.
- **Topological phase** (`inverse_ssh.py`) → SSH hoppings `(t₁, t₂)` for a target gap
  *and* a target invariant, checked three ways (bulk gap, winding number, and the
  finite-chain edge states — 2 if topological, 0 if trivial).
- **Honest degeneracy** (`degeneracy.py`): the inverse is many-to-one, so it returns
  the **family** of structures hitting one target (e.g. photonic bandwidth fixes the
  index *contrast*, leaving the absolute index free), not a single false-unique answer.

The same idea closes over computation in the sister repo **DRIFT**
(`inverse_logic.py`): a target Boolean truth table is compiled to QUBO couplings by
linear feasibility (and XOR is correctly reported as not 2-local, then built by
composition) — inverse design of a *computation* rather than a material.

<p align="center">
  <img src="https://raw.githubusercontent.com/QuantumDrizzy/AETHER/master/figures/inverse_degeneracy.png" alt="inverse-design degeneracy: a family of structures per target" width="720">
</p>

### Optimization & Rust core
Simulated annealing / QUBO (validated against brute-force optima) · Rust
workspace (`Material`/`Experiment` types, SQLite persistence, CLI, native egui GUI
with live band-structure plots).

## GPU acceleration — where CUDA pays (and where it doesn't)

CUDA goes where it earns its keep, with honest CPU-vs-GPU benchmarks on an
RTX 5060 Ti (sm_120) — not on toy sizes where launch overhead loses. Three HPC
patterns, measured (`*_gpu.py`):

| pattern | module | GPU speedup |
|---|---|---|
| lattice Monte Carlo | 2D Ising (`criticality/ising2d_gpu.py`) | 0.3× at 64² → **~199× at 1024²** |
| stencil / PDE | Gray–Scott (`reaction_diffusion/gray_scott_gpu.py`) | 1× at 128² → **~199× at 1024²** |
| dense eigensolve | Anderson (`electronic_structure/anderson_gpu.py`) | **~31× at L=500 → ~9× at 3000** |

The lattice/stencil cases lose below ~L≈100 (overhead) and win decisively when
large; the dense eigensolve wins across the range. GPU results are validated
against the CPU physics. Small models (graphene's 2×2 Hamiltonian, a 500-node
Kuramoto) stay on CPU on purpose — GPU there would be slower.

<p align="center">
  <img src="https://raw.githubusercontent.com/QuantumDrizzy/AETHER/master/figures/ising2d_gpu_speedup.png" alt="2D Ising CPU vs GPU speedup" width="600">
</p>

## Validation

~150 tests across Python and Rust; every physics claim is checked against a closed
form or a published reference (Dirac bandwidth 6t, SSH 2 edge states, Haldane
Chern ±1, Kane–Mele Z₂, QHO ladder, Debye T³, Hopfield α_c, percolation p_c,
Landauer k_BT ln2, …). Two "too-good" results were distrusted and corrected by
their own tests — a Maxwell-demon protocol that beat the information bound, and a
reservoir "efficiency" that was a saturation artefact.

```bash
pytest tests/python -q          # Python research layer
cargo test --workspace          # Rust core
```

## Architecture

```
AETHER/
├── crates/        ← Rust: core types · SQLite · CLI · native GUI · PyO3 FFI
├── research/      ← electronic_structure · phonons · metamaterials · em_simulation ·
│                    daemons · active_matter · programmable_matter ·
│                    reaction_diffusion · cellular_automata · criticality ·
│                    reservoir_computing · neuro · networks · fractals · quantum_annealing
├── figures/       ← generated benchmarks & visualisations
└── tests/         ← Python + Rust validation suites
```

## Quick start

```bash
cargo build --workspace --release
cargo run -p aether-cli -- init
cargo run -p aether-cli -- material seed     # curated library (graphene, carbyne, Bi, …)
python -m research.electronic_structure.haldane     # e.g. the topological model
```

---

*Bare-metal · local-first · honest benchmarks. © Antonio Zambudio.*
