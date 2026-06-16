# AETHER

> A computational-materials and unconventional-computing-substrates lab.

A bare-metal research lab where **physics computes by minimizing energy** — and
every claim is checked against a closed form or a reference. Rust core (types,
SQLite, native GUI) under a Python research layer spanning electronic structure
and topology, metamaterials, self-organisation, criticality, and the
thermodynamics of computation. **~90 validation tests, all green.**

Three threads run through the lab:

- **the edge of chaos** — memory capacity, task performance and order all peak at
  the same critical boundary, across unrelated substrates;
- **structure sets the property** — auxetic metamaterials, band gaps and
  topological invariants come from geometry/symmetry, not the base material;
- **information ↔ energy ↔ matter** — Landauer, Maxwell's demon, and the
  thermodynamic price of computation, measured.

## The map

### Electronic structure & topology — `research/electronic_structure/`
Graphene tight-binding (Dirac cones, Fermi velocity, DOS) · hexagonal boron
nitride (gap = 2Δ from sublattice asymmetry) · the full topological set: **SSH**
(1D winding Z + bulk-boundary correspondence), **Haldane** (2D Chern = ±1,
quantum anomalous Hall), **Kane–Mele** (Z₂ quantum spin Hall) · **Anderson
localization** (disorder kills the metal). CPU/GPU k-space solver, validated GPU==CPU.

<p align="center">
  <img src="https://raw.githubusercontent.com/QuantumDrizzy/AETHER/master/figures/haldane_phase_diagram.png" alt="Haldane topological phase diagram" width="600">
</p>

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
(capacity collapse at α_c ≈ 0.138) · box-counting fractal dimension (exact on
Sierpiński).

<p align="center">
  <img src="https://raw.githubusercontent.com/QuantumDrizzy/AETHER/master/figures/universal_criticality.png" alt="universal criticality" width="640">
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

~90 tests across Python and Rust; every physics claim is checked against a closed
form or a published reference (Dirac bandwidth 6t, SSH 2 edge states, Haldane
Chern ±1, Kane–Mele Z₂, Hopfield α_c, percolation p_c, Landauer k_BT ln2, …). Two
"too-good" results were distrusted and corrected by their own tests — a Maxwell-
demon protocol that beat the information bound, and a reservoir "efficiency" that
was a saturation artefact.

```bash
pytest tests/python -q          # Python research layer
cargo test --workspace          # Rust core
```

## Architecture

```
AETHER/
├── crates/        ← Rust: core types · SQLite · CLI · native GUI · PyO3 FFI
├── research/      ← electronic_structure · metamaterials · em_simulation ·
│                    daemons · active_matter · programmable_matter ·
│                    reaction_diffusion · cellular_automata · criticality ·
│                    reservoir_computing · neuro · fractals · quantum_annealing
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
