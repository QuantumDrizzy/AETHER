# AETHER LAB

> Advanced Materials Research Engine — iNFAMØUS OS

Plataforma de investigación de materiales avanzados: cristales, metamateriales, nanomateriales, aleaciones. Simulación cuántica-inspirada + compatibilidad multi-métrica + base de conocimiento evolutiva.

## Stack

- **Core Engine**: Rust (workspace con 6 crates)
- **Research Layer**: Python (Quantum Annealing, Reservoir Computing, EM Simulation)
- **Database**: SQLite embebida (rusqlite)
- **GUI**: egui + wgpu (nativo)
- **IPC**: ZMQ PUB/SUB (integración con THEIA/SUBSTRATE/KHAOS)

## Quick Start

```bash
# Build
cargo build --workspace --release

# Initialize database
cargo run -p aether-cli -- init

# Add materials
cargo run -p aether-cli -- material add "Cuarzo" --category crystal --dielectric 4.5
cargo run -p aether-cli -- material add "BaTiO3" --category crystal --dielectric 1200

# Compute compatibility (NOTE: CLI command is a stub today — see Status below)
cargo run -p aether-cli -- compat compute "Cuarzo" "BaTiO3"

# Run Python simulations
python -m research.quantum_annealing.annealer
python -m research.reservoir_computing.esn
python -m research.em_simulation.resonance
python -m research.compatibility.scorer
```

## Architecture

```
AETHER/
├── crates/                  ← Rust workspace
│   ├── aether-core/         ← Material, Experiment, Compatibility types
│   ├── aether-db/           ← SQLite persistence
│   ├── aether-acq/          ← Hardware acquisition (future)
│   ├── aether-ffi/          ← PyO3 Rust↔Python bridge
│   ├── aether-cli/          ← Terminal interface
│   └── aether-gui/          ← Native GUI (egui+wgpu)
├── research/                ← Python research layer
│   ├── quantum_annealing/   ← QUBO + Simulated QA
│   ├── reservoir_computing/ ← Echo State Networks
│   ├── em_simulation/       ← FDTD + resonance
│   ├── compatibility/       ← Multi-metric scoring
│   └── knowledge/           ← Pattern learning
└── data/                    ← SQLite DB + results
```

## Status (honest)

This is an early lab. What is actually working vs. scaffolded today:

| Component | Status |
|-----------|--------|
| Electronic structure — graphene tight-binding (Dirac cones, v_F, DOS) | ✅ implemented + **validated** vs closed form (7/7) |
| General N×N tight-binding solver (CPU/GPU k-space) | ✅ implemented + **validated** (reproduces graphene; GPU==CPU, 4/4) |
| EM simulation — 1-D FDTD (Yee + PML, Courant) | ✅ implemented + **validated** vs physics refs (3/3) |
| Reservoir computing — leaky-integrator ESN | ✅ implemented + **validated** (NARMA-10 NRMSE, spectral radius, 2/2) |
| (Simulated/quantum) annealing — QUBO + SA, OpenJij optional | ✅ implemented + **validated** (finds brute-force optimum; a delta-energy bug was caught & fixed, 2/2) |
| Rust core — `Material`/`Experiment` types, SQLite persistence | ✅ implemented |
| Compatibility engine (Rust, `aether-core::compatibility`, 6 dims) | ✅ implemented, ⚠️ **not yet wired to CLI/FFI** |
| `compat compute` / `compat matrix` (CLI) | ⚠️ prints only — does not call the engine yet |
| PyO3 FFI bridge (`aether-ffi`) | ⚠️ stub (exposes `Material.name` only) |
| Python compatibility scorer (`research/compatibility`) | ⚠️ placeholder — 2 of 7 metrics implemented; rest report as unimplemented |
| ZMQ IPC to THEIA/SUBSTRATE | ⚠️ declared in config, not wired |
| Validation harness | 🟢 18 tests, 5 modules (FDTD 3, graphene 7, TB solver 4, ESN 2, SA 2); Rust unit tests pending |

**First validated results** (fixed seeds):
- FDTD vacuum propagation speed: **0.966 c** (expected numerical dispersion at
  20 cells/λ, Courant 0.99).
- Graphene tight-binding: bandwidth **6t**, Dirac gap → 0 at K/K', Fermi velocity
  **8.74×10⁵ m/s (~ c/343)**, van Hove singularities at **|E| = 2.69 eV ≈ t** —
  all matching closed-form results.
- GPU k-space scaling (general N×N solver, 8192 k-points, RTX 5060 Ti vs NumPy;
  GPU time includes CPU-built H + host→device transfer, only the eigensolve is on
  GPU): N=2 **0.03×** (GPU loses — overhead), N=64 **6.4×**, N=256 **66×**
  (1440 s → 21.7 s). Crossover ~N=32. Honest takeaway: GPU is pointless for tiny
  models like graphene (2×2) and decisive for large ones (supercells, ribbons).

Everything else still carries no performance claims. See
`docs/ADR-0001-aether-hardening.md` for the plan; run the suites with
`python tests/python/test_fdtd_validation.py` and `python tests/python/test_graphene_tb.py`.

## License

Proprietary — iNFAMØUS OS
