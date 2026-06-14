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
| EM simulation — 1-D FDTD (Yee + PML, Courant) | ✅ implemented, runnable |
| Reservoir computing — leaky-integrator ESN | ✅ implemented, runnable |
| (Simulated/quantum) annealing — QUBO + SA, OpenJij optional | ✅ implemented, runnable |
| Rust core — `Material`/`Experiment` types, SQLite persistence | ✅ implemented |
| Compatibility engine (Rust, `aether-core::compatibility`, 6 dims) | ✅ implemented, ⚠️ **not yet wired to CLI/FFI** |
| `compat compute` / `compat matrix` (CLI) | ⚠️ prints only — does not call the engine yet |
| PyO3 FFI bridge (`aether-ffi`) | ⚠️ stub (exposes `Material.name` only) |
| Python compatibility scorer (`research/compatibility`) | ⚠️ placeholder — 2 of 7 metrics implemented; rest report as unimplemented |
| ZMQ IPC to THEIA/SUBSTRATE | ⚠️ declared in config, not wired |
| Tests / validation harness | ❌ none yet (see `docs/ADR-0001`) |

**No benchmarks are claimed yet** — nothing here is validated against analytic
references. That validation harness is the next milestone. See
`docs/ADR-0001-aether-hardening.md` for the plan.

## License

Proprietary — iNFAMØUS OS
