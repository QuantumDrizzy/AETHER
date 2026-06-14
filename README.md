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

# Compute compatibility
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

## License

Proprietary — iNFAMØUS OS
