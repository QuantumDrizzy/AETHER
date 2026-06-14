# CLAUDE.md — AETHER LAB | Advanced Materials Research Engine
# Research Software Engineer | Bare-Metal | Materials Science

## IDENTIDAD
- **AETHER LAB**: Motor de investigación de materiales avanzados de iNFAMØUS OS.
- Cristales, metamateriales, nanomateriales, aleaciones, composites.
- Simulación cuántica-inspirada + compatibilidad multi-métrica + base de conocimiento evolutiva.
- Bridge bidireccional con THEIA via ZMQ PUB/SUB (tcp://localhost:5557).
- Módulos integrados:
  - **Quantum Annealing**: Optimización de combinaciones de materiales (OpenJij SQA).
  - **Reservoir Computing**: Predicción de comportamiento temporal (ReservoirPy ESN).
  - **EM Simulation**: FDTD para cristales y metamateriales.
  - **Compatibility Engine**: Motor de scoring multi-métrica con 10 dimensiones.
  - **Knowledge Base**: Aprendizaje evolutivo de correlaciones.

## RESTRICCIONES CRÍTICAS
- **GPU compartida**: THEIA, SUBSTRATE y AETHER comparten la misma RTX 5060 Ti 16GB.
  - Liberar VRAM tras cada simulación. `torch.cuda.empty_cache()` + `gc.collect()`.
- **IPC**: ZMQ PUB socket en `tcp://*:5557`. THEIA escucha como SUB.
- **Determinismo**: Seeds fijos para reproducibilidad científica.
  - `torch.manual_seed(42)` + `np.random.seed(42)` en cada run.
- **Bare-metal**: Todo local. Nada web, nada cloud, nada TypeScript.

## STACK
- **Rust**: Core engine, DB, CLI, GUI (egui+wgpu), FFI (PyO3).
- **Python**: Research layer (simulaciones, ML, scoring).
- **SQLite**: Base de datos embebida (rusqlite bundled).
- **TOML**: Configuración (aether.toml).

## ARQUITECTURA
```
AETHER/
├── crates/
│   ├── aether-core/     ← Tipos fundamentales: Material, Experiment, Compatibility
│   ├── aether-db/       ← SQLite schema + CRUD operations
│   ├── aether-acq/      ← Hardware acquisition (stub, futuro)
│   ├── aether-ffi/      ← PyO3 bridge Rust↔Python
│   ├── aether-cli/      ← Terminal interface (clap)
│   └── aether-gui/      ← Native GUI (egui + wgpu)
├── research/            ← Python research layer
│   ├── quantum_annealing/
│   ├── reservoir_computing/
│   ├── em_simulation/
│   ├── compatibility/
│   └── knowledge/
└── data/                ← SQLite DB + experiment results
```

## REGLAS DE CÓDIGO
1. **NUNCA reescribas archivos completos**. Diff-only.
2. **VRAM**: Liberar siempre tras simulación. No dejar tensores huérfanos.
3. **Seeds**: Determinismo estricto en cada simulación.
4. **Type hints** obligatorios en Python.
5. **Rust**: `Result<T, E>`, no `unwrap()`. Documentar unsafe.
6. **Observaciones subjetivas**: Dato de primera clase con timestamp y condiciones.

## COMANDOS
```bash
# Build
cargo build --workspace --release

# Tests
cargo test --workspace
pytest tests/python/ -v

# CLI
cargo run -p aether-cli -- material add "Cuarzo" --category crystal
cargo run -p aether-cli -- material list
cargo run -p aether-cli -- experiment run --type compatibility
cargo run -p aether-cli -- compat matrix

# Python research (standalone)
python -m research.quantum_annealing.annealer
python -m research.reservoir_computing.esn
python -m research.em_simulation.resonance

# Clippy
cargo clippy --workspace -- -D warnings
```

## PROTOCOLO
- Opus: Diseño de nuevos módulos de investigación, arquitectura.
- Sonnet: Implementación, optimización, debugging numérico.
- Validación: `cargo test` + `pytest` ANTES de review.
