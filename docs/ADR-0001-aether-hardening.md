# ADR-0001: Hardening AETHER from scaffold to a real, validated lab

**Status:** Accepted
**Date:** 2026-06-14
**Deciders:** Antonio (QuantumDrizzy)

## Context

AETHER was bootstrapped quickly (Antigravity) into an ambitious shape: a Rust
workspace (core/db/acq/ffi/cli/gui) + a Python research layer (FDTD EM, reservoir
computing, simulated/quantum annealing) + SQLite, aimed at advanced-materials
research (crystals, metamaterials, nanomaterials). The goal is to bring it to the
level of SUBSTRATE: a **real, validated** lab — not impressive-looking scaffolding.

An honest audit (2026-06-14) found a split between genuine work and façade:

**Real and runnable:**
- `research/em_simulation` — 1-D FDTD, Yee algorithm, PML boundaries, Courant factor.
- `research/reservoir_computing` — leaky-integrator ESN with spectral-radius rescaling
  and ridge readout.
- `research/quantum_annealing` — QUBO formulator + simulated annealing (Metropolis),
  with optional OpenJij SQA.
- `crates/aether-core` — `Material`/`Experiment` types; **`compatibility.rs` is a real
  6-dimension engine** (lattice mismatch, CTE+melting point, dielectric+conductivity,
  resonant-frequency overlap, piezoelectric coefficient similarity, mechanical
  moduli), with `0.5` used only as a missing-data fallback.
- `crates/aether-db` — SQLite schema + CRUD.

**Façade / disconnected / unvalidated:**
- The real Rust compatibility engine **is not called by anything**: the CLI
  `compat compute`/`compat matrix` commands only `println!`, and the PyO3 FFI
  (`aether-ffi`) exposes nothing but `Material.name`.
- A **second**, standalone compatibility implementation exists in Python
  (`research/compatibility`) where **5 of 7 metrics were hardcoded `return 0.5`** —
  a fake neutral score. (Already neutralized: they now return `None` and the scorer
  omits them.)
- Config drift: `aether.toml` declared **10** weighted dimensions; code computes 6
  (Rust) / 2 (Python). (Already aligned to the 6 real Rust dims.)
- Docs (README/CLAUDE.md) claimed a working "10-dimension multi-metric engine".
  (Already corrected to reflect reality.)
- **Zero tests / no validation harness** in the entire repo — so no number it
  produces can be trusted.

## Decision

1. **The Rust `aether-core::compatibility` engine is the single source of truth**
   for compatibility scoring. The Python `research/compatibility` package is a
   placeholder and must NOT grow a parallel implementation; once the FFI exists it
   will call the Rust engine. Until then it is explicitly marked as such and only
   aggregates the dimensions it actually computes.
2. **No claims without validation.** README/CLAUDE state real status; every
   performance or accuracy claim must be backed by a test against an analytic or
   published reference. Unimplemented pieces are marked `[KNOWN_LIMIT]`.
3. **Config mirrors code.** No phantom dimensions/weights for things no code computes.
4. Bring it to SUBSTRATE level via the phased plan below, validation first.

## Trade-off / why this way

- Two competing compat implementations is the core smell. Picking Rust as
  authoritative (it's the better one, it owns the types, and FFI is the intended
  bridge) removes the duplication honestly instead of patching the worse copy.
- Validation-first (P1 before P2 feature work) is what separates a "lab" from a
  tech demo, and matches the engineering-honesty standard: real benchmarks vs a
  stated baseline, `[KNOWN_LIMIT]` over hidden gaps.

## Consequences

**Easier:** trustworthy outputs; a clear single engine; honest README that won't
embarrass under scrutiny.
**Harder:** must build a test/validation harness before claiming results; FFI/CLI
wiring becomes required work, not optional.
**Revisit:** whether the Python compat package should be deleted outright once the
FFI bridge lands (likely yes).

## Plan (phases)

**P0 — Honesty & safety (this ADR).** ✅ git init + push (private); neutralize the
Python façade (None + skip); align `aether.toml` to 6 real dims; correct
README/CLAUDE; this ADR.

**P1 — Validation harness (makes it a lab).**
- FDTD: vacuum phase velocity ≈ c; dielectric-slab reflection vs Fresnel; PML
  residual reflection below a stated threshold.
- ESN: NRMSE on Mackey-Glass / NARMA-10 within published ranges (fixed seed).
- SA: on small QUBO, recover the brute-force optimum in ≥ X% of seeded runs.
- Rust: unit tests for `aether-db` CRUD, `Material` (de)serialization round-trips,
  and each `compatibility` dimension on known inputs.

**P2 — Real science in the compat layer.**
- Wire the Rust engine into the CLI (`compat compute`/`matrix` call it for real).
- Implement remaining physics where it adds value (e.g. EM/acoustic impedance
  matching, bandgap alignment) with a unit test each — in Rust, the source of truth.

**P3 — Ecosystem integration.**
- PyO3 FFI: expose the DB + compatibility engine to Python; round-trip tested.
- ZMQ PUB/SUB bridge to SUBSTRATE, tested end-to-end.
- GUI to the level of sibyl's egui pass; align `eframe` (0.27 → current).

## Status — ARCHIVABLE (2026-06-15)

AETHER reached a coherent, validated, usable milestone — **parked, not deleted**
(stays on the desktop with the rest of the ecosystem):
- **35 validation tests** green: 22 Python (FDTD, graphene + DOS, general N×N TB
  solver CPU/GPU, ESN, SA, SSH) + 13 Rust (compat engine, physics, db CRUD,
  materials library). Whole workspace **clippy-clean**.
- Scientific spine: electronic structure (graphene TB, DOS, general solver),
  topological SSH, photonic FDTD; honest GPU benchmark (66× at N=256, 0.03× at N=2).
- Real **curated materials library** (carbon allotropes incl. carbyne, bismuth,
  2D materials, …) seeded into the DB.
- **Native GUI**: dashboard + seed, materials browser, Compatibility wired to the
  Rust engine, live graphene band-structure plot.
- A real SA delta-energy double-counting bug was caught & fixed by the harness.

Deferred (P3, optional, beyond archivable): real PyO3 FFI bridge, ZMQ to
SUBSTRATE, richer GUI physics panels (DOS/FDTD), enrich `material add`.
Next ecosystem step: **Spectra** (shared spectral spine), then DRIFT/computronium.

## Action items
1. [x] P0 — git + push private; neutralize Python façade; align config; fix docs.
2. [x] P1 — validation harness (FDTD/ESN/SA + graphene/SSH + Rust unit tests).
3. [x] P2 — wire `compat` CLI → Rust engine.
4. [x] GUI — clean theme, wired Compatibility, live band plot.
5. [ ] P3 (deferred) — real FFI + ZMQ + extra GUI panels; then decide on
   `research/compatibility`.
