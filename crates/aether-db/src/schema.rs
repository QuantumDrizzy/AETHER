//! SQLite schema definitions and migration logic.

use crate::{AetherDb, DbResult};

/// Run all migrations (idempotent — uses IF NOT EXISTS everywhere).
pub fn migrate(db: &AetherDb) -> DbResult<()> {
    db.conn.execute_batch(SCHEMA)?;
    tracing::info!("Database schema migration complete");
    Ok(())
}

const SCHEMA: &str = r#"
-- ===================================================================
-- AETHER LAB — SQLite Schema
-- ===================================================================

-- Materials ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS materials (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    category        TEXT NOT NULL,
    classification  TEXT NOT NULL DEFAULT '[]',     -- JSON array
    physical        TEXT NOT NULL DEFAULT '{}',     -- JSON object
    electromagnetic TEXT NOT NULL DEFAULT '{}',     -- JSON object
    mechanical      TEXT NOT NULL DEFAULT '{}',     -- JSON object
    quantum         TEXT NOT NULL DEFAULT '{}',     -- JSON object
    crystal         TEXT,                            -- JSON object or NULL
    metamaterial    TEXT,                            -- JSON object or NULL
    composition     TEXT,                            -- JSON object or NULL
    custom_properties TEXT NOT NULL DEFAULT '{}',   -- JSON object
    source          TEXT,
    confidence      REAL,
    notes           TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_materials_name     ON materials(name);
CREATE INDEX IF NOT EXISTS idx_materials_category ON materials(category);

-- Observations ------------------------------------------------------
CREATE TABLE IF NOT EXISTS observations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    material_id TEXT NOT NULL REFERENCES materials(id) ON DELETE CASCADE,
    timestamp   TEXT NOT NULL,
    observer    TEXT NOT NULL,
    category    TEXT NOT NULL,
    intensity   REAL NOT NULL,
    description TEXT NOT NULL,
    conditions  TEXT NOT NULL DEFAULT '{}',         -- JSON object
    confidence  REAL NOT NULL DEFAULT 0.5
);

CREATE INDEX IF NOT EXISTS idx_observations_material ON observations(material_id);

-- Experiments -------------------------------------------------------
CREATE TABLE IF NOT EXISTS experiments (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT,
    config          TEXT NOT NULL,                  -- JSON ExperimentConfig
    status          TEXT NOT NULL DEFAULT 'Pending',
    result          TEXT,                            -- JSON ExperimentResult
    error           TEXT,
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    completed_at    TEXT
);

CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);

-- Compatibility results ---------------------------------------------
CREATE TABLE IF NOT EXISTS compatibilities (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    material_a      TEXT NOT NULL REFERENCES materials(id),
    material_b      TEXT NOT NULL REFERENCES materials(id),
    overall_score   REAL NOT NULL,
    dimensions      TEXT NOT NULL,                  -- JSON array
    synergies       TEXT NOT NULL DEFAULT '[]',
    conflicts       TEXT NOT NULL DEFAULT '[]',
    recommendation  TEXT NOT NULL,
    computed_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_compat_pair ON compatibilities(material_a, material_b);

-- Knowledge entries -------------------------------------------------
CREATE TABLE IF NOT EXISTS knowledge_entries (
    id                  TEXT PRIMARY KEY,
    entry_type          TEXT NOT NULL,
    title               TEXT NOT NULL,
    description         TEXT NOT NULL,
    evidence            TEXT NOT NULL DEFAULT '[]', -- JSON array
    confidence          REAL NOT NULL DEFAULT 0.5,
    tags                TEXT NOT NULL DEFAULT '[]', -- JSON array
    source_experiments  TEXT NOT NULL DEFAULT '[]', -- JSON array of UUIDs
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_knowledge_type ON knowledge_entries(entry_type);

-- Signal captures ---------------------------------------------------
CREATE TABLE IF NOT EXISTS signal_captures (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    channel     TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    data_values TEXT NOT NULL,                      -- JSON array of f64
    metadata    TEXT NOT NULL DEFAULT '{}',
    material_id TEXT REFERENCES materials(id)
);

CREATE INDEX IF NOT EXISTS idx_signals_channel   ON signal_captures(channel);
CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signal_captures(timestamp);
"#;
