//! # AETHER Database
//!
//! SQLite persistence layer for the AETHER materials research platform.
//! Provides CRUD operations for materials, experiments, and knowledge entries.

pub mod schema;
pub mod materials;
pub mod experiments;
pub mod knowledge;

use std::path::Path;

use rusqlite::Connection;
use thiserror::Error;

/// Database errors.
#[derive(Debug, Error)]
pub enum DbError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Database error: {0}")]
    Other(String),
}

/// Result alias for database operations.
pub type DbResult<T> = Result<T, DbError>;

/// Main database handle for AETHER.
///
/// Wraps a single SQLite connection. In production this would be a connection
/// pool; for now we keep things simple with a single synchronous connection.
pub struct AetherDb {
    pub(crate) conn: Connection,
}

impl AetherDb {
    /// Open (or create) the database at the given path and run migrations.
    pub fn init(path: impl AsRef<Path>) -> DbResult<Self> {
        let conn = Connection::open(path)?;

        // Enable WAL for better concurrent-read performance.
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;

        let db = Self { conn };
        schema::migrate(&db)?;
        Ok(db)
    }

    /// Open an in-memory database (useful for testing).
    pub fn in_memory() -> DbResult<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch("PRAGMA foreign_keys=ON;")?;
        let db = Self { conn };
        schema::migrate(&db)?;
        Ok(db)
    }
}

impl std::fmt::Debug for AetherDb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AetherDb").finish_non_exhaustive()
    }
}
