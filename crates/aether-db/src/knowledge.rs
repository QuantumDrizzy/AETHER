//! CRUD operations for knowledge entries.

use rusqlite::params;
use uuid::Uuid;

use aether_core::knowledge::*;
use crate::{AetherDb, DbError, DbResult};

impl AetherDb {
    /// Insert a new knowledge entry.
    pub fn insert_entry(&self, entry: &KnowledgeEntry) -> DbResult<()> {
        self.conn.execute(
            r#"INSERT INTO knowledge_entries
                (id, entry_type, title, description, evidence, confidence,
                 tags, source_experiments, created_at, updated_at)
               VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)"#,
            params![
                entry.id.to_string(),
                serde_json::to_string(&entry.entry_type)?,
                entry.title,
                entry.description,
                serde_json::to_string(&entry.evidence)?,
                entry.confidence,
                serde_json::to_string(&entry.tags)?,
                serde_json::to_string(&entry.source_experiments)?,
                entry.created_at.to_rfc3339(),
                entry.updated_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    /// Retrieve a knowledge entry by UUID.
    pub fn get_entry(&self, id: &Uuid) -> DbResult<KnowledgeEntry> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM knowledge_entries WHERE id = ?1",
        )?;

        let entry = stmt.query_row(params![id.to_string()], |row| {
            Ok(row_to_entry(row))
        })?.map_err(|e| DbError::Other(e.to_string()))?;

        Ok(entry)
    }

    /// List all knowledge entries.
    pub fn list_entries(&self) -> DbResult<Vec<KnowledgeEntry>> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM knowledge_entries ORDER BY updated_at DESC",
        )?;
        let rows = stmt.query_map([], |row| Ok(row_to_entry(row)))?;

        let mut entries = Vec::new();
        for row in rows {
            let entry = row?.map_err(|e| DbError::Other(e.to_string()))?;
            entries.push(entry);
        }
        Ok(entries)
    }

    /// Search knowledge entries by tag (case-insensitive substring in JSON array).
    pub fn search_by_tags(&self, tag: &str) -> DbResult<Vec<KnowledgeEntry>> {
        let pattern = format!("%\"{}%", tag);
        let mut stmt = self.conn.prepare(
            "SELECT * FROM knowledge_entries WHERE tags LIKE ?1 ORDER BY confidence DESC",
        )?;
        let rows = stmt.query_map(params![pattern], |row| Ok(row_to_entry(row)))?;

        let mut entries = Vec::new();
        for row in rows {
            let entry = row?.map_err(|e| DbError::Other(e.to_string()))?;
            entries.push(entry);
        }
        Ok(entries)
    }

    /// Get knowledge entries linked to a specific material via source experiments.
    ///
    /// This performs a substring search on the source_experiments JSON array
    /// for any experiment whose material_ids contain the given material UUID.
    pub fn get_entries_for_material(&self, material_id: &Uuid) -> DbResult<Vec<KnowledgeEntry>> {
        // First, find experiment IDs that reference this material.
        let mut exp_stmt = self.conn.prepare(
            "SELECT id FROM experiments WHERE config LIKE ?1",
        )?;
        let pattern = format!("%{}%", material_id);
        let exp_rows = exp_stmt.query_map(params![pattern], |row| {
            row.get::<_, String>(0)
        })?;

        let mut exp_ids: Vec<String> = Vec::new();
        for row in exp_rows {
            exp_ids.push(row?);
        }

        if exp_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Now find knowledge entries that reference any of these experiments.
        let mut entries = Vec::new();
        for exp_id in &exp_ids {
            let search = format!("%{}%", exp_id);
            let mut stmt = self.conn.prepare(
                "SELECT * FROM knowledge_entries WHERE source_experiments LIKE ?1",
            )?;
            let rows = stmt.query_map(params![search], |row| Ok(row_to_entry(row)))?;
            for row in rows {
                let entry = row?.map_err(|e| DbError::Other(e.to_string()))?;
                // Deduplicate.
                if !entries.iter().any(|e: &KnowledgeEntry| e.id == entry.id) {
                    entries.push(entry);
                }
            }
        }

        Ok(entries)
    }
}

fn row_to_entry(row: &rusqlite::Row<'_>) -> Result<KnowledgeEntry, String> {
    let id_str: String = row.get(0).map_err(|e| e.to_string())?;
    let type_str: String = row.get(1).map_err(|e| e.to_string())?;
    let title: String = row.get(2).map_err(|e| e.to_string())?;
    let description: String = row.get(3).map_err(|e| e.to_string())?;
    let evidence_str: String = row.get(4).map_err(|e| e.to_string())?;
    let confidence: f64 = row.get(5).map_err(|e| e.to_string())?;
    let tags_str: String = row.get(6).map_err(|e| e.to_string())?;
    let source_str: String = row.get(7).map_err(|e| e.to_string())?;
    let created_str: String = row.get(8).map_err(|e| e.to_string())?;
    let updated_str: String = row.get(9).map_err(|e| e.to_string())?;

    Ok(KnowledgeEntry {
        id: Uuid::parse_str(&id_str).map_err(|e| e.to_string())?,
        entry_type: serde_json::from_str(&type_str).map_err(|e| e.to_string())?,
        title,
        description,
        evidence: serde_json::from_str(&evidence_str).map_err(|e| e.to_string())?,
        confidence,
        tags: serde_json::from_str(&tags_str).map_err(|e| e.to_string())?,
        source_experiments: serde_json::from_str(&source_str).map_err(|e| e.to_string())?,
        created_at: chrono::DateTime::parse_from_rfc3339(&created_str)
            .map_err(|e| e.to_string())?
            .with_timezone(&chrono::Utc),
        updated_at: chrono::DateTime::parse_from_rfc3339(&updated_str)
            .map_err(|e| e.to_string())?
            .with_timezone(&chrono::Utc),
    })
}
