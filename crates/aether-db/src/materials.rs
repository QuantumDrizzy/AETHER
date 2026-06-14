//! CRUD operations for materials and observations.

use chrono::Utc;
use rusqlite::params;
use uuid::Uuid;

use aether_core::material::*;
use crate::{AetherDb, DbError, DbResult};

impl AetherDb {
    /// Insert a new material into the database.
    pub fn insert_material(&self, material: &Material) -> DbResult<()> {
        self.conn.execute(
            r#"INSERT INTO materials
                (id, name, category, classification, physical, electromagnetic,
                 mechanical, quantum, crystal, metamaterial, composition,
                 custom_properties, source, confidence, notes, created_at, updated_at)
               VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17)"#,
            params![
                material.id.to_string(),
                material.name,
                serde_json::to_string(&material.category)?,
                serde_json::to_string(&material.classification)?,
                serde_json::to_string(&material.physical)?,
                serde_json::to_string(&material.electromagnetic)?,
                serde_json::to_string(&material.mechanical)?,
                serde_json::to_string(&material.quantum)?,
                material.crystal.as_ref().map(|c| serde_json::to_string(c)).transpose()?,
                material.metamaterial.as_ref().map(|m| serde_json::to_string(m)).transpose()?,
                material.composition.as_ref().map(|c| serde_json::to_string(c)).transpose()?,
                serde_json::to_string(&material.custom_properties)?,
                material.source,
                material.confidence,
                material.notes,
                material.created_at.to_rfc3339(),
                material.updated_at.to_rfc3339(),
            ],
        )?;

        // Insert observations.
        for obs in &material.observations {
            self.insert_observation(&material.id, obs)?;
        }

        Ok(())
    }

    /// Retrieve a material by its UUID.
    pub fn get_material(&self, id: &Uuid) -> DbResult<Material> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM materials WHERE id = ?1",
        )?;

        let material = stmt.query_row(params![id.to_string()], |row| {
            Ok(row_to_material(row))
        })?.map_err(|e| DbError::Other(e.to_string()))?;

        // Load observations.
        let mut mat = material;
        mat.observations = self.get_observations(&mat.id)?;
        Ok(mat)
    }

    /// List all materials, optionally filtered by category.
    pub fn list_materials(&self, category: Option<&str>) -> DbResult<Vec<Material>> {
        let mut materials = Vec::new();

        if let Some(cat) = category {
            let mut stmt = self.conn.prepare(
                "SELECT * FROM materials WHERE category = ?1 ORDER BY name",
            )?;
            let rows = stmt.query_map(params![cat], |row| Ok(row_to_material(row)))?;
            for row in rows {
                let mut mat = row?.map_err(|e| DbError::Other(e.to_string()))?;
                mat.observations = self.get_observations(&mat.id)?;
                materials.push(mat);
            }
        } else {
            let mut stmt = self.conn.prepare(
                "SELECT * FROM materials ORDER BY name",
            )?;
            let rows = stmt.query_map([], |row| Ok(row_to_material(row)))?;
            for row in rows {
                let mut mat = row?.map_err(|e| DbError::Other(e.to_string()))?;
                mat.observations = self.get_observations(&mat.id)?;
                materials.push(mat);
            }
        }

        Ok(materials)
    }

    /// Update an existing material (full replace).
    pub fn update_material(&self, material: &Material) -> DbResult<()> {
        let rows = self.conn.execute(
            r#"UPDATE materials SET
                name=?2, category=?3, classification=?4, physical=?5,
                electromagnetic=?6, mechanical=?7, quantum=?8, crystal=?9,
                metamaterial=?10, composition=?11, custom_properties=?12,
                source=?13, confidence=?14, notes=?15, updated_at=?16
               WHERE id=?1"#,
            params![
                material.id.to_string(),
                material.name,
                serde_json::to_string(&material.category)?,
                serde_json::to_string(&material.classification)?,
                serde_json::to_string(&material.physical)?,
                serde_json::to_string(&material.electromagnetic)?,
                serde_json::to_string(&material.mechanical)?,
                serde_json::to_string(&material.quantum)?,
                material.crystal.as_ref().map(|c| serde_json::to_string(c)).transpose()?,
                material.metamaterial.as_ref().map(|m| serde_json::to_string(m)).transpose()?,
                material.composition.as_ref().map(|c| serde_json::to_string(c)).transpose()?,
                serde_json::to_string(&material.custom_properties)?,
                material.source,
                material.confidence,
                material.notes,
                Utc::now().to_rfc3339(),
            ],
        )?;

        if rows == 0 {
            return Err(DbError::NotFound(format!("Material {}", material.id)));
        }

        // Replace observations.
        self.conn.execute(
            "DELETE FROM observations WHERE material_id = ?1",
            params![material.id.to_string()],
        )?;
        for obs in &material.observations {
            self.insert_observation(&material.id, obs)?;
        }

        Ok(())
    }

    /// Delete a material by UUID.
    pub fn delete_material(&self, id: &Uuid) -> DbResult<()> {
        let rows = self.conn.execute(
            "DELETE FROM materials WHERE id = ?1",
            params![id.to_string()],
        )?;

        if rows == 0 {
            return Err(DbError::NotFound(format!("Material {}", id)));
        }
        Ok(())
    }

    /// Search materials by name (case-insensitive substring match).
    pub fn search_materials(&self, query: &str) -> DbResult<Vec<Material>> {
        let pattern = format!("%{}%", query);
        let mut stmt = self.conn.prepare(
            "SELECT * FROM materials WHERE name LIKE ?1 ORDER BY name",
        )?;
        let rows = stmt.query_map(params![pattern], |row| Ok(row_to_material(row)))?;

        let mut materials = Vec::new();
        for row in rows {
            let mut mat = row?.map_err(|e| DbError::Other(e.to_string()))?;
            mat.observations = self.get_observations(&mat.id)?;
            materials.push(mat);
        }
        Ok(materials)
    }

    // ------------------------------------------------------------------
    // Observations (internal helpers)
    // ------------------------------------------------------------------

    fn insert_observation(&self, material_id: &Uuid, obs: &SubjectiveObservation) -> DbResult<()> {
        self.conn.execute(
            r#"INSERT INTO observations
                (material_id, timestamp, observer, category, intensity,
                 description, conditions, confidence)
               VALUES (?1,?2,?3,?4,?5,?6,?7,?8)"#,
            params![
                material_id.to_string(),
                obs.timestamp.to_rfc3339(),
                obs.observer,
                serde_json::to_string(&obs.category)?,
                obs.intensity,
                obs.description,
                serde_json::to_string(&obs.conditions)?,
                obs.confidence,
            ],
        )?;
        Ok(())
    }

    fn get_observations(&self, material_id: &Uuid) -> DbResult<Vec<SubjectiveObservation>> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM observations WHERE material_id = ?1 ORDER BY timestamp",
        )?;

        let rows = stmt.query_map(params![material_id.to_string()], |row| {
            let timestamp_str: String = row.get(2)?;
            let category_str: String = row.get(4)?;
            let conditions_str: String = row.get(7)?;

            Ok((timestamp_str, row.get::<_, String>(3)?, category_str,
                row.get::<_, f64>(5)?, row.get::<_, String>(6)?,
                conditions_str, row.get::<_, f64>(8)?))
        })?;

        let mut observations = Vec::new();
        for row in rows {
            let (ts, observer, cat, intensity, desc, cond, conf) = row?;
            observations.push(SubjectiveObservation {
                timestamp: chrono::DateTime::parse_from_rfc3339(&ts)
                    .map_err(|e| DbError::Other(e.to_string()))?
                    .with_timezone(&chrono::Utc),
                observer,
                category: serde_json::from_str(&cat)
                    .map_err(|e| DbError::Other(e.to_string()))?,
                intensity,
                description: desc,
                conditions: serde_json::from_str(&cond)
                    .map_err(|e| DbError::Other(e.to_string()))?,
                confidence: conf,
            });
        }
        Ok(observations)
    }
}

/// Convert a SQLite row to a Material (without observations).
fn row_to_material(row: &rusqlite::Row<'_>) -> Result<Material, String> {
    let id_str: String = row.get(0).map_err(|e| e.to_string())?;
    let name: String = row.get(1).map_err(|e| e.to_string())?;
    let category_str: String = row.get(2).map_err(|e| e.to_string())?;
    let classification_str: String = row.get(3).map_err(|e| e.to_string())?;
    let physical_str: String = row.get(4).map_err(|e| e.to_string())?;
    let em_str: String = row.get(5).map_err(|e| e.to_string())?;
    let mech_str: String = row.get(6).map_err(|e| e.to_string())?;
    let quantum_str: String = row.get(7).map_err(|e| e.to_string())?;
    let crystal_str: Option<String> = row.get(8).map_err(|e| e.to_string())?;
    let meta_str: Option<String> = row.get(9).map_err(|e| e.to_string())?;
    let comp_str: Option<String> = row.get(10).map_err(|e| e.to_string())?;
    let custom_str: String = row.get(11).map_err(|e| e.to_string())?;
    let source: Option<String> = row.get(12).map_err(|e| e.to_string())?;
    let confidence: Option<f64> = row.get(13).map_err(|e| e.to_string())?;
    let notes: Option<String> = row.get(14).map_err(|e| e.to_string())?;
    let created_str: String = row.get(15).map_err(|e| e.to_string())?;
    let updated_str: String = row.get(16).map_err(|e| e.to_string())?;

    Ok(Material {
        id: Uuid::parse_str(&id_str).map_err(|e| e.to_string())?,
        name,
        category: serde_json::from_str(&category_str).map_err(|e| e.to_string())?,
        classification: serde_json::from_str(&classification_str).map_err(|e| e.to_string())?,
        physical: serde_json::from_str(&physical_str).map_err(|e| e.to_string())?,
        electromagnetic: serde_json::from_str(&em_str).map_err(|e| e.to_string())?,
        mechanical: serde_json::from_str(&mech_str).map_err(|e| e.to_string())?,
        quantum: serde_json::from_str(&quantum_str).map_err(|e| e.to_string())?,
        crystal: crystal_str.map(|s| serde_json::from_str(&s)).transpose().map_err(|e| e.to_string())?,
        metamaterial: meta_str.map(|s| serde_json::from_str(&s)).transpose().map_err(|e| e.to_string())?,
        composition: comp_str.map(|s| serde_json::from_str(&s)).transpose().map_err(|e| e.to_string())?,
        observations: Vec::new(), // Loaded separately.
        custom_properties: serde_json::from_str(&custom_str).map_err(|e| e.to_string())?,
        source,
        confidence,
        notes,
        created_at: chrono::DateTime::parse_from_rfc3339(&created_str)
            .map_err(|e| e.to_string())?
            .with_timezone(&chrono::Utc),
        updated_at: chrono::DateTime::parse_from_rfc3339(&updated_str)
            .map_err(|e| e.to_string())?
            .with_timezone(&chrono::Utc),
    })
}
