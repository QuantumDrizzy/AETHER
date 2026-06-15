//! CRUD operations for experiments.

use rusqlite::params;
use uuid::Uuid;

use aether_core::experiment::*;
use crate::{AetherDb, DbError, DbResult};

impl AetherDb {
    /// Insert a new experiment.
    pub fn insert_experiment(&self, exp: &Experiment) -> DbResult<()> {
        self.conn.execute(
            r#"INSERT INTO experiments
                (id, name, description, config, status, result, error,
                 created_at, started_at, completed_at)
               VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)"#,
            params![
                exp.id.to_string(),
                exp.name,
                exp.description,
                serde_json::to_string(&exp.config)?,
                serde_json::to_string(&exp.status)?,
                exp.result.as_ref().map(serde_json::to_string).transpose()?,
                exp.error,
                exp.created_at.to_rfc3339(),
                exp.started_at.map(|t| t.to_rfc3339()),
                exp.completed_at.map(|t| t.to_rfc3339()),
            ],
        )?;
        Ok(())
    }

    /// Retrieve an experiment by UUID.
    pub fn get_experiment(&self, id: &Uuid) -> DbResult<Experiment> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM experiments WHERE id = ?1",
        )?;

        let exp = stmt.query_row(params![id.to_string()], |row| {
            Ok(row_to_experiment(row))
        })?.map_err(|e| DbError::Other(e.to_string()))?;

        Ok(exp)
    }

    /// List all experiments.
    pub fn list_experiments(&self) -> DbResult<Vec<Experiment>> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM experiments ORDER BY created_at DESC",
        )?;
        let rows = stmt.query_map([], |row| Ok(row_to_experiment(row)))?;

        let mut experiments = Vec::new();
        for row in rows {
            let exp = row?.map_err(|e| DbError::Other(e.to_string()))?;
            experiments.push(exp);
        }
        Ok(experiments)
    }

    /// Update the status of an experiment.
    pub fn update_experiment_status(
        &self,
        id: &Uuid,
        status: &ExperimentStatus,
    ) -> DbResult<()> {
        let now = chrono::Utc::now().to_rfc3339();
        let status_str = serde_json::to_string(status)?;

        let rows = match status {
            ExperimentStatus::Running => {
                self.conn.execute(
                    "UPDATE experiments SET status=?2, started_at=?3 WHERE id=?1",
                    params![id.to_string(), status_str, now],
                )?
            }
            ExperimentStatus::Completed | ExperimentStatus::Failed => {
                self.conn.execute(
                    "UPDATE experiments SET status=?2, completed_at=?3 WHERE id=?1",
                    params![id.to_string(), status_str, now],
                )?
            }
            _ => {
                self.conn.execute(
                    "UPDATE experiments SET status=?2 WHERE id=?1",
                    params![id.to_string(), status_str],
                )?
            }
        };

        if rows == 0 {
            return Err(DbError::NotFound(format!("Experiment {}", id)));
        }
        Ok(())
    }

    /// Set the result of a completed experiment.
    pub fn set_experiment_result(
        &self,
        id: &Uuid,
        result: &ExperimentResult,
    ) -> DbResult<()> {
        let rows = self.conn.execute(
            r#"UPDATE experiments SET
                status=?2, result=?3, completed_at=?4
               WHERE id=?1"#,
            params![
                id.to_string(),
                serde_json::to_string(&ExperimentStatus::Completed)?,
                serde_json::to_string(result)?,
                chrono::Utc::now().to_rfc3339(),
            ],
        )?;

        if rows == 0 {
            return Err(DbError::NotFound(format!("Experiment {}", id)));
        }
        Ok(())
    }
}

fn row_to_experiment(row: &rusqlite::Row<'_>) -> Result<Experiment, String> {
    let id_str: String = row.get(0).map_err(|e| e.to_string())?;
    let name: String = row.get(1).map_err(|e| e.to_string())?;
    let description: Option<String> = row.get(2).map_err(|e| e.to_string())?;
    let config_str: String = row.get(3).map_err(|e| e.to_string())?;
    let status_str: String = row.get(4).map_err(|e| e.to_string())?;
    let result_str: Option<String> = row.get(5).map_err(|e| e.to_string())?;
    let error: Option<String> = row.get(6).map_err(|e| e.to_string())?;
    let created_str: String = row.get(7).map_err(|e| e.to_string())?;
    let started_str: Option<String> = row.get(8).map_err(|e| e.to_string())?;
    let completed_str: Option<String> = row.get(9).map_err(|e| e.to_string())?;

    Ok(Experiment {
        id: Uuid::parse_str(&id_str).map_err(|e| e.to_string())?,
        name,
        description,
        config: serde_json::from_str(&config_str).map_err(|e| e.to_string())?,
        status: serde_json::from_str(&status_str).map_err(|e| e.to_string())?,
        result: result_str
            .map(|s| serde_json::from_str(&s))
            .transpose()
            .map_err(|e| e.to_string())?,
        error,
        created_at: chrono::DateTime::parse_from_rfc3339(&created_str)
            .map_err(|e| e.to_string())?
            .with_timezone(&chrono::Utc),
        started_at: started_str
            .map(|s| chrono::DateTime::parse_from_rfc3339(&s).map(|t| t.with_timezone(&chrono::Utc)))
            .transpose()
            .map_err(|e| e.to_string())?,
        completed_at: completed_str
            .map(|s| chrono::DateTime::parse_from_rfc3339(&s).map(|t| t.with_timezone(&chrono::Utc)))
            .transpose()
            .map_err(|e| e.to_string())?,
    })
}
