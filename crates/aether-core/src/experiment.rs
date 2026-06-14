//! Experiment definition and lifecycle types.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// The kind of experiment being conducted.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExperimentType {
    Measurement,
    Simulation,
    Compatibility,
    QuantumAnnealing,
    ReservoirComputing,
    EMSimulation,
    Custom(String),
}

impl std::fmt::Display for ExperimentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Custom(s) => write!(f, "Custom({})", s),
            other => write!(f, "{:?}", other),
        }
    }
}

/// Lifecycle state of an experiment.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExperimentStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

impl std::fmt::Display for ExperimentStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Configuration for a single experiment run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    /// What kind of experiment this is.
    pub experiment_type: ExperimentType,
    /// Arbitrary key-value parameters.
    pub parameters: HashMap<String, serde_json::Value>,
    /// Materials involved in this experiment.
    pub material_ids: Vec<Uuid>,
    /// Optional RNG seed for reproducibility.
    pub seed: Option<u64>,
}

/// Output of a completed experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResult {
    /// Primary result data (experiment-specific JSON blob).
    pub data: serde_json::Value,
    /// Named scalar metrics produced by the experiment.
    pub metrics: HashMap<String, f64>,
    /// Paths to generated artifact files (plots, CSVs, etc.).
    pub artifacts: Vec<String>,
    /// Wall-clock duration of the experiment in milliseconds.
    pub duration_ms: u64,
}

/// A single experiment in the AETHER platform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    /// Unique identifier.
    pub id: Uuid,
    /// Human-readable name.
    pub name: String,
    /// Optional longer description.
    pub description: Option<String>,
    /// Configuration used for this run.
    pub config: ExperimentConfig,
    /// Current lifecycle status.
    pub status: ExperimentStatus,
    /// Result (populated upon successful completion).
    pub result: Option<ExperimentResult>,
    /// Error message (populated upon failure).
    pub error: Option<String>,
    /// When the experiment record was created.
    pub created_at: DateTime<Utc>,
    /// When the experiment began executing.
    pub started_at: Option<DateTime<Utc>>,
    /// When the experiment finished (success or failure).
    pub completed_at: Option<DateTime<Utc>>,
}

impl Experiment {
    /// Create a new pending experiment.
    pub fn new(
        name: impl Into<String>,
        config: ExperimentConfig,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            description: None,
            config,
            status: ExperimentStatus::Pending,
            result: None,
            error: None,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
        }
    }

    /// Mark the experiment as running.
    pub fn start(&mut self) {
        self.status = ExperimentStatus::Running;
        self.started_at = Some(Utc::now());
    }

    /// Mark the experiment as completed with a result.
    pub fn complete(&mut self, result: ExperimentResult) {
        self.status = ExperimentStatus::Completed;
        self.result = Some(result);
        self.completed_at = Some(Utc::now());
    }

    /// Mark the experiment as failed.
    pub fn fail(&mut self, error: impl Into<String>) {
        self.status = ExperimentStatus::Failed;
        self.error = Some(error.into());
        self.completed_at = Some(Utc::now());
    }
}
