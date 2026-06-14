//! Configuration loading for AETHER LAB.
//!
//! Reads the `aether.toml` configuration file and deserialises it into
//! strongly-typed Rust structs.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Top-level AETHER configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AetherConfig {
    pub aether: AetherMeta,
    pub database: DatabaseConfig,
    pub ipc: IpcConfig,
    pub acquisition: AcquisitionConfig,
    pub research: ResearchConfig,
    pub gui: GuiConfig,
    pub output: OutputConfig,
}

/// Basic identity / version metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AetherMeta {
    pub name: String,
    pub version: String,
}

/// SQLite database settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Path to the SQLite database file.
    pub path: String,
    /// Hours between automatic backups.
    pub backup_interval_hours: u64,
}

/// Inter-process communication settings (ZMQ).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcConfig {
    pub zmq_pub_port: u16,
    pub zmq_sub_ports: Vec<u16>,
    pub protocol: String,
}

/// Sensor acquisition hardware settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcquisitionConfig {
    pub enabled: bool,
    pub serial_port: String,
    pub baud_rate: u32,
}

/// Research module configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchConfig {
    /// Path to the Python virtual environment.
    pub python_venv: String,
    pub quantum_annealing: QuantumAnnealingConfig,
    pub reservoir_computing: ReservoirComputingConfig,
    pub em_simulation: EmSimulationConfig,
    pub compatibility: CompatibilityConfig,
}

/// Quantum annealing solver settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumAnnealingConfig {
    pub enabled: bool,
    pub num_reads: u32,
    pub method: String,
}

/// Reservoir computing settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReservoirComputingConfig {
    pub enabled: bool,
    pub reservoir_size: u32,
    pub spectral_radius: f64,
    pub leak_rate: f64,
}

/// Electromagnetic simulation settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmSimulationConfig {
    pub enabled: bool,
    pub backend: String,
    pub resolution: u32,
}

/// Compatibility analysis settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityConfig {
    pub enabled: bool,
    pub weights: CompatibilityWeights,
}

/// Named weights for each compatibility dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityWeights {
    pub crystallographic: f64,
    pub thermal: f64,
    pub electromagnetic: f64,
    pub resonant: f64,
    pub piezoelectric: f64,
    pub mechanical: f64,
    pub quantum_annealing: f64,
    pub reservoir: f64,
    pub subjective: f64,
    pub custom: f64,
}

impl CompatibilityWeights {
    /// Convert to a HashMap for use with the compatibility engine.
    pub fn to_map(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("crystallographic".into(), self.crystallographic);
        m.insert("thermal".into(), self.thermal);
        m.insert("electromagnetic".into(), self.electromagnetic);
        m.insert("resonant".into(), self.resonant);
        m.insert("piezoelectric".into(), self.piezoelectric);
        m.insert("mechanical".into(), self.mechanical);
        m.insert("quantum_annealing".into(), self.quantum_annealing);
        m.insert("reservoir".into(), self.reservoir);
        m.insert("subjective".into(), self.subjective);
        m.insert("custom".into(), self.custom);
        m
    }
}

/// GUI theming / rendering settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuiConfig {
    pub theme: String,
    pub fps_target: u32,
}

/// Output path settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    pub db_path: String,
    pub report_path: String,
    pub export_path: String,
}

/// Load the AETHER configuration from a TOML file.
///
/// # Errors
///
/// Returns an error if the file cannot be read or parsed.
pub fn load_config(path: impl AsRef<Path>) -> anyhow::Result<AetherConfig> {
    let content = std::fs::read_to_string(path.as_ref())?;
    let config: AetherConfig = toml::from_str(&content)?;
    Ok(config)
}
