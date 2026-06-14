//! Universal material model for AETHER LAB.
//!
//! Represents any material — from conventional crystals and alloys to
//! metamaterials, nanomaterials, and theoretical constructs — with a
//! comprehensive property schema that spans physical, electromagnetic,
//! mechanical, quantum, and crystallographic domains.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

/// Top-level material classification.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MaterialCategory {
    Crystal,
    Metamaterial,
    Nanomaterial,
    Alloy,
    Polymer,
    Composite,
    Biological,
    Theoretical,
    Custom(String),
}

impl std::fmt::Display for MaterialCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Custom(s) => write!(f, "Custom({})", s),
            other => write!(f, "{:?}", other),
        }
    }
}

/// Seven crystal systems.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CrystalSystem {
    Cubic,
    Tetragonal,
    Orthorhombic,
    Hexagonal,
    Trigonal,
    Monoclinic,
    Triclinic,
}

/// Metamaterial unit-cell geometry type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GeometryType {
    SplitRingResonator,
    Fishnet,
    Wire,
    Helix,
    Chiral,
    Custom(String),
}

/// Category of a subjective / qualitative observation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ObservationCategory {
    Tactile,
    Visual,
    Auditory,
    Thermal,
    Energetic,
    Electromagnetic,
    Resonant,
    Intuitive,
    Custom(String),
}

/// Dynamically-typed property value stored in the custom property map.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PropertyValue {
    Float(f64),
    Int(i64),
    Text(String),
    Bool(bool),
    Vector(Vec<f64>),
    Complex(Complex64),
}

// ---------------------------------------------------------------------------
// Property structs
// ---------------------------------------------------------------------------

/// Bulk physical / thermal properties.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PhysicalProperties {
    /// Density in kg/m³.
    pub density: Option<f64>,
    /// Melting point in Kelvin.
    pub melting_point: Option<f64>,
    /// Thermal conductivity in W/(m·K).
    pub thermal_conductivity: Option<f64>,
    /// Coefficient of thermal expansion in 1/K.
    pub thermal_expansion: Option<f64>,
    /// Specific heat capacity in J/(kg·K).
    pub specific_heat: Option<f64>,
    /// Human-readable colour description or hex code.
    pub color: Option<String>,
    /// Optical transparency fraction [0, 1].
    pub transparency: Option<f64>,
}

/// Electromagnetic / optical properties.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ElectromagneticProperties {
    /// Relative dielectric constant (real part at low frequency).
    pub dielectric_constant: Option<f64>,
    /// Relative magnetic permeability.
    pub magnetic_permeability: Option<f64>,
    /// Electrical conductivity in S/m.
    pub electrical_conductivity: Option<f64>,
    /// Refractive index (real part).
    pub refractive_index: Option<f64>,
    /// Piezoelectric coefficients (d-tensor components, pC/N).
    pub piezoelectric_coeff: Option<Vec<f64>>,
    /// Notable resonant frequencies in Hz.
    pub resonant_frequencies: Vec<f64>,
    /// Absorption spectrum as (frequency_Hz, absorption_coefficient) pairs.
    pub em_absorption_spectrum: Vec<(f64, f64)>,
}

/// Mechanical properties.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MechanicalProperties {
    /// Mohs hardness (1-10).
    pub hardness_mohs: Option<f64>,
    /// Young's modulus in GPa.
    pub youngs_modulus: Option<f64>,
    /// Ultimate tensile strength in MPa.
    pub tensile_strength: Option<f64>,
    /// Fracture toughness in MPa·√m.
    pub fracture_toughness: Option<f64>,
}

/// Quantum / electronic band-structure properties.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QuantumProperties {
    /// Band gap in eV.
    pub band_gap: Option<f64>,
    /// Electron mobility in cm²/(V·s).
    pub electron_mobility: Option<f64>,
    /// Spin coherence time in seconds.
    pub spin_coherence_time: Option<f64>,
    /// Superconducting critical temperature in Kelvin.
    pub superconducting_tc: Option<f64>,
    /// Topological invariant (e.g. Z₂ index).
    pub topological_index: Option<f64>,
}

/// Lattice parameters for a crystal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeParameters {
    /// Cell length a in Å.
    pub a: f64,
    /// Cell length b in Å.
    pub b: f64,
    /// Cell length c in Å.
    pub c: f64,
    /// Angle α in degrees.
    pub alpha: f64,
    /// Angle β in degrees.
    pub beta: f64,
    /// Angle γ in degrees.
    pub gamma: f64,
}

/// Crystallographic structure description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrystalStructure {
    /// One of seven crystal systems.
    pub system: CrystalSystem,
    /// Hermann-Mauguin space group symbol (e.g. "Fm-3m").
    pub space_group: String,
    /// Lattice parameters.
    pub lattice_params: LatticeParameters,
    /// Number of atoms in the conventional unit cell.
    pub atoms_per_unit_cell: u32,
    /// Point group symbol.
    pub point_group: String,
    /// Raw CIF data, if available.
    pub cif_data: Option<String>,
}

/// Properties specific to engineered metamaterials.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetamaterialProperties {
    /// Unit-cell geometry type.
    pub geometry: GeometryType,
    /// Unit cell size in metres.
    pub unit_cell_size: f64,
    /// Effective (homogenised) electric permittivity.
    pub effective_permittivity: Complex64,
    /// Effective (homogenised) magnetic permeability.
    pub effective_permeability: Complex64,
    /// Frequency range (Hz) where n < 0, as (f_low, f_high).
    pub negative_index_range: Option<(f64, f64)>,
    /// Operating bandwidth in Hz.
    pub bandwidth: Option<f64>,
}

/// Elemental / molecular composition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Composition {
    /// (element_symbol, atomic_fraction) pairs.
    pub elements: Vec<(String, f64)>,
}

/// A qualitative observation recorded by a researcher.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectiveObservation {
    /// When the observation was recorded.
    pub timestamp: DateTime<Utc>,
    /// Observer name or identifier.
    pub observer: String,
    /// Category of the observation.
    pub category: ObservationCategory,
    /// Subjective intensity rating [0, 10].
    pub intensity: f64,
    /// Free-text description.
    pub description: String,
    /// Environmental / contextual conditions.
    pub conditions: HashMap<String, String>,
    /// Observer confidence in the report [0, 1].
    pub confidence: f64,
}

// ---------------------------------------------------------------------------
// Material — the central type
// ---------------------------------------------------------------------------

/// A material in the AETHER knowledge base.
///
/// Combines quantitative physical data with qualitative observations and
/// arbitrary custom properties to form a single, serialisable record that
/// drives every downstream computation (compatibility, simulation, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Material {
    /// Unique identifier.
    pub id: Uuid,
    /// Human-readable name.
    pub name: String,
    /// Top-level category.
    pub category: MaterialCategory,
    /// Hierarchical classification tags (e.g. ["metal", "transition", "ferrous"]).
    pub classification: Vec<String>,

    // — Domain property groups —
    pub physical: PhysicalProperties,
    pub electromagnetic: ElectromagneticProperties,
    pub mechanical: MechanicalProperties,
    pub quantum: QuantumProperties,

    // — Optional structured extensions —
    pub crystal: Option<CrystalStructure>,
    pub metamaterial: Option<MetamaterialProperties>,
    pub composition: Option<Composition>,

    // — Qualitative data —
    pub observations: Vec<SubjectiveObservation>,

    // — Extensibility —
    pub custom_properties: HashMap<String, PropertyValue>,

    // — Provenance —
    /// Data source / reference.
    pub source: Option<String>,
    /// Overall confidence in the data [0, 1].
    pub confidence: Option<f64>,
    /// Free-text notes.
    pub notes: Option<String>,

    // — Timestamps —
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Material {
    /// Create a new material with sensible defaults.
    pub fn new(name: impl Into<String>, category: MaterialCategory) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            category,
            classification: Vec::new(),
            physical: PhysicalProperties::default(),
            electromagnetic: ElectromagneticProperties::default(),
            mechanical: MechanicalProperties::default(),
            quantum: QuantumProperties::default(),
            crystal: None,
            metamaterial: None,
            composition: None,
            observations: Vec::new(),
            custom_properties: HashMap::new(),
            source: None,
            confidence: None,
            notes: None,
            created_at: now,
            updated_at: now,
        }
    }

    // — Builder-style setters (consume and return self) —

    /// Set the density in kg/m³.
    pub fn with_density(mut self, density: f64) -> Self {
        self.physical.density = Some(density);
        self
    }

    /// Set the melting point in Kelvin.
    pub fn with_melting_point(mut self, t: f64) -> Self {
        self.physical.melting_point = Some(t);
        self
    }

    /// Set thermal conductivity in W/(m·K).
    pub fn with_thermal_conductivity(mut self, k: f64) -> Self {
        self.physical.thermal_conductivity = Some(k);
        self
    }

    /// Set coefficient of thermal expansion in 1/K.
    pub fn with_thermal_expansion(mut self, alpha: f64) -> Self {
        self.physical.thermal_expansion = Some(alpha);
        self
    }

    /// Set specific heat capacity in J/(kg·K).
    pub fn with_specific_heat(mut self, cp: f64) -> Self {
        self.physical.specific_heat = Some(cp);
        self
    }

    /// Set the dielectric constant.
    pub fn with_dielectric_constant(mut self, eps: f64) -> Self {
        self.electromagnetic.dielectric_constant = Some(eps);
        self
    }

    /// Set the magnetic permeability.
    pub fn with_magnetic_permeability(mut self, mu: f64) -> Self {
        self.electromagnetic.magnetic_permeability = Some(mu);
        self
    }

    /// Set the electrical conductivity in S/m.
    pub fn with_electrical_conductivity(mut self, sigma: f64) -> Self {
        self.electromagnetic.electrical_conductivity = Some(sigma);
        self
    }

    /// Set the refractive index.
    pub fn with_refractive_index(mut self, n: f64) -> Self {
        self.electromagnetic.refractive_index = Some(n);
        self
    }

    /// Set piezoelectric coefficients.
    pub fn with_piezoelectric_coeff(mut self, coeffs: Vec<f64>) -> Self {
        self.electromagnetic.piezoelectric_coeff = Some(coeffs);
        self
    }

    /// Set the band gap in eV.
    pub fn with_band_gap(mut self, eg: f64) -> Self {
        self.quantum.band_gap = Some(eg);
        self
    }

    /// Set electron mobility in cm²/(V·s).
    pub fn with_electron_mobility(mut self, mu: f64) -> Self {
        self.quantum.electron_mobility = Some(mu);
        self
    }

    /// Set the superconducting critical temperature in Kelvin.
    pub fn with_superconducting_tc(mut self, tc: f64) -> Self {
        self.quantum.superconducting_tc = Some(tc);
        self
    }

    /// Set Mohs hardness.
    pub fn with_hardness(mut self, h: f64) -> Self {
        self.mechanical.hardness_mohs = Some(h);
        self
    }

    /// Set Young's modulus in GPa.
    pub fn with_youngs_modulus(mut self, e: f64) -> Self {
        self.mechanical.youngs_modulus = Some(e);
        self
    }

    /// Set tensile strength in MPa.
    pub fn with_tensile_strength(mut self, ts: f64) -> Self {
        self.mechanical.tensile_strength = Some(ts);
        self
    }

    /// Set fracture toughness in MPa·√m.
    pub fn with_fracture_toughness(mut self, kic: f64) -> Self {
        self.mechanical.fracture_toughness = Some(kic);
        self
    }

    /// Set the crystal structure.
    pub fn with_crystal(mut self, crystal: CrystalStructure) -> Self {
        self.crystal = Some(crystal);
        self
    }

    /// Set metamaterial properties.
    pub fn with_metamaterial(mut self, meta: MetamaterialProperties) -> Self {
        self.metamaterial = Some(meta);
        self
    }

    /// Set elemental composition.
    pub fn with_composition(mut self, comp: Composition) -> Self {
        self.composition = Some(comp);
        self
    }

    /// Set the data source reference.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Set overall data confidence [0, 1].
    pub fn with_confidence(mut self, c: f64) -> Self {
        self.confidence = Some(c);
        self
    }

    /// Add a classification tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.classification.push(tag.into());
        self
    }

    /// Set free-text notes.
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }

    /// Insert a custom property.
    pub fn with_custom(mut self, key: impl Into<String>, value: PropertyValue) -> Self {
        self.custom_properties.insert(key.into(), value);
        self
    }

    /// Add a subjective observation.
    pub fn add_observation(&mut self, obs: SubjectiveObservation) {
        self.observations.push(obs);
        self.updated_at = Utc::now();
    }
}
