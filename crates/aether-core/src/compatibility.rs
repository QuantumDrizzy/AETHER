//! Material compatibility engine.
//!
//! Computes pairwise compatibility scores across multiple physical
//! dimensions and produces an aggregated recommendation.

use std::collections::HashMap;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::material::Material;

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// A single scored dimension within a compatibility analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityDimension {
    /// Human-readable dimension name (e.g. "thermal").
    pub name: String,
    /// Score in [0, 1] where 1 = perfectly compatible.
    pub score: f64,
    /// Relative weight used during aggregation.
    pub weight: f64,
    /// Method / algorithm that produced this score.
    pub method: String,
    /// Free-text details or rationale.
    pub details: String,
}

/// A beneficial synergy between two materials.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Synergy {
    /// What the synergy is.
    pub description: String,
    /// Magnitude [0, 1].
    pub magnitude: f64,
    /// Supporting evidence.
    pub evidence: String,
}

/// A detrimental conflict between two materials.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conflict {
    /// What the conflict is.
    pub description: String,
    /// Severity [0, 1].
    pub severity: f64,
    /// Supporting evidence.
    pub evidence: String,
}

/// Overall compatibility recommendation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Recommendation {
    HighlyCompatible,
    Compatible,
    Neutral,
    Incompatible,
    HighlyIncompatible,
}

impl std::fmt::Display for Recommendation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HighlyCompatible => write!(f, "Highly Compatible"),
            Self::Compatible => write!(f, "Compatible"),
            Self::Neutral => write!(f, "Neutral"),
            Self::Incompatible => write!(f, "Incompatible"),
            Self::HighlyIncompatible => write!(f, "Highly Incompatible"),
        }
    }
}

/// The full compatibility report for a material pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityResult {
    /// First material.
    pub material_a: Uuid,
    /// Second material.
    pub material_b: Uuid,
    /// Weighted overall compatibility score [0, 1].
    pub overall_score: f64,
    /// Per-dimension scores.
    pub dimensions: Vec<CompatibilityDimension>,
    /// Identified synergies.
    pub synergies: Vec<Synergy>,
    /// Identified conflicts.
    pub conflicts: Vec<Conflict>,
    /// Summary recommendation.
    pub recommendation: Recommendation,
    /// When this result was computed.
    pub computed_at: chrono::DateTime<chrono::Utc>,
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Computes multi-dimensional compatibility between materials.
#[derive(Debug, Clone)]
pub struct CompatibilityEngine;

impl CompatibilityEngine {
    /// Create a new engine instance.
    pub fn new() -> Self {
        Self
    }

    /// Compute full compatibility between two materials.
    pub fn compute(
        &self,
        material_a: &Material,
        material_b: &Material,
        weights: &HashMap<String, f64>,
    ) -> CompatibilityResult {
        let mut dimensions = Vec::new();

        // Compute each dimension.
        let dim_fns: Vec<(&str, fn(&CompatibilityEngine, &Material, &Material) -> CompatibilityDimension)> = vec![
            ("crystallographic", Self::crystallographic),
            ("thermal", Self::thermal),
            ("electromagnetic", Self::electromagnetic),
            ("resonant", Self::resonant),
            ("piezoelectric", Self::piezoelectric),
            ("mechanical", Self::mechanical),
        ];

        for (name, func) in &dim_fns {
            let mut dim = func(self, material_a, material_b);
            if let Some(&w) = weights.get(*name) {
                dim.weight = w;
            }
            dimensions.push(dim);
        }

        // Weighted average.
        let total_weight: f64 = dimensions.iter().map(|d| d.weight).sum();
        let overall_score = if total_weight > 0.0 {
            dimensions.iter().map(|d| d.score * d.weight).sum::<f64>() / total_weight
        } else {
            0.5
        };

        // Detect synergies and conflicts.
        let mut synergies = Vec::new();
        let mut conflicts = Vec::new();

        for dim in &dimensions {
            if dim.score > 0.85 {
                synergies.push(Synergy {
                    description: format!("Strong {} compatibility", dim.name),
                    magnitude: dim.score,
                    evidence: dim.details.clone(),
                });
            } else if dim.score < 0.3 {
                conflicts.push(Conflict {
                    description: format!("Poor {} compatibility", dim.name),
                    severity: 1.0 - dim.score,
                    evidence: dim.details.clone(),
                });
            }
        }

        let recommendation = match overall_score {
            s if s >= 0.8 => Recommendation::HighlyCompatible,
            s if s >= 0.6 => Recommendation::Compatible,
            s if s >= 0.4 => Recommendation::Neutral,
            s if s >= 0.2 => Recommendation::Incompatible,
            _ => Recommendation::HighlyIncompatible,
        };

        CompatibilityResult {
            material_a: material_a.id,
            material_b: material_b.id,
            overall_score,
            dimensions,
            synergies,
            conflicts,
            recommendation,
            computed_at: Utc::now(),
        }
    }

    // ------------------------------------------------------------------
    // Dimension calculators
    // ------------------------------------------------------------------

    /// Crystallographic compatibility based on lattice parameter mismatch.
    pub fn crystallographic(&self, a: &Material, b: &Material) -> CompatibilityDimension {
        let score = match (&a.crystal, &b.crystal) {
            (Some(ca), Some(cb)) => {
                if ca.system == cb.system {
                    // Compute lattice mismatch percentage.
                    let mismatch_a = ((ca.lattice_params.a - cb.lattice_params.a) / ca.lattice_params.a).abs();
                    let mismatch_b = ((ca.lattice_params.b - cb.lattice_params.b) / ca.lattice_params.b).abs();
                    let mismatch_c = ((ca.lattice_params.c - cb.lattice_params.c) / ca.lattice_params.c).abs();
                    let avg_mismatch = (mismatch_a + mismatch_b + mismatch_c) / 3.0;
                    (1.0 - avg_mismatch * 10.0).max(0.0).min(1.0)
                } else {
                    0.3 // Different crystal systems are moderately incompatible.
                }
            }
            _ => 0.5, // Insufficient data.
        };

        CompatibilityDimension {
            name: "crystallographic".into(),
            score,
            weight: 1.0,
            method: "lattice_mismatch".into(),
            details: format!(
                "Crystallographic compatibility: {:.2} (systems: {:?} vs {:?})",
                score,
                a.crystal.as_ref().map(|c| &c.system),
                b.crystal.as_ref().map(|c| &c.system),
            ),
        }
    }

    /// Thermal compatibility based on expansion coefficient and melting point ratios.
    pub fn thermal(&self, a: &Material, b: &Material) -> CompatibilityDimension {
        let mut score = 0.5;
        let mut details = String::from("Thermal compatibility: ");

        match (a.physical.thermal_expansion, b.physical.thermal_expansion) {
            (Some(ea), Some(eb)) => {
                let ratio = if ea > eb { eb / ea } else { ea / eb };
                score = ratio.max(0.0).min(1.0);
                details.push_str(&format!("CTE ratio {:.3}", ratio));
            }
            _ => {
                details.push_str("insufficient CTE data");
            }
        }

        // Adjust for melting point proximity.
        if let (Some(ma), Some(mb)) = (a.physical.melting_point, b.physical.melting_point) {
            let mp_ratio = if ma > mb { mb / ma } else { ma / mb };
            score = (score + mp_ratio) / 2.0;
            details.push_str(&format!(", MP ratio {:.3}", mp_ratio));
        }

        CompatibilityDimension {
            name: "thermal".into(),
            score,
            weight: 1.2,
            method: "cte_and_melting_point".into(),
            details,
        }
    }

    /// Electromagnetic compatibility based on dielectric and conductivity matching.
    pub fn electromagnetic(&self, a: &Material, b: &Material) -> CompatibilityDimension {
        let mut score = 0.5;
        let mut details = String::from("EM compatibility: ");

        match (
            a.electromagnetic.dielectric_constant,
            b.electromagnetic.dielectric_constant,
        ) {
            (Some(da), Some(db)) => {
                let ratio = if da > db { db / da } else { da / db };
                score = ratio.max(0.0).min(1.0);
                details.push_str(&format!("ε ratio {:.3}", ratio));
            }
            _ => {
                details.push_str("insufficient dielectric data");
            }
        }

        if let (
            Some(sa),
            Some(sb),
        ) = (
            a.electromagnetic.electrical_conductivity,
            b.electromagnetic.electrical_conductivity,
        ) {
            if sa > 0.0 && sb > 0.0 {
                let log_ratio = (sa.ln() - sb.ln()).abs();
                let cond_score = (-log_ratio / 10.0_f64).exp();
                score = (score + cond_score) / 2.0;
                details.push_str(&format!(", σ log-ratio {:.3}", log_ratio));
            }
        }

        CompatibilityDimension {
            name: "electromagnetic".into(),
            score,
            weight: 1.5,
            method: "dielectric_conductivity".into(),
            details,
        }
    }

    /// Resonant compatibility based on overlapping resonant frequencies.
    pub fn resonant(&self, a: &Material, b: &Material) -> CompatibilityDimension {
        let fa = &a.electromagnetic.resonant_frequencies;
        let fb = &b.electromagnetic.resonant_frequencies;

        let score = if fa.is_empty() || fb.is_empty() {
            0.5
        } else {
            // Find the best pairwise frequency ratio.
            let mut best = 0.0_f64;
            for &freq_a in fa {
                for &freq_b in fb {
                    if freq_a > 0.0 && freq_b > 0.0 {
                        let ratio = if freq_a > freq_b {
                            freq_b / freq_a
                        } else {
                            freq_a / freq_b
                        };
                        best = best.max(ratio);
                    }
                }
            }
            best.min(1.0)
        };

        CompatibilityDimension {
            name: "resonant".into(),
            score,
            weight: 1.3,
            method: "frequency_overlap".into(),
            details: format!(
                "Resonant compatibility: {:.2} ({} vs {} frequencies)",
                score,
                fa.len(),
                fb.len()
            ),
        }
    }

    /// Piezoelectric compatibility based on coefficient similarity.
    pub fn piezoelectric(&self, a: &Material, b: &Material) -> CompatibilityDimension {
        let score = match (
            &a.electromagnetic.piezoelectric_coeff,
            &b.electromagnetic.piezoelectric_coeff,
        ) {
            (Some(pa), Some(pb)) => {
                let len = pa.len().min(pb.len());
                if len == 0 {
                    0.5
                } else {
                    let mut sum = 0.0;
                    for i in 0..len {
                        let max_val = pa[i].abs().max(pb[i].abs());
                        if max_val > 0.0 {
                            sum += 1.0 - ((pa[i] - pb[i]).abs() / max_val).min(1.0);
                        } else {
                            sum += 1.0;
                        }
                    }
                    sum / len as f64
                }
            }
            _ => 0.5,
        };

        CompatibilityDimension {
            name: "piezoelectric".into(),
            score,
            weight: 1.4,
            method: "coefficient_similarity".into(),
            details: format!("Piezoelectric compatibility: {:.2}", score),
        }
    }

    /// Mechanical compatibility based on modulus and hardness similarity.
    pub fn mechanical(&self, a: &Material, b: &Material) -> CompatibilityDimension {
        let mut scores = Vec::new();

        if let (Some(ya), Some(yb)) = (a.mechanical.youngs_modulus, b.mechanical.youngs_modulus) {
            let ratio = if ya > yb { yb / ya } else { ya / yb };
            scores.push(ratio);
        }

        if let (Some(ha), Some(hb)) = (a.mechanical.hardness_mohs, b.mechanical.hardness_mohs) {
            let diff = (ha - hb).abs() / 10.0;
            scores.push(1.0 - diff);
        }

        if let (Some(ta), Some(tb)) = (a.mechanical.tensile_strength, b.mechanical.tensile_strength)
        {
            let ratio = if ta > tb { tb / ta } else { ta / tb };
            scores.push(ratio);
        }

        let score = if scores.is_empty() {
            0.5
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        };

        CompatibilityDimension {
            name: "mechanical".into(),
            score: score.max(0.0).min(1.0),
            weight: 0.8,
            method: "modulus_hardness".into(),
            details: format!("Mechanical compatibility: {:.2}", score),
        }
    }
}

impl Default for CompatibilityEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::{
        CrystalStructure, CrystalSystem, LatticeParameters, Material, MaterialCategory,
    };
    use std::collections::HashMap;

    fn cubic(a: f64) -> CrystalStructure {
        CrystalStructure {
            system: CrystalSystem::Cubic,
            space_group: "Fm-3m".into(),
            lattice_params: LatticeParameters {
                a,
                b: a,
                c: a,
                alpha: 90.0,
                beta: 90.0,
                gamma: 90.0,
            },
            atoms_per_unit_cell: 4,
            point_group: "m-3m".into(),
            cif_data: None,
        }
    }

    fn dim<'a>(r: &'a CompatibilityResult, name: &str) -> &'a CompatibilityDimension {
        r.dimensions.iter().find(|d| d.name == name).expect("dimension present")
    }

    #[test]
    fn identical_materials_are_compatible() {
        let a = Material::new("X", MaterialCategory::Crystal)
            .with_crystal(cubic(4.0))
            .with_thermal_expansion(1e-5)
            .with_melting_point(1000.0)
            .with_dielectric_constant(5.0);
        let b = a.clone();
        let r = CompatibilityEngine::new().compute(&a, &b, &HashMap::new());
        assert!(dim(&r, "crystallographic").score > 0.99, "identical lattice should score ~1");
        assert!((0.0..=1.0).contains(&r.overall_score));
        assert!(matches!(
            r.recommendation,
            Recommendation::Compatible | Recommendation::HighlyCompatible
        ));
    }

    #[test]
    fn missing_data_is_neutral() {
        let a = Material::new("A", MaterialCategory::Crystal);
        let b = Material::new("B", MaterialCategory::Crystal);
        let r = CompatibilityEngine::new().compute(&a, &b, &HashMap::new());
        for d in &r.dimensions {
            assert!((d.score - 0.5).abs() < 1e-9, "{} should fall back to 0.5", d.name);
        }
        assert!((r.overall_score - 0.5).abs() < 1e-9);
        assert_eq!(r.recommendation, Recommendation::Neutral);
    }

    #[test]
    fn all_dimension_scores_are_in_unit_interval() {
        let a = Material::new("A", MaterialCategory::Crystal)
            .with_crystal(cubic(4.0))
            .with_thermal_expansion(1e-5);
        let b = Material::new("B", MaterialCategory::Crystal)
            .with_crystal(cubic(5.5))
            .with_thermal_expansion(3e-5);
        let r = CompatibilityEngine::new().compute(&a, &b, &HashMap::new());
        assert!((0.0..=1.0).contains(&r.overall_score));
        for d in &r.dimensions {
            assert!((0.0..=1.0).contains(&d.score), "{} out of range: {}", d.name, d.score);
        }
    }

    #[test]
    fn closer_lattice_scores_higher() {
        let eng = CompatibilityEngine::new();
        let base = Material::new("base", MaterialCategory::Crystal).with_crystal(cubic(4.0));
        let near = Material::new("near", MaterialCategory::Crystal).with_crystal(cubic(4.1));
        let far = Material::new("far", MaterialCategory::Crystal).with_crystal(cubic(6.0));
        let s_near = dim(&eng.compute(&base, &near, &HashMap::new()), "crystallographic").score;
        let s_far = dim(&eng.compute(&base, &far, &HashMap::new()), "crystallographic").score;
        assert!(s_near > s_far, "closer lattice ({s_near}) should beat far ({s_far})");
    }
}
