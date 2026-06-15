//! Curated materials library: load real, sourced materials (JSON) into the
//! canonical `Material` schema. This is the single source of curated data; the
//! CLI `material seed` command loads it into the database.

use serde::Deserialize;

use aether_core::material::{
    CrystalStructure, CrystalSystem, LatticeParameters, Material, MaterialCategory,
};

use crate::DbResult;

/// A curated input record — only the fields we actually author, mapped onto the
/// full `Material` via its builder API. Keeps the JSON terse and editable.
#[derive(Debug, Deserialize)]
pub struct MaterialSpec {
    pub name: String,
    pub category: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub crystal_system: Option<String>,
    #[serde(default)]
    pub space_group: Option<String>,
    /// [a, b, c, alpha, beta, gamma] — Å and degrees.
    #[serde(default)]
    pub lattice: Option<[f64; 6]>,
    #[serde(default)]
    pub density: Option<f64>,
    #[serde(default)]
    pub melting_point: Option<f64>,
    #[serde(default)]
    pub thermal_expansion: Option<f64>,
    #[serde(default)]
    pub dielectric_constant: Option<f64>,
    #[serde(default)]
    pub electrical_conductivity: Option<f64>,
    #[serde(default)]
    pub youngs_modulus: Option<f64>,
    #[serde(default)]
    pub hardness_mohs: Option<f64>,
    #[serde(default)]
    pub band_gap: Option<f64>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub notes: Option<String>,
    /// Properties are DFT-predicted / not all measured (e.g. carbyne).
    #[serde(default)]
    pub predicted: bool,
}

fn category_from(s: &str) -> MaterialCategory {
    match s.to_lowercase().as_str() {
        "crystal" => MaterialCategory::Crystal,
        "metamaterial" => MaterialCategory::Metamaterial,
        "nanomaterial" => MaterialCategory::Nanomaterial,
        "alloy" => MaterialCategory::Alloy,
        "polymer" => MaterialCategory::Polymer,
        "composite" => MaterialCategory::Composite,
        "biological" => MaterialCategory::Biological,
        "theoretical" => MaterialCategory::Theoretical,
        other => MaterialCategory::Custom(other.to_string()),
    }
}

fn crystal_system_from(s: &str) -> Option<CrystalSystem> {
    Some(match s.to_lowercase().as_str() {
        "cubic" => CrystalSystem::Cubic,
        "tetragonal" => CrystalSystem::Tetragonal,
        "orthorhombic" => CrystalSystem::Orthorhombic,
        "hexagonal" => CrystalSystem::Hexagonal,
        "trigonal" => CrystalSystem::Trigonal,
        "monoclinic" => CrystalSystem::Monoclinic,
        "triclinic" => CrystalSystem::Triclinic,
        _ => return None,
    })
}

impl MaterialSpec {
    /// Build a full `Material` from this curated spec.
    pub fn into_material(self) -> Material {
        let mut m = Material::new(&self.name, category_from(&self.category));
        m.classification = self.tags;

        if let Some(v) = self.density {
            m = m.with_density(v);
        }
        if let Some(v) = self.melting_point {
            m = m.with_melting_point(v);
        }
        if let Some(v) = self.thermal_expansion {
            m = m.with_thermal_expansion(v);
        }
        if let Some(v) = self.dielectric_constant {
            m = m.with_dielectric_constant(v);
        }
        if let Some(v) = self.electrical_conductivity {
            m = m.with_electrical_conductivity(v);
        }
        if let Some(v) = self.youngs_modulus {
            m = m.with_youngs_modulus(v);
        }
        if let Some(v) = self.hardness_mohs {
            m = m.with_hardness(v);
        }
        if let Some(v) = self.band_gap {
            m = m.with_band_gap(v);
        }

        let system = self.crystal_system.as_deref().and_then(crystal_system_from);
        if let (Some(system), Some(l)) = (system, self.lattice) {
            m = m.with_crystal(CrystalStructure {
                system,
                space_group: self.space_group.unwrap_or_default(),
                lattice_params: LatticeParameters {
                    a: l[0],
                    b: l[1],
                    c: l[2],
                    alpha: l[3],
                    beta: l[4],
                    gamma: l[5],
                },
                atoms_per_unit_cell: 0,
                point_group: String::new(),
                cif_data: None,
            });
        }

        if let Some(src) = self.source {
            m = m.with_source(src);
        }
        let mut notes = self.notes.unwrap_or_default();
        if self.predicted {
            if !notes.is_empty() {
                notes.push(' ');
            }
            notes.push_str("[KNOWN_LIMIT] properties are DFT-predicted, not all measured.");
        }
        if !notes.is_empty() {
            m = m.with_notes(notes);
        }
        m
    }
}

/// Parse a JSON array of [`MaterialSpec`] into canonical [`Material`]s.
pub fn parse_library(json: &str) -> DbResult<Vec<Material>> {
    let specs: Vec<MaterialSpec> = serde_json::from_str(json)?;
    Ok(specs.into_iter().map(MaterialSpec::into_material).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    // The real curated library, embedded at compile time.
    const LIB: &str = include_str!("../../../data/materials/library.json");

    #[test]
    fn real_library_parses_with_key_materials() {
        let mats = parse_library(LIB).unwrap();
        assert!(mats.len() >= 10, "expected a real library, got {}", mats.len());
        let has = |s: &str| mats.iter().any(|m| m.name.to_lowercase().contains(s));
        assert!(has("graphene"));
        assert!(has("bismuth"));
        assert!(has("carbyne"));
        assert!(has("diamond"));
    }

    #[test]
    fn graphene_lattice_is_correct() {
        let mats = parse_library(LIB).unwrap();
        let g = mats.iter().find(|m| m.name.eq_ignore_ascii_case("Graphene")).unwrap();
        let c = g.crystal.as_ref().expect("graphene has a crystal");
        assert_eq!(c.system, CrystalSystem::Hexagonal);
        assert!((c.lattice_params.a - 2.46).abs() < 0.05);
    }

    #[test]
    fn carbyne_is_flagged_predicted() {
        let mats = parse_library(LIB).unwrap();
        let c = mats.iter().find(|m| m.name.eq_ignore_ascii_case("Carbyne")).unwrap();
        assert!(c.notes.as_deref().unwrap_or("").contains("[KNOWN_LIMIT]"));
    }

    #[test]
    fn spec_maps_category_and_fields() {
        let json = r#"[{"name":"T","category":"crystal","crystal_system":"cubic",
            "lattice":[4.0,4.0,4.0,90,90,90],"band_gap":1.1,"density":2000}]"#;
        let m = &parse_library(json).unwrap()[0];
        assert_eq!(m.category, MaterialCategory::Crystal);
        assert_eq!(m.crystal.as_ref().unwrap().system, CrystalSystem::Cubic);
        assert_eq!(m.quantum.band_gap, Some(1.1));
        assert_eq!(m.physical.density, Some(2000.0));
    }
}
