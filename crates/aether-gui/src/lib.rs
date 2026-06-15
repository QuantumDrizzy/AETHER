//! AETHER LAB — native GUI (egui).
//!
//! A clean, emoji-free lab dashboard: browse the materials library, compute
//! compatibility with the real Rust engine, and launch Python research sims.

use std::collections::HashMap;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;

use eframe::egui;
use egui_plot::{Line, Plot};

use aether_core::compatibility::{CompatibilityEngine, CompatibilityResult, Recommendation};
use aether_core::material::Material;
use aether_db::AetherDb;

const DB_PATH: &str = "data/aether.db";
const LIBRARY_PATH: &str = "data/materials/library.json";

// ── palette: clean slate dark, one restrained accent ─────────────────────────
const BG: egui::Color32 = egui::Color32::from_rgb(20, 22, 28);
const PANEL: egui::Color32 = egui::Color32::from_rgb(28, 31, 39);
const TEXT: egui::Color32 = egui::Color32::from_rgb(222, 226, 233);
const MUTED: egui::Color32 = egui::Color32::from_rgb(140, 146, 158);
const ACCENT: egui::Color32 = egui::Color32::from_rgb(90, 196, 184); // muted teal
const GOOD: egui::Color32 = egui::Color32::from_rgb(122, 200, 140);
const BAD: egui::Color32 = egui::Color32::from_rgb(224, 146, 120);
const VIOLET: egui::Color32 = egui::Color32::from_rgb(170, 150, 240);

fn style(ctx: &egui::Context) {
    let mut v = egui::Visuals::dark();
    v.override_text_color = Some(TEXT);
    v.panel_fill = BG;
    v.window_fill = PANEL;
    v.extreme_bg_color = egui::Color32::from_rgb(16, 18, 23);
    v.selection.bg_fill = ACCENT.linear_multiply(0.35);
    let r = egui::Rounding::same(6.0);
    for w in [&mut v.widgets.inactive, &mut v.widgets.hovered, &mut v.widgets.active] {
        w.rounding = r;
    }
    v.widgets.inactive.weak_bg_fill = PANEL;
    v.widgets.hovered.weak_bg_fill = egui::Color32::from_rgb(40, 44, 54);
    v.widgets.active.weak_bg_fill = egui::Color32::from_rgb(48, 53, 64);
    ctx.set_visuals(v);
    ctx.style_mut(|s| {
        s.spacing.item_spacing = egui::vec2(8.0, 8.0);
        s.spacing.button_padding = egui::vec2(12.0, 7.0);
    });
}

#[derive(PartialEq)]
enum Tab {
    Dashboard,
    Materials,
    Compatibility,
    Bands,
    Experiments,
}

pub struct AetherApp {
    db: Option<AetherDb>,
    materials: Vec<Material>,
    tab: Tab,
    status: Option<String>,
    // compatibility tab
    sel_a: usize,
    sel_b: usize,
    compat: Option<CompatibilityResult>,
    // band-structure tab
    t_ev: f64,
    // experiments tab (background python)
    sim_output: Arc<Mutex<Option<String>>>,
    is_sim_running: Arc<Mutex<bool>>,
}

impl Default for AetherApp {
    fn default() -> Self {
        Self::new()
    }
}

impl AetherApp {
    pub fn new() -> Self {
        let mut app = Self {
            db: None,
            materials: Vec::new(),
            tab: Tab::Dashboard,
            status: None,
            sel_a: 0,
            sel_b: 1,
            compat: None,
            t_ev: 2.7,
            sim_output: Arc::new(Mutex::new(None)),
            is_sim_running: Arc::new(Mutex::new(false)),
        };
        match AetherDb::init(DB_PATH) {
            Ok(db) => {
                app.db = Some(db);
                app.reload_materials();
            }
            Err(e) => app.status = Some(format!("DB error: {e}")),
        }
        app
    }

    fn reload_materials(&mut self) {
        if let Some(db) = &self.db {
            match db.list_materials(None) {
                Ok(mats) => {
                    self.materials = mats;
                    self.sel_a = self.sel_a.min(self.materials.len().saturating_sub(1));
                    self.sel_b = self.sel_b.min(self.materials.len().saturating_sub(1));
                }
                Err(e) => self.status = Some(format!("load error: {e}")),
            }
        }
    }

    fn seed_library(&mut self) {
        let Some(db) = &self.db else { return };
        let result = std::fs::read_to_string(LIBRARY_PATH)
            .map_err(|e| e.to_string())
            .and_then(|json| aether_db::library::parse_library(&json).map_err(|e| e.to_string()));
        match result {
            Ok(mats) => {
                let existing: std::collections::HashSet<String> =
                    self.materials.iter().map(|m| m.name.to_lowercase()).collect();
                let mut added = 0;
                for m in mats {
                    if existing.contains(&m.name.to_lowercase()) {
                        continue;
                    }
                    if db.insert_material(&m).is_ok() {
                        added += 1;
                    }
                }
                self.status = Some(format!("Seeded {added} materials (skipped existing)."));
            }
            Err(e) => self.status = Some(format!("seed failed: {e}")),
        }
        self.reload_materials();
    }
}

impl eframe::App for AetherApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        style(ctx);

        egui::SidePanel::left("sidebar")
            .resizable(false)
            .exact_width(190.0)
            .frame(egui::Frame::default().fill(PANEL).inner_margin(14.0))
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.label(egui::RichText::new("AETHER LAB").size(20.0).color(ACCENT).strong());
                ui.label(egui::RichText::new("materials research").size(12.0).color(MUTED).italics());
                ui.add_space(18.0);
                let tab = |ui: &mut egui::Ui, cur: &mut Tab, t: Tab, label: &str| {
                    let selected = *cur == t;
                    if ui.add_sized([160.0, 30.0], egui::SelectableLabel::new(selected, label)).clicked() {
                        *cur = t;
                    }
                };
                tab(ui, &mut self.tab, Tab::Dashboard, "Dashboard");
                tab(ui, &mut self.tab, Tab::Materials, "Materials");
                tab(ui, &mut self.tab, Tab::Compatibility, "Compatibility");
                tab(ui, &mut self.tab, Tab::Bands, "Band structure");
                tab(ui, &mut self.tab, Tab::Experiments, "Experiments");

                ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                    ui.add_space(6.0);
                    ui.label(egui::RichText::new("iNFAMØUS OS").size(11.0).color(MUTED));
                });
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::default().fill(BG).inner_margin(20.0))
            .show(ctx, |ui| {
                match self.tab {
                    Tab::Dashboard => self.ui_dashboard(ui),
                    Tab::Materials => self.ui_materials(ui),
                    Tab::Compatibility => self.ui_compat(ui),
                    Tab::Bands => self.ui_bands(ui),
                    Tab::Experiments => self.ui_experiments(ui),
                }
                if let Some(s) = &self.status {
                    ui.add_space(12.0);
                    ui.separator();
                    ui.label(egui::RichText::new(s).size(12.0).color(MUTED));
                }
            });
    }
}

impl AetherApp {
    fn ui_dashboard(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Dashboard").size(24.0).color(TEXT).strong());
        ui.add_space(8.0);
        ui.label(
            egui::RichText::new("Advanced-materials research engine — electronic structure, photonics, and compatibility, on a validated core.")
                .color(MUTED),
        );
        ui.add_space(16.0);
        ui.label(egui::RichText::new(format!("Materials in database: {}", self.materials.len())).size(16.0));
        ui.add_space(14.0);
        ui.horizontal(|ui| {
            if ui.button("Seed materials library").clicked() {
                self.seed_library();
            }
            if ui.button("Refresh").clicked() {
                self.reload_materials();
            }
        });
    }

    fn ui_materials(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Materials").size(24.0).color(TEXT).strong());
        ui.add_space(8.0);
        if self.materials.is_empty() {
            ui.label(egui::RichText::new("No materials yet — seed the library from the Dashboard.").color(MUTED));
            return;
        }
        egui::ScrollArea::vertical().show(ui, |ui| {
            for m in &self.materials {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new(&m.name).size(16.0).color(ACCENT).strong());
                        ui.label(egui::RichText::new(format!("· {}", m.category)).color(MUTED));
                    });
                    if !m.classification.is_empty() {
                        ui.label(egui::RichText::new(m.classification.join(", ")).size(12.0).color(MUTED).italics());
                    }
                    ui.horizontal_wrapped(|ui| {
                        if let Some(c) = &m.crystal {
                            ui.label(format!("{:?}  a={:.3} Å", c.system, c.lattice_params.a));
                            ui.separator();
                        }
                        if let Some(d) = m.physical.density {
                            ui.label(format!("ρ={d:.0} kg/m³"));
                            ui.separator();
                        }
                        if let Some(g) = m.quantum.band_gap {
                            ui.label(format!("Eg={g:.2} eV"));
                        }
                    });
                    if let Some(n) = &m.notes {
                        ui.label(egui::RichText::new(n).size(11.0).color(MUTED));
                    }
                });
            }
        });
    }

    fn ui_compat(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Compatibility").size(24.0).color(TEXT).strong());
        ui.add_space(8.0);
        if self.materials.len() < 2 {
            ui.label(egui::RichText::new("Seed at least two materials to compare.").color(MUTED));
            return;
        }
        let names: Vec<String> = self.materials.iter().map(|m| m.name.clone()).collect();
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_source("mat_a")
                .selected_text(names[self.sel_a].as_str())
                .show_ui(ui, |ui| {
                    for (i, n) in names.iter().enumerate() {
                        ui.selectable_value(&mut self.sel_a, i, n.as_str());
                    }
                });
            ui.label("vs");
            egui::ComboBox::from_id_source("mat_b")
                .selected_text(names[self.sel_b].as_str())
                .show_ui(ui, |ui| {
                    for (i, n) in names.iter().enumerate() {
                        ui.selectable_value(&mut self.sel_b, i, n.as_str());
                    }
                });
            if ui.button("Compute").clicked() {
                let r = CompatibilityEngine::new().compute(
                    &self.materials[self.sel_a],
                    &self.materials[self.sel_b],
                    &HashMap::new(),
                );
                self.compat = Some(r);
            }
        });

        if let Some(r) = &self.compat {
            ui.add_space(14.0);
            let (col, rec) = match r.recommendation {
                Recommendation::HighlyCompatible | Recommendation::Compatible => (GOOD, &r.recommendation),
                Recommendation::Neutral => (MUTED, &r.recommendation),
                _ => (BAD, &r.recommendation),
            };
            ui.label(egui::RichText::new(format!("Overall {:.0}% — {rec}", r.overall_score * 100.0)).size(18.0).color(col).strong());
            ui.add_space(8.0);
            for d in &r.dimensions {
                ui.horizontal(|ui| {
                    ui.add_sized([130.0, 18.0], egui::Label::new(egui::RichText::new(&d.name).color(MUTED)));
                    ui.add(egui::ProgressBar::new(d.score as f32).desired_width(260.0).fill(ACCENT).text(format!("{:.2}", d.score)));
                });
            }
            for s in &r.synergies {
                ui.label(egui::RichText::new(format!("+ {}", s.description)).size(12.0).color(GOOD));
            }
            for c in &r.conflicts {
                ui.label(egui::RichText::new(format!("− {}", c.description)).size(12.0).color(BAD));
            }
        }
    }

    fn ui_bands(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Graphene band structure").size(24.0).color(TEXT).strong());
        ui.add_space(6.0);
        ui.label(
            egui::RichText::new("Nearest-neighbor tight-binding, closed form  E = ±t·|f(k)|  along Γ–K–M–Γ. Live.")
                .color(MUTED),
        );
        ui.add_space(10.0);
        ui.add(egui::Slider::new(&mut self.t_ev, 0.5..=4.0).text("hopping t (eV)"));
        ui.add_space(8.0);

        let (dist, lo, hi) = aether_core::physics::graphene_band_path(self.t_ev, 1.42, 200);
        let valence: Vec<[f64; 2]> = dist.iter().zip(&lo).map(|(&d, &e)| [d, e]).collect();
        let conduction: Vec<[f64; 2]> = dist.iter().zip(&hi).map(|(&d, &e)| [d, e]).collect();
        Plot::new("graphene_bands").height(360.0).show(ui, |p| {
            p.line(Line::new(valence).color(ACCENT).name("valence"));
            p.line(Line::new(conduction).color(VIOLET).name("conduction"));
        });
        ui.label(
            egui::RichText::new(format!(
                "bandwidth 6t = {:.2} eV · the bands touch at the Dirac point K (gap → 0)",
                6.0 * self.t_ev
            ))
            .size(12.0)
            .color(MUTED),
        );
    }

    fn ui_experiments(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Experiments").size(24.0).color(TEXT).strong());
        ui.add_space(8.0);
        ui.label(egui::RichText::new("Run Python research simulations from the lab.").color(MUTED));
        ui.add_space(12.0);

        let running = *self.is_sim_running.lock().unwrap();
        if running {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Simulation running in the background…");
            });
        } else if ui.button("Run quantum annealing (SQA)").clicked() {
            *self.is_sim_running.lock().unwrap() = true;
            *self.sim_output.lock().unwrap() = Some("Launching…\n".to_string());
            let sim_out = self.sim_output.clone();
            let is_run = self.is_sim_running.clone();
            thread::spawn(move || {
                let output = Command::new("python")
                    .args(["-m", "research.quantum_annealing.annealer"])
                    .output();
                let text = match output {
                    Ok(o) => {
                        format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))
                    }
                    Err(e) => format!("Failed to execute python: {e}"),
                };
                *sim_out.lock().unwrap() = Some(text);
                *is_run.lock().unwrap() = false;
            });
        }

        ui.add_space(14.0);
        egui::ScrollArea::vertical().max_height(420.0).show(ui, |ui| {
            if let Some(out) = self.sim_output.lock().unwrap().as_ref() {
                ui.add(egui::TextEdit::multiline(&mut out.as_str()).desired_width(f32::INFINITY).font(egui::TextStyle::Monospace));
            } else {
                ui.label(egui::RichText::new("No simulations run yet.").color(MUTED).weak());
            }
        });
    }
}
