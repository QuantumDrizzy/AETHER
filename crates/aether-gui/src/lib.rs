use eframe::egui;
use aether_db::AetherDb;
use aether_core::material::Material;
use std::sync::{Arc, Mutex};
use std::thread;
use std::process::Command;

pub struct AetherApp {
    db: Arc<Mutex<Option<AetherDb>>>,
    materials: Vec<Material>,
    selected_tab: Tab,
    error_msg: Option<String>,
    sim_output: Arc<Mutex<Option<String>>>,
    is_sim_running: Arc<Mutex<bool>>,
}

#[derive(PartialEq)]
enum Tab {
    Dashboard,
    Materials,
    Compatibility,
    Experiments,
}

impl AetherApp {
    pub fn new() -> Self {
        let db_result = AetherDb::init("data/aether.db");
        let mut db_opt = None;
        let mut error_msg = None;
        let mut materials = Vec::new();

        match db_result {
            Ok(db) => {
                match db.list_materials(None) {
                    Ok(mats) => materials = mats,
                    Err(e) => error_msg = Some(format!("Failed to load materials: {}", e)),
                }
                db_opt = Some(db);
            }
            Err(e) => {
                error_msg = Some(format!("Failed to init DB: {}", e));
            }
        }

        Self {
            db: Arc::new(Mutex::new(db_opt)),
            materials,
            selected_tab: Tab::Dashboard,
            error_msg,
            sim_output: Arc::new(Mutex::new(None)),
            is_sim_running: Arc::new(Mutex::new(false)),
        }
    }

    fn draw_sidebar(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("sidebar").show(ctx, |ui| {
            ui.heading("AETHER LAB");
            ui.separator();
            ui.selectable_value(&mut self.selected_tab, Tab::Dashboard, "📊 Dashboard");
            ui.selectable_value(&mut self.selected_tab, Tab::Materials, "💎 Materials");
            ui.selectable_value(&mut self.selected_tab, Tab::Compatibility, "🧬 Compatibility");
            ui.selectable_value(&mut self.selected_tab, Tab::Experiments, "🧪 Experiments");
            
            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 0.0;
                    ui.label("iNFAMØUS ");
                    ui.label(egui::RichText::new("OS").strong());
                });
            });
        });
    }

    fn draw_materials_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("Materials Inventory");
        ui.separator();

        if self.materials.is_empty() {
            ui.label("No materials found in the database.");
            return;
        }

        egui::ScrollArea::vertical().show(ui, |ui| {
            for mat in &self.materials {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.strong(&mat.name);
                        ui.label(format!("({:?})", mat.category));
                    });
                    ui.label(format!("ID: {}", mat.id));
                    if let Some(cryst) = &mat.crystal {
                        ui.label(format!("Crystal System: {:?}", cryst.system));
                    }
                });
            }
        });
    }
}

impl eframe::App for AetherApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.draw_sidebar(ctx);

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(err) = &self.error_msg {
                ui.colored_label(egui::Color32::RED, err);
            }

            match self.selected_tab {
                Tab::Dashboard => {
                    ui.heading("Dashboard");
                    ui.label("Welcome to AETHER LAB.");
                    ui.label(format!("Total materials loaded: {}", self.materials.len()));
                    
                    if ui.button("Refresh Data").clicked() {
                        if let Ok(db_guard) = self.db.lock() {
                            if let Some(db) = db_guard.as_ref() {
                                if let Ok(mats) = db.list_materials(None) {
                                    self.materials = mats;
                                }
                            }
                        }
                    }
                }
                Tab::Materials => self.draw_materials_tab(ui),
                Tab::Compatibility => {
                    ui.heading("Compatibility Engine");
                    ui.label("Select two materials to compute compatibility metrics.");
                    // Coming soon
                }
                Tab::Experiments => {
                    ui.heading("Quantum & RC Experiments");
                    ui.label("Run advanced python simulations directly from the AETHER interface.");
                    ui.add_space(10.0);
                    
                    let is_running = *self.is_sim_running.lock().unwrap();
                    
                    if is_running {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("Simulation is running in the background...");
                        });
                    } else {
                        if ui.button("▶ Run Quantum Annealing (SQA)").clicked() {
                            *self.is_sim_running.lock().unwrap() = true;
                            *self.sim_output.lock().unwrap() = Some("Initializing OpenJij environment...\n".to_string());
                            
                            let sim_out = self.sim_output.clone();
                            let is_run = self.is_sim_running.clone();
                            
                            thread::spawn(move || {
                                let output = Command::new("python")
                                    .arg("-m")
                                    .arg("research.quantum_annealing.annealer")
                                    .output();
                                    
                                let mut final_text = String::new();
                                match output {
                                    Ok(o) => {
                                        final_text.push_str(&String::from_utf8_lossy(&o.stdout));
                                        final_text.push_str(&String::from_utf8_lossy(&o.stderr));
                                    }
                                    Err(e) => {
                                        final_text.push_str(&format!("Failed to execute python: {}", e));
                                    }
                                }
                                *sim_out.lock().unwrap() = Some(final_text);
                                *is_run.lock().unwrap() = false;
                            });
                        }
                    }
                    
                    ui.add_space(20.0);
                    ui.heading("Simulation Output");
                    egui::ScrollArea::vertical().max_height(400.0).show(ui, |ui| {
                        if let Some(out) = self.sim_output.lock().unwrap().as_ref() {
                            ui.add(egui::TextEdit::multiline(&mut out.as_str()).desired_width(f32::INFINITY).font(egui::TextStyle::Monospace));
                        } else {
                            ui.label(egui::RichText::new("No active simulations.").weak());
                        }
                    });
                }
            }
        });
    }
}
