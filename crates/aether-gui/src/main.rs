#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

fn main() -> eframe::Result<()> {
    // tracing_subscriber::fmt::init();
    
    let native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 700.0])
            .with_title("AETHER LAB"),
        ..Default::default()
    };
    
    eframe::run_native(
        "AETHER LAB",
        native_options,
        Box::new(|_cc| {
            Box::new(aether_gui::AetherApp::new())
        }),
    )
}
