use std::collections::HashMap;

use clap::{Parser, Subcommand};
use aether_core::compatibility::CompatibilityEngine;
use aether_core::material::{Material, MaterialCategory};

#[derive(Parser)]
#[command(name = "aether")]
#[command(about = "AETHER LAB - Advanced Materials Research Engine", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Material {
        #[command(subcommand)]
        cmd: MaterialCommands,
    },
    Experiment {
        #[command(subcommand)]
        cmd: ExperimentCommands,
    },
    Compat {
        #[command(subcommand)]
        cmd: CompatCommands,
    },
    Knowledge {
        #[command(subcommand)]
        cmd: KnowledgeCommands,
    },
    Init,
}

#[derive(Subcommand)]
enum MaterialCommands {
    Add {
        name: String,
        #[arg(long)]
        category: String,
    },
    List {
        #[arg(long)]
        category: Option<String>,
    },
    Seed {
        #[arg(long, default_value = "data/materials/library.json")]
        file: String,
    },
    Show {
        id_or_name: String,
    },
    Observe {
        name: String,
        #[arg(long)]
        category: String,
        #[arg(long)]
        intensity: f64,
        #[arg(long)]
        description: String,
    },
}

#[derive(Subcommand)]
enum ExperimentCommands {
    Create {
        name: String,
        #[arg(long)]
        type_: String,
        #[arg(long)]
        materials: Vec<String>,
    },
    List,
    Run {
        id: String,
    },
}

#[derive(Subcommand)]
enum CompatCommands {
    Compute {
        material_a: String,
        material_b: String,
    },
    Matrix,
}

#[derive(Subcommand)]
enum KnowledgeCommands {
    List,
    Search {
        query: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Init => {
            println!("Initializing database...");
            let _db = aether_db::AetherDb::init("data/aether.db")?;
            println!("Database initialized.");
        }
        Commands::Material { cmd } => match cmd {
            MaterialCommands::Add { name, category } => {
                println!("Adding material {} of category {}", name, category);
                let cat = match category.to_lowercase().as_str() {
                    "crystal" => MaterialCategory::Crystal,
                    "metamaterial" => MaterialCategory::Metamaterial,
                    "nanomaterial" => MaterialCategory::Nanomaterial,
                    "alloy" => MaterialCategory::Alloy,
                    "polymer" => MaterialCategory::Polymer,
                    "composite" => MaterialCategory::Composite,
                    "biological" => MaterialCategory::Biological,
                    "theoretical" => MaterialCategory::Theoretical,
                    _ => MaterialCategory::Custom(category),
                };
                let mat = Material::new(&name, cat);
                let db = aether_db::AetherDb::init("data/aether.db")?;
                db.insert_material(&mat)?;
                println!("Material added with ID: {}", mat.id);
            }
            MaterialCommands::List { category } => {
                let db = aether_db::AetherDb::init("data/aether.db")?;
                let mats = db.list_materials(category.as_deref())?;
                for mat in mats {
                    if let Some(cat) = &category {
                        // Very simple filtering
                        println!("- {} ({})", mat.name, cat);
                    } else {
                        println!("- {} ({:?})", mat.name, mat.category);
                    }
                }
            }
            MaterialCommands::Seed { file } => {
                let json = std::fs::read_to_string(&file)?;
                let mats = aether_db::library::parse_library(&json)?;
                let db = aether_db::AetherDb::init("data/aether.db")?;
                let existing: std::collections::HashSet<String> = db
                    .list_materials(None)?
                    .into_iter()
                    .map(|m| m.name.to_lowercase())
                    .collect();
                let mut added = 0usize;
                for m in mats {
                    if existing.contains(&m.name.to_lowercase()) {
                        continue;
                    }
                    db.insert_material(&m)?;
                    added += 1;
                }
                println!("Seeded {added} materials from {file} (skipped existing).");
            }
            MaterialCommands::Show { id_or_name } => {
                println!("Showing material {}", id_or_name);
            }
            MaterialCommands::Observe { name, category, intensity, description: _ } => {
                println!("Observing {} ({}) with intensity {}", name, category, intensity);
            }
        },
        Commands::Experiment { cmd } => match cmd {
            ExperimentCommands::Create { name, type_, materials: _ } => {
                println!("Creating experiment {} of type {}", name, type_);
            }
            ExperimentCommands::List => {
                println!("Listing experiments...");
            }
            ExperimentCommands::Run { id } => {
                println!("Running experiment {}", id);
            }
        },
        Commands::Compat { cmd } => {
            let db = aether_db::AetherDb::init("data/aether.db")?;
            let engine = CompatibilityEngine::new();
            let weights = HashMap::new(); // default per-dimension weights
            match cmd {
                CompatCommands::Compute { material_a, material_b } => {
                    let a = find_material(&db, &material_a)?;
                    let b = find_material(&db, &material_b)?;
                    let r = engine.compute(&a, &b, &weights);
                    println!("Compatibility: {} vs {}", a.name, b.name);
                    println!("  Overall: {:.3}  ({})", r.overall_score, r.recommendation);
                    for d in &r.dimensions {
                        println!("  {:>16}: {:.3}  [{}]", d.name, d.score, d.method);
                    }
                    for s in &r.synergies {
                        println!("  + synergy : {} ({:.2})", s.description, s.magnitude);
                    }
                    for c in &r.conflicts {
                        println!("  - conflict: {} ({:.2})", c.description, c.severity);
                    }
                }
                CompatCommands::Matrix => {
                    let mats = db.list_materials(None)?;
                    if mats.is_empty() {
                        println!("No materials in the database.");
                    } else {
                        print!("{:>14}", "");
                        for m in &mats {
                            print!(" {:>10.10}", m.name);
                        }
                        println!();
                        for a in &mats {
                            print!("{:>14.14}", a.name);
                            for b in &mats {
                                let s = engine.compute(a, b, &weights).overall_score;
                                print!(" {:>10.3}", s);
                            }
                            println!();
                        }
                    }
                }
            }
        }
        Commands::Knowledge { cmd } => match cmd {
            KnowledgeCommands::List => {
                println!("Listing knowledge entries...");
            }
            KnowledgeCommands::Search { query } => {
                println!("Searching knowledge base for {}", query);
            }
        },
    }

    Ok(())
}

/// Find a material by exact (case-insensitive) name, else the first substring match.
fn find_material(db: &aether_db::AetherDb, name: &str) -> anyhow::Result<Material> {
    let matches = db.search_materials(name)?;
    if let Some(m) = matches.iter().find(|m| m.name.eq_ignore_ascii_case(name)) {
        return Ok(m.clone());
    }
    matches
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no material matching '{}'", name))
}
