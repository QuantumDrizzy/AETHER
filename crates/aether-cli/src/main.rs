use clap::{Parser, Subcommand};
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
        Commands::Compat { cmd } => match cmd {
            CompatCommands::Compute { material_a, material_b } => {
                println!("Computing compatibility between {} and {}", material_a, material_b);
            }
            CompatCommands::Matrix => {
                println!("Generating compatibility matrix...");
            }
        },
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
