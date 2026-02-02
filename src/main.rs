use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "llama-rs")]
#[command(about = "Rust implementation of llama.cpp", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Info { model: String },
    Run {
        model: String,
        #[arg(short, long)]
        prompt: Option<String>,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Info { model } => println!("Model info: {}", model),
        Commands::Run { model, prompt } => {
            println!("Running model: {}", model);
            if let Some(p) = prompt { println!("Prompt: {}", p); }
        }
    }
}
