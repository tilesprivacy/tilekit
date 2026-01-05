use std::error::Error;

use clap::{Args, Parser, Subcommand};
mod commands;
#[derive(Debug, Parser)]
#[command(name = "tiles")]
#[command(version, about = "Run, fine-tune models locally with Modelfile", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Runs the given Modelfile (runs the default model if none passed)
    Run { modelfile_path: Option<String> },

    /// Checks the status of dependencies
    Health,

    /// start or stop the daemon server
    Server(ServerArgs),
}

#[derive(Debug, Args)]
#[command(args_conflicts_with_subcommands = true)]
#[command(flatten_help = true)]
struct ServerArgs {
    #[command(subcommand)]
    command: Option<ServerCommands>,
}

#[derive(Debug, Subcommand)]
enum ServerCommands {
    /// Start the py server as a daemon
    Start,

    /// Stops the daemon py server
    Stop,
}
#[tokio::main]
pub async fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run { modelfile_path } => {
            commands::run(modelfile_path).await;
        }
        Commands::Health => {
            commands::check_health();
        }
        Commands::Server(server) => match server.command {
            Some(ServerCommands::Start) => commands::start_server().await,
            Some(ServerCommands::Stop) => commands::stop_server().await,
            _ => println!("Expected start or stop"),
        },
    }
    Ok(())
}
