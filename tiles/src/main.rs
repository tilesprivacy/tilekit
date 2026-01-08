use std::error::Error;

use clap::{Args, Parser, Subcommand};
use tiles::runtime::{RunArgs, build_runtime};
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
    Run {
        /// Path to the Modelfile (uses default model if not provided)
        modelfile_path: Option<String>,

        #[command(flatten)]
        flags: RunFlags,
    },

    /// Runs a benchmark and saves results to log file
    Bench {
        /// Path to the Modelfile (uses default model if not provided)
        modelfile_path: Option<String>,
    },

    /// Checks the status of dependencies
    Health,

    /// start or stop the daemon server
    Server(ServerArgs),
}

#[derive(Debug, Args)]
struct RunFlags {
    /// Max times cli communicates with the model until it gets a proper reply for a user prompt
    #[arg(short = 'r', long, default_value_t = 10)]
    relay_count: u32,
    // Future flags go here:
    // #[arg(long, default_value_t = 6969)]
    // port: u16,
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
#[tokio::main(flavor = "current_thread")]
pub async fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let runtime = build_runtime();
    match cli.command {
        Commands::Run {
            modelfile_path,
            flags,
        } => {
            let run_args = RunArgs {
                modelfile_path,
                relay_count: flags.relay_count,
            };
            commands::run(&runtime, run_args).await;
        }
        Commands::Bench { modelfile_path } => {
            let run_args = RunArgs {
                modelfile_path,
                relay_count: 0, // unused by bench
            };
            commands::bench(&runtime, run_args).await;
        }
        Commands::Health => {
            commands::check_health();
        }
        Commands::Server(server) => match server.command {
            Some(ServerCommands::Start) => commands::start_server(&runtime).await,
            Some(ServerCommands::Stop) => commands::stop_server(&runtime).await,
            _ => println!("Expected start or stop"),
        },
    }
    Ok(())
}
