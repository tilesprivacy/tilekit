// Module that handles CLI commands

use anyhow::Result;
use tiles::{
    core::{
        health,
        modelfile::{self, Modelfile},
    },
    runner::mlx,
};

const DEFAULT_MODELFILE: &str = "
  FROM driaforall/mem-agent-mlx-4bit 
";

pub async fn run(modelfile: Option<String>) {
    let modelfile_parse_result: Result<Modelfile, String> = if let Some(modelfile_str) = modelfile {
        modelfile::parse_from_file(modelfile_str.as_str())
    } else {
        modelfile::parse(DEFAULT_MODELFILE)
    };

    match modelfile_parse_result {
        Ok(modelfile) => {
            mlx::run(modelfile).await;
        }
        Err(err) => println!("{}", err),
    }
}

pub fn check_health() {
    health::check_health();
}

pub async fn start_server() {
    let _ = mlx::start_server_daemon().await;
}

pub async fn stop_server() {
    let _ = mlx::stop_server_daemon().await;
}
