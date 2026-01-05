// Module that handles CLI commands

use anyhow::Result;
use tilekit::{modelfile, modelfile::Modelfile};
use tiles::runtime::Runtime;
use tiles::{core::health, runtime::RunArgs};
const DEFAULT_MODELFILE: &str = "
  FROM driaforall/mem-agent-mlx-4bit 
";

pub async fn run(runtime: &Runtime, modelfile: Option<String>) {
    let modelfile_parse_result: Result<Modelfile, String> = if let Some(modelfile_str) = modelfile {
        modelfile::parse_from_file(modelfile_str.as_str())
    } else {
        modelfile::parse(DEFAULT_MODELFILE)
    };
    match modelfile_parse_result {
        Ok(modelfile) => {
            let run_args = RunArgs { modelfile };
            runtime.run(run_args).await;
        }
        Err(_err) => println!("Invalid Modelfile"),
    }
}

pub fn check_health() {
    health::check_health();
}

pub async fn start_server(runtime: &Runtime) {
    let _ = runtime.start_server_daemon().await;
}

pub async fn stop_server(runtime: &Runtime) {
    let _ = runtime.stop_server_daemon().await;
}
