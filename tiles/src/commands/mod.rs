// Module that handles CLI commands

use owo_colors::OwoColorize;
use tiles::runtime::Runtime;
use tiles::utils::config::set_memory_path;
use tiles::{core::health, runtime::RunArgs};

pub async fn run(runtime: &Runtime, run_args: RunArgs) {
    runtime.run(run_args).await;
}

pub fn set_memory(path: &str) {
    match set_memory_path(path) {
        Ok(msg) => {
            println!("{}", msg.green());
        }
        Err(err) => {
            let error_msg = format!("Error setting memory path due to {:?}", err);
            println!("{}", error_msg.red());
        }
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
