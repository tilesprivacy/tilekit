// Module that handles CLI commands

use tiles::runtime::Runtime;
use tiles::{core::health, runtime::RunArgs};

pub async fn run(runtime: &Runtime, run_args: RunArgs) {
    runtime.run(run_args).await;
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
