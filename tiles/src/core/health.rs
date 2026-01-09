// Contains functions for health checking various dependencies

use std::{env, process::Command};

use crate::runtime::mlx::ping;

pub async fn check_health() {
    let os = env::consts::OS;
    println!("Running diagnosis...");
    check_python3();
    if os == "macos" {
        check_server_status().await
    }
}

fn check_python3() {
    let output = Command::new("python3")
        .arg("--version")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());

    if let Some(version) = output {
        println!("Python3: ✅ {}", version)
    } else {
        println!("Python3: ❌ hint: Install Python3 for your OS")
    }
}
async fn check_server_status() {
    if ping().await.is_ok() {
        println!("Model server is UP: ✅")
    } else {
        println!("Model server is DOWN: ❌, try `tiles server start`")
    }
}
