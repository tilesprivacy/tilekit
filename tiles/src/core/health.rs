// Contains functions for health checking various dependencies

use std::{env, process::Command};

//TODO: In future we can install the dependencies if not there..
pub fn check_health() {
    let os = env::consts::OS;
    println!("Running diagnosis...");
    check_python3();
    if os == "macos" {
        check_mlx_lm()
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

fn check_mlx_lm() {
    match Command::new("mlx_lm.chat").arg("--help").output() {
        Ok(_) => println!("mlx_lm: ✅"),
        _ => println!("mlx_lm: ❌ hint: run `pip install mlx-lm`"),
    }
}
