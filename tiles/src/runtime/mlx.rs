use crate::runtime::RunArgs;
use crate::utils::hf_model_downloader::*;
use anyhow::{Context, Result};
use futures_util::StreamExt;
use owo_colors::OwoColorize;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;
use std::{env, fs};
use std::{io, process::Command};
use tilekit::modelfile::Modelfile;
use tokio::time::sleep;
pub struct MLXRuntime {}

impl MLXRuntime {}

#[derive(Debug, Deserialize, Serialize)]
pub struct BenchmarkMetrics {
    ttft_ms: f64,
    total_tokens: i32,
    tokens_per_second: f64,
    total_latency_s: f64,
}

pub struct ChatResponse {
    // think: String,
    reply: String,
    code: String,
    metrics: Option<BenchmarkMetrics>,
}

impl Default for MLXRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl MLXRuntime {
    pub fn new() -> Self {
        MLXRuntime {}
    }

    pub async fn run(&self, run_args: super::RunArgs) {
        const DEFAULT_MODELFILE: &str = "FROM driaforall/mem-agent-mlx-4bit";

        // Parse modelfile
        let modelfile_parse_result = if let Some(modelfile_str) = &run_args.modelfile_path {
            tilekit::modelfile::parse_from_file(modelfile_str.as_str())
        } else {
            tilekit::modelfile::parse(DEFAULT_MODELFILE)
        };

        let modelfile = match modelfile_parse_result {
            Ok(mf) => mf,
            Err(_err) => {
                println!("Invalid Modelfile");
                return;
            }
        };

        let model = modelfile.from.as_ref().unwrap();
        if model.starts_with("driaforall/mem-agent") {
            let _res = run_model_with_server(self, modelfile, &run_args).await;
        } else {
            run_model_by_sub_process(modelfile);
        }
    }

    #[allow(clippy::zombie_processes)]
    pub async fn start_server_daemon(&self) -> Result<()> {
        // check if the server is running
        // start server as a child process
        // save the pid in a file under ~/.config/tiles/server_pid

        if (ping().await).is_ok() {
            println!("server is already up");
            return Ok(());
        }

        let config_dir = get_config_dir()?;
        let mut server_dir = get_server_dir()?;
        let pid_file = config_dir.join("server.pid");
        fs::create_dir_all(&config_dir).context("Failed to create config directory")?;

        let stdout_log = File::create(config_dir.join("server.out.log"))?;
        let stderr_log = File::create(config_dir.join("server.err.log"))?;
        let server_path = server_dir.join(".venv/bin/python3");
        server_dir.pop();
        let child = Command::new(server_path)
            .args(["-m", "server.main"])
            .current_dir(server_dir)
            .env("PYTHONDONTWRITEBYTECODE", "1") // Disable .pyc caching
            .stdin(Stdio::null())
            .stdout(Stdio::from(stdout_log))
            .stderr(Stdio::from(stderr_log))
            .spawn()
            .expect("failed to start server");

        fs::create_dir_all(&config_dir).context("Failed to create config directory")?;
        std::fs::write(pid_file, child.id().to_string()).unwrap();
        println!("Server started with PID {}", child.id());
        Ok(())
    }

    pub async fn stop_server_daemon(&self) -> Result<()> {
        if (ping().await).is_err() {
            println!("Server is not running");
            return Ok(());
        }
        let pid_file = get_config_dir()?.join("server.pid");

        if !pid_file.exists() {
            eprintln!("server pid doesnt exist");
            return Ok(());
        }

        let pid = std::fs::read_to_string(&pid_file).unwrap();
        Command::new("kill").arg(pid.trim()).status().unwrap();
        std::fs::remove_file(pid_file).unwrap();
        println!("Server stopped.");
        Ok(())
    }

    pub async fn bench(&self, run_args: super::RunArgs) {
        const DEFAULT_MODELFILE: &str = "FROM driaforall/mem-agent-mlx-4bit";

        // Parse modelfile
        let modelfile_parse_result = if let Some(modelfile_str) = run_args.modelfile_path {
            tilekit::modelfile::parse_from_file(modelfile_str.as_str())
        } else {
            tilekit::modelfile::parse(DEFAULT_MODELFILE)
        };

        let modelfile = match modelfile_parse_result {
            Ok(mf) => mf,
            Err(_err) => {
                println!("Invalid Modelfile");
                return;
            }
        };

        let modelname = modelfile.from.as_ref().unwrap();

        // Start server if not running
        if (ping().await).is_err() {
            println!("Starting server...");
            let _res = self.start_server_daemon().await;
            let _ = wait_until_server_is_up().await;
        }

        // Load model
        let memory_path = get_memory_path()
            .context("Retrieving memory_path failed")
            .unwrap();
        if let Err(e) = load_model(modelname, &memory_path).await {
            println!("Failed to load model: {}", e);
            return;
        }

        println!("Running benchmark...");

        // Run benchmark prompt
        let benchmark_prompt = "What is 2+2? Please answer concisely.";
        match chat(benchmark_prompt, modelname, true, "").await {
            Ok(response) => {
                if let Some(metrics) = response.metrics {
                    println!("\n{} Benchmark Results:", "✓".green());
                    println!("  TTFT:       {:.0}ms", metrics.ttft_ms);
                    println!("  Throughput: {:.1} tok/s", metrics.tokens_per_second);
                    println!("  Tokens:     {}", metrics.total_tokens);
                    println!("  Latency:    {:.2}s", metrics.total_latency_s);

                    // Save to log file
                    save_benchmark_to_log(&metrics, modelname).unwrap_or_else(|e| {
                        eprintln!("Failed to save benchmark log: {}", e);
                    });
                } else {
                    println!("No metrics returned");
                }
            }
            Err(e) => {
                println!("Benchmark failed: {}", e);
            }
        }
    }
}

fn save_benchmark_to_log(metrics: &BenchmarkMetrics, model: &str) -> Result<()> {
    use chrono::Local;

    let config_dir = get_config_dir()?;
    let log_file = config_dir.join("benchmark_log.jsonl");

    // Create benchmark log entry
    let log_entry = serde_json::json!({
        "timestamp": Local::now().to_rfc3339(),
        "model": model,
        "ttft_ms": metrics.ttft_ms,
        "total_tokens": metrics.total_tokens,
        "tokens_per_second": metrics.tokens_per_second,
        "total_latency_s": metrics.total_latency_s,
    });

    // Append to log file
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_file)?;

    use std::io::Write;
    writeln!(file, "{}", log_entry)?;

    println!("\n{} Saved to: {}", "💾".cyan(), log_file.display());

    Ok(())
}

fn run_model_by_sub_process(modelfile: Modelfile) {
    // build the arg list from modelfile
    let mut args: Vec<String> = vec![];
    args.push("--model".to_owned());
    args.push(modelfile.from.unwrap());
    for parameter in modelfile.parameters {
        let param_value = parameter.value.to_string();
        match parameter.param_type.as_str() {
            "num_predict" => {
                args.push("--max-tokens".to_owned());
                args.push(param_value);
            }
            "temperature" => {
                args.push("--temp".to_owned());
                args.push(param_value);
            }
            "top_p" => {
                args.push("--top-p".to_owned());
                args.push(param_value);
            }
            "seed" => {
                args.push("--seed".to_owned());
                args.push(param_value);
            }
            _ => {}
        }
    }
    if let Some(system_prompt) = modelfile.system {
        args.push("--system-prompt".to_owned());
        args.push(system_prompt);
    }
    if let Some(adapter_path) = modelfile.adapter {
        args.push("--adapter-path".to_owned());
        args.push(adapter_path);
    }
    let mut mlx = match Command::new("mlx_lm.chat").args(args).spawn() {
        Ok(child) => child,
        Err(e) => {
            if e.kind() == std::io::ErrorKind::NotFound {
                eprintln!("❌ Error: mlx_lm.chat command not found");
                eprintln!("💡 Hint: Install mlx-lm by running: pip install mlx-lm");
                eprintln!("📝 Note: mlx-lm is only available on macOS with Apple Silicon");
                std::process::exit(1);
            } else {
                eprintln!("❌ Error: Failed to spawn mlx_lm.chat: {}", e);
                std::process::exit(1);
            }
        }
    };

    if let Err(err) = mlx.wait() {
        eprintln!("❌ Error: Failed to wait for mlx_lm: {}", err);
    }
}

async fn run_model_with_server(
    mlx_runtime: &MLXRuntime,
    modelfile: Modelfile,
    run_args: &RunArgs,
) -> reqwest::Result<()> {
    if !cfg!(debug_assertions) {
        let _res = mlx_runtime.start_server_daemon().await;
        let _ = wait_until_server_is_up().await;
    }
    // loading the model from mem-agent via daemon server
    let memory_path = get_memory_path()
        .context("Retrieving memory_path failed")
        .unwrap();
    let modelname = modelfile.from.as_ref().unwrap();
    match load_model(modelname, &memory_path).await {
        Ok(_) => start_repl(mlx_runtime, modelname, run_args).await,
        Err(err) => println!("{}", err),
    }
    Ok(())
}

async fn start_repl(mlx_runtime: &MLXRuntime, modelname: &str, run_args: &RunArgs) {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    println!("Running in interactive mode");
    // TODO: Handle "enter" key press or any key press when repl is processing an input
    loop {
        print!(">> ");
        stdout.flush().unwrap();
        let mut input = String::new();
        stdin.read_line(&mut input).unwrap();
        let input = input.trim();
        match input {
            "exit" => {
                println!("Exiting interactive mode");
                if !cfg!(debug_assertions) {
                    let _res = mlx_runtime.stop_server_daemon().await;
                }
                break;
            }
            _ => {
                let mut remaining_count = run_args.relay_count;
                let mut g_reply: String = "".to_owned();
                let mut python_code: String = "".to_owned();
                loop {
                    if remaining_count > 0 {
                        let chat_start = remaining_count == run_args.relay_count;
                        if let Ok(response) = chat(input, modelname, chat_start, &python_code).await
                        {
                            if response.reply.is_empty() {
                                if !response.code.is_empty() {
                                    python_code = response.code;
                                }
                                remaining_count -= 1;
                            } else {
                                g_reply = response.reply.clone();
                                println!("\n>> {}", response.reply.trim());

                                // Display benchmark metrics if available
                                if let Some(metrics) = response.metrics {
                                    println!(
                                        "\n{} {:.1} tok/s | {} tokens | {:.0}ms TTFT",
                                        "💡".yellow(),
                                        metrics.tokens_per_second,
                                        metrics.total_tokens,
                                        metrics.ttft_ms
                                    );
                                }

                                break;
                            }
                        } else {
                            println!("\n>> failed to respond");
                            break;
                        }
                    }
                }
                if g_reply.is_empty() {
                    println!(">> No reply")
                }
            }
        }
    }
}

async fn ping() -> Result<(), String> {
    let client = Client::new();
    let res = client.get("http://127.0.0.1:6969/ping").send().await;

    match res {
        Err(_) => Err(String::from("Server is down")),
        _ => Ok(()),
    }
}

async fn load_model(model_name: &str, memory_path: &str) -> Result<(), String> {
    let client = Client::new();
    let body = json!({
        "model": model_name,
        "memory_path": memory_path
    });

    //TODO: Fix the unwrap here
    let res = client
        .post("http://127.0.0.1:6969/start")
        .json(&body)
        .send()
        .await
        .unwrap();
    match res.status() {
        StatusCode::OK => Ok(()),
        StatusCode::NOT_FOUND => {
            println!("Downloading {}\n", model_name);
            match pull_model(model_name).await {
                Ok(_) => {
                    println!("\nDownloading completed \n");
                    Ok(())
                }
                Err(err) => Err(err),
            }
        }
        _ => {
            println!("err {:?}", res);
            Err(format!(
                "Failed to load model {} due to {:?}",
                model_name, res
            ))
        }
    }
}

async fn chat(
    input: &str,
    model_name: &str,
    chat_start: bool,
    python_code: &str,
) -> Result<ChatResponse, String> {
    let client = Client::new();

    let body = json!({
        "model": model_name,
        "chat_start": chat_start,
        "stream": true,
        "python_code": python_code,
        "messages": [{"role": "user", "content": input}]
    });
    let res = client
        .post("http://127.0.0.1:6969/v1/chat/completions")
        .json(&body)
        .send()
        .await
        .unwrap();

    let mut stream = res.bytes_stream();
    let mut accumulated = String::new();
    let mut metrics: Option<BenchmarkMetrics> = None;
    println!();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        let s = String::from_utf8_lossy(&chunk);
        for line in s.lines() {
            if !line.starts_with("data: ") {
                continue;
            }

            let data = line.trim_start_matches("data: ");

            if data == "[DONE]" {
                return Ok(convert_to_chat_response(&accumulated, metrics));
            }
            // Parse JSON
            let v: Value = serde_json::from_str(data).unwrap();

            // Check for metrics in the response
            if let Some(metrics_obj) = v.get("metrics") {
                metrics = serde_json::from_value(metrics_obj.clone()).ok();
            }

            if let Some(delta) = v["choices"][0]["delta"]["content"].as_str() {
                accumulated.push_str(delta);
                print!("{}", delta.dimmed());
                use std::io::Write;
                std::io::stdout().flush().ok();
            }
        }
    }
    Err(String::from("request failed"))
}

fn convert_to_chat_response(content: &str, metrics: Option<BenchmarkMetrics>) -> ChatResponse {
    ChatResponse {
        reply: extract_reply(content),
        code: extract_python(content),
        metrics,
    }
}

fn extract_reply(content: &str) -> String {
    if content.contains("<reply>") && content.contains("</reply>") {
        let list_a = content.split("<reply>").collect::<Vec<&str>>();
        let list_b = list_a[1].split("</reply>").collect::<Vec<&str>>();
        list_b[0].to_owned()
    } else {
        "".to_owned()
    }
}

fn extract_python(content: &str) -> String {
    if content.contains("<python>") && content.contains("</python>") {
        let list_a = content.split("<python>").collect::<Vec<&str>>();
        let list_b = list_a[1].split("</python>").collect::<Vec<&str>>();
        list_b[0].to_owned()
    } else {
        "".to_owned()
    }
}

// fn extract_think(content: &str) -> String {
//     if content.contains("<think>") && content.contains("</think>") {
//         let list_a = content.split("<think>").collect::<Vec<&str>>();
//         let list_b = list_a[1].split("</think>").collect::<Vec<&str>>();
//         list_b[0].to_owned()
//     } else if content.contains("</think") {
//         let list_a = content.split("</think>").collect::<Vec<&str>>();
//         list_a[0].to_owned()
//     } else {
//         "".to_owned()
//     }
// }

fn get_memory_path() -> Result<String> {
    let tiles_config_dir = get_config_dir()?;
    let tiles_data_dir = get_data_dir()?;
    let mut is_memory_path_found: bool = false;
    let mut memory_path: String = String::from("");
    if tiles_config_dir.is_dir()
        && let Ok(content) = fs::read_to_string(tiles_config_dir.join(".memory_path"))
    {
        memory_path = content;
        is_memory_path_found = true;
    }

    if is_memory_path_found {
        Ok(memory_path)
    } else {
        let memory_path = tiles_data_dir.join("memory");
        fs::create_dir_all(&memory_path).context("Failed to create tiles memory directory")?;
        fs::create_dir_all(&tiles_config_dir).context("Failed to create tiles config directory")?;
        fs::write(
            tiles_config_dir.join(".memory_path"),
            memory_path.to_str().unwrap(),
        )
        .context("Failed to write the default path to .memory_path")?;
        Ok(memory_path.to_string_lossy().to_string())
    }
}

fn get_server_dir() -> Result<PathBuf> {
    if cfg!(debug_assertions) {
        let base_dir = env::current_dir().context("Failed to fetch CURRENT_DIR")?;
        Ok(base_dir.join("server"))
    } else {
        let home_dir = env::home_dir().context("Failed to fetch $HOME")?;
        let data_dir = match env::var("XDG_DATA_HOME") {
            Ok(val) => PathBuf::from(val),
            Err(_err) => home_dir.join(".local/share"),
        };
        Ok(data_dir.join("tiles/server"))
    }
}
fn get_config_dir() -> Result<PathBuf> {
    if cfg!(debug_assertions) {
        let base_dir = env::current_dir().context("Failed to fetch CURRENT_DIR")?;
        Ok(base_dir.join(".tiles_dev/tiles"))
    } else {
        let home_dir = env::home_dir().context("Failed to fetch $HOME")?;
        let config_dir = match env::var("XDG_CONFIG_HOME") {
            Ok(val) => PathBuf::from(val),
            Err(_err) => home_dir.join(".config"),
        };
        Ok(config_dir.join("tiles"))
    }
}

fn get_data_dir() -> Result<PathBuf> {
    if cfg!(debug_assertions) {
        let base_dir = env::current_dir().context("Failed to fetch CURRENT_DIR")?;
        Ok(base_dir.join(".tiles_dev/tiles"))
    } else {
        let home_dir = env::home_dir().context("Failed to fetch $HOME")?;
        let data_dir = match env::var("XDG_DATA_HOME") {
            Ok(val) => PathBuf::from(val),
            Err(_err) => home_dir.join(".local/share"),
        };
        Ok(data_dir.join("tiles"))
    }
}

async fn wait_until_server_is_up() {
    loop {
        match ping().await {
            Ok(()) => {
                break;
            }
            Err(_) => {
                println!("tiling...");
                sleep(Duration::from_secs(5)).await;
            }
        }
    }
}
