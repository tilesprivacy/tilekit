use crate::runtime::RunArgs;
use crate::utils::config::{
    create_default_memory_folder, get_config_dir, get_default_memory_path, get_memory_path,
    get_server_dir, set_memory_path,
};
use crate::utils::hf_model_downloader::*;
use anyhow::{Context, Result};
use futures_util::StreamExt;
use owo_colors::OwoColorize;
use reqwest::{Client, StatusCode};
use serde_json::{Value, json};
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::str::FromStr;
use std::time::Duration;
use std::{io, process::Command};
use tilekit::modelfile::Modelfile;
use tokio::time::sleep;
pub struct MLXRuntime {}

impl MLXRuntime {}
pub struct ChatResponse {
    // think: String,
    reply: String,
    code: String,
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
        //Parse modelfile
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
) -> Result<()> {
    if !cfg!(debug_assertions) {
        let _ = mlx_runtime.start_server_daemon().await.inspect_err(|e| {
            eprintln!("Failed to start daemon server due to {:?}", e);
        });
        let _ = wait_until_server_is_up().await;
    }
    // loading the model from mem-agent via daemon server
    let memory_path = get_or_set_memory_path().context("Setting/Retrieving memory_path failed")?;
    let modelname = modelfile.from.as_ref().unwrap();
    match load_model(modelname, &memory_path).await {
        Ok(_) => start_repl(mlx_runtime, modelname, run_args).await,
        Err(err) => println!("{}", err),
    }
    Ok(())
}

fn get_or_set_memory_path() -> Result<String> {
    match get_memory_path() {
        Ok(memory_path) => Ok(memory_path),
        Err(_err) => {
            let stdin = io::stdin();
            let mut default_memory = get_default_memory_path()?;
            let mut chose_yes = false;

            println!(
                "{}",
                format!(
                    "Default Memory location will be set at {:?}\n",
                    default_memory.to_str().unwrap()
                )
                .yellow()
            );
            println!("You can always change the location with `tiles memory set-path <PATH>``\n");
            println!("Do you want to add a custom memory location right now instead? [Y/N]");
            loop {
                let mut input = String::new();
                stdin.read_line(&mut input).unwrap();
                input = input.trim().to_owned();
                println!("{}", input);
                if (input == "Y" || input == "y") || chose_yes {
                    if !chose_yes {
                        chose_yes = true;
                        println!("Add the path for your custom memory");
                        continue;
                    }
                    match set_memory_path(input.as_str()) {
                        Ok(msg) => {
                            default_memory = PathBuf::from_str(input.as_str())?;
                            println!("{}", msg.green());
                            println!(
                                "You can always change the location with tiles memory set-path <PATH>"
                            );
                            break;
                        }
                        Err(err) => {
                            let error_msg =
                                format!("Try again, Error setting memory path due to {:?}", err);
                            println!("{}", error_msg.red());
                            continue;
                        }
                    }
                } else {
                    create_default_memory_folder()?;
                    match set_memory_path(default_memory.to_str().unwrap()) {
                        Ok(msg) => {
                            println!("{}", msg.green());
                            println!(
                                "You can always change the location with tiles memory set-path <PATH>"
                            );
                            break;
                        }
                        Err(err) => {
                            let error_msg = format!("Error setting memory path due to {:?}", err);
                            println!("{}", error_msg.red());
                            continue;
                        }
                    }
                }
            }
            let default_memory_str = default_memory.to_str().unwrap();
            Ok(default_memory_str.to_owned())
        }
    }
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
                return Ok(convert_to_chat_response(&accumulated));
            }
            // Parse JSON
            let v: Value = serde_json::from_str(data).unwrap();
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

fn convert_to_chat_response(content: &str) -> ChatResponse {
    // content.split()
    ChatResponse {
        reply: extract_reply(content),
        code: extract_python(content),
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
