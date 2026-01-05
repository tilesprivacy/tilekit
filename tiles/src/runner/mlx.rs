use anyhow::{Context, Result};
use futures_util::StreamExt;
use owo_colors::OwoColorize;
use reqwest::{Client, StatusCode};
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
pub struct ChatResponse {
    // think: String,
    reply: String,
    code: String,
}

pub async fn run(modelfile: Modelfile) {
    let model = modelfile.from.as_ref().unwrap();
    if model.starts_with("driaforall/mem-agent") {
        let _res = run_model_with_server(modelfile).await;
    } else {
        run_model_by_sub_process(modelfile);
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

#[allow(clippy::zombie_processes)]
pub async fn start_server_daemon() -> Result<()> {
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

pub async fn stop_server_daemon() -> Result<()> {
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
async fn run_model_with_server(modelfile: Modelfile) -> reqwest::Result<()> {
    if !cfg!(debug_assertions) {
        let _res = start_server_daemon().await;
        let _ = wait_until_server_is_up().await;
    }
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    // loading the model from mem-agent via daemon server
    let memory_path = get_memory_path()
        .context("Retrieving memory_path failed")
        .unwrap();
    let modelname = modelfile.from.as_ref().unwrap();
    load_model(modelname, &memory_path).await.unwrap();
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
                    let _res = stop_server_daemon().await;
                }
                break;
            }
            _ => {
                let mut remaining_count = 6;
                let mut g_reply: String = "".to_owned();
                let mut python_code: String = "".to_owned();
                loop {
                    if remaining_count > 0 {
                        let chat_start = remaining_count == 6;
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
    Ok(())
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
    let res = client
        .post("http://127.0.0.1:6969/start")
        .json(&body)
        .send()
        .await
        .unwrap();
    match res.status() {
        StatusCode::OK => Ok(()),
        StatusCode::NOT_FOUND => download_model(model_name).await,
        _ => {
            println!("err {:?}", res);
            Ok(())
        }
    }
}

async fn download_model(model_name: &str) -> Result<(), String> {
    println!("Downloading the model {} ....", model_name);
    let client = Client::new();
    let body = json!({
        "model": model_name
    });
    let res = client
        .post("http://127.0.0.1:6969/download")
        .json(&body)
        .send()
        .await
        .unwrap();
    if res.status() == 200 {
        Ok(())
    } else {
        Err(String::from("Downloading model failed"))
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
