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
use rustyline::completion::Completer;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::history::DefaultHistory;
use rustyline::validate::Validator;
use rustyline::{Config, Editor, Helper};
use serde_json::{Value, json};
use std::fs;
use std::fs::File;
use std::io::{self, Write};
use std::process::{Command, Stdio};
use std::time::Duration;

use tilekit::modelfile::Modelfile;
use tokio::time::sleep;
pub struct MLXRuntime {}

impl MLXRuntime {}
pub struct ChatResponse {
    #[allow(dead_code)]
    analysis: String,
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
        // Track-aware model selection
        // TILES_TRACK=regular (default): Use gpt-oss model
        // TILES_TRACK=insider: Use Dria model
        let default_modelfile = match std::env::var("TILES_TRACK")
            .unwrap_or_else(|_| "regular".to_string())
            .to_lowercase()
            .as_str()
        {
            "insider" => "FROM driaforall/mem-agent-mlx-4bit",
            _ => "FROM mlx-community/gpt-oss-20b-MXFP4-Q4",
        };

        //Parse modelfile
        let modelfile_parse_result = if let Some(modelfile_str) = &run_args.modelfile_path {
            tilekit::modelfile::parse_from_file(modelfile_str.as_str())
        } else {
            tilekit::modelfile::parse(default_modelfile)
        };

        let modelfile = match modelfile_parse_result {
            Ok(mf) => mf,
            Err(_err) => {
                println!("Invalid Modelfile");
                return;
            }
        };

        let _res = run_model_with_server(self, modelfile, &run_args)
            .await
            .inspect_err(|e| eprintln!("Failed to run the model due to {e}"));
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

#[allow(dead_code)]
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

struct TilesHinter;

impl Hinter for TilesHinter {
    type Hint = String;

    fn hint(&self, line: &str, _pos: usize, _ctx: &rustyline::Context<'_>) -> Option<Self::Hint> {
        if line.is_empty() {
            Some("Send a message (/? for help)".to_string())
        } else {
            None
        }
    }
}

impl Completer for TilesHinter {
    type Candidate = String;
}

impl Highlighter for TilesHinter {
    fn highlight_hint<'h>(&self, hint: &'h str) -> std::borrow::Cow<'h, str> {
        std::borrow::Cow::Owned(format!("\x1b[2m{}\x1b[0m", hint))
    }
}

impl Validator for TilesHinter {}

impl Helper for TilesHinter {}

enum SlashCommand {
    Continue,
    Exit,
    NotACommand,
}

fn handle_slash_command(input: &str, modelname: &str) -> SlashCommand {
    if let Some(cmd) = input.strip_prefix('/') {
        match cmd {
            "help" | "?" => {
                show_help(modelname);
                SlashCommand::Continue
            }
            "bye" => SlashCommand::Exit,
            "" => {
                println!("Empty command. Type /help for available commands.");
                SlashCommand::Continue
            }
            _ => {
                println!(
                    "Unknown command: /{}. Type /help for available commands.",
                    cmd
                );
                SlashCommand::Continue
            }
        }
    } else {
        SlashCommand::NotACommand
    }
}

fn show_help(model_name: &str) {
    println!("\n=== Tiles REPL Help ===\n");

    println!("Available Commands:");
    println!("  /?          Show this help message");
    println!("  /help       Show this help message");
    println!("  /bye        Exit the REPL");
    println!();

    println!("Current Model:");
    println!("  {}", model_name);
    println!();

    println!("Usage Tips:");
    println!("  - Type your questions or prompts directly");
    println!("  - Model outputs <think>, <python>, and <reply> tags");
    println!("  - Only <reply> content is shown as final output");
    println!();
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
    // Pass system prompt from modelfile if available
    let system_prompt = modelfile.system.as_deref();
    match load_model(modelname, &memory_path, system_prompt).await {
        Ok(_) => start_repl(mlx_runtime, modelname, run_args).await,
        Err(err) => return Err(anyhow::anyhow!(err)),
    }
    Ok(())
}

fn get_or_set_memory_path() -> Result<String> {
    match get_memory_path() {
        Ok(memory_path) => Ok(memory_path),
        Err(_err) => {
            let stdin = io::stdin();
            let default_memory_pathbuf = get_default_memory_path()?;
            let mut default_memory = default_memory_pathbuf
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("Invalid path"))?;
            let mut chose_yes = false;

            println!(
                "{}",
                format!(
                    "Default Memory location will be set at {:?}\n",
                    default_memory
                )
                .yellow()
            );
            println!("You can always change the location with `tiles memory set-path <PATH>`\n");
            println!("Do you want to add a custom memory location right now instead? [Y/N]");
            let mut input = String::new();
            loop {
                input.clear();
                stdin.read_line(&mut input)?;
                input = input.trim().to_owned();
                if (input == "Y" || input == "y") || chose_yes {
                    if !chose_yes {
                        chose_yes = true;
                        println!("Add the path for your custom memory");
                        continue;
                    }
                    match set_memory_path(input.as_str()) {
                        Ok(msg) => {
                            default_memory = input.as_str();
                            println!("{}", msg.green());
                            println!(
                                "You can always change the location with `tiles memory set-path <PATH>`\n"
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
                    match set_memory_path(default_memory) {
                        Ok(msg) => {
                            println!("{}", msg.green());
                            println!(
                                "You can always change the location with `tiles memory set-path <PATH>`\n"
                            );
                            break;
                        }
                        Err(err) => {
                            let error_msg = format!("Error setting memory path due to {:?}", err);
                            println!("{}", error_msg.red());
                            return Err(anyhow::anyhow!("Error setting default memory path"));
                        }
                    }
                }
            }
            Ok(default_memory.to_owned())
        }
    }
}

async fn start_repl(mlx_runtime: &MLXRuntime, modelname: &str, run_args: &RunArgs) {
    println!("Running in interactive mode");

    let config = Config::builder().auto_add_history(true).build();
    let mut editor = Editor::<TilesHinter, DefaultHistory>::with_config(config).unwrap();
    editor.set_helper(Some(TilesHinter));
    let mut g_reply: String = "".to_owned();
    loop {
        let readline = editor.readline(">>> ");
        let input = match readline {
            Ok(line) => line.trim().to_string(),
            Err(_) => {
                // User pressed Ctrl+C or Ctrl+D
                println!("Exiting interactive mode");
                if !cfg!(debug_assertions) {
                    let _res = mlx_runtime.stop_server_daemon().await;
                }
                break;
            }
        };

        match handle_slash_command(&input, modelname) {
            SlashCommand::Continue => continue,
            SlashCommand::Exit => {
                println!("Exiting interactive mode");
                if !cfg!(debug_assertions) {
                    let _res = mlx_runtime.stop_server_daemon().await;
                }
                break;
            }
            SlashCommand::NotACommand => {}
        }

        if input.is_empty() {
            continue;
        }
        let mut remaining_count = run_args.relay_count;
        let mut python_code: String = "".to_owned();
        loop {
            if remaining_count > 0 {
                let chat_start = remaining_count == run_args.relay_count;
                if let Ok(response) =
                    chat(&input, modelname, chat_start, &python_code, &g_reply).await
                {
                    if response.reply.is_empty() {
                        if !response.code.is_empty() {
                            python_code = response.code;
                        }
                        remaining_count -= 1;
                    } else {
                        g_reply = response.reply.clone();
                        // Note: Streaming already displays the content, no need to print again
                        break;
                    }
                } else {
                    println!("\nFailed to respond");
                    break;
                }
            } else {
                // if out of relay count, then clear the global_reply and ready for next fresh prompt
                g_reply.clear();
                break;
            }
        }
        if g_reply.is_empty() {
            println!("\nNo reply, try another prompt");
        }
    }
}

pub async fn ping() -> Result<(), String> {
    let client = Client::new();
    let res = client.get("http://127.0.0.1:6969/ping").send().await;

    match res {
        Err(_) => Err(String::from("Server is down")),
        _ => Ok(()),
    }
}

async fn load_model(
    model_name: &str,
    memory_path: &str,
    system_prompt: Option<&str>,
) -> Result<()> {
    let client = Client::new();
    let mut body = json!({
        "model": model_name,
        "memory_path": memory_path
    });

    // Add system_prompt to request if provided
    if let Some(prompt) = system_prompt {
        body["system_prompt"] = serde_json::Value::String(prompt.to_string());
    }

    let res = client
        .post("http://127.0.0.1:6969/start")
        .json(&body)
        .send()
        .await?;
    match res.status() {
        StatusCode::OK => Ok(()),
        StatusCode::NOT_FOUND => {
            println!("Downloading {}\n", model_name);
            match pull_model(model_name).await {
                Ok(_) => {
                    println!("\nDownloading completed \n");
                    Ok(())
                }
                Err(err) => Err(anyhow::anyhow!(format!("Download failed due to {:?}", err))),
            }
        }
        _ => Err(anyhow::anyhow!(format!(
            "Failed to load model {} due to {:?}",
            model_name, res
        ))),
    }
}

async fn chat(
    input: &str,
    model_name: &str,
    chat_start: bool,
    python_code: &str,
    g_reply: &str,
) -> Result<ChatResponse, String> {
    let client = Client::new();

    let body = json!({
        "model": model_name,
        "chat_start": chat_start,
        "stream": true,
        "python_code": python_code,
        "messages": [{"role": "assistant", "content": g_reply}, {"role": "user", "content": input}]
    });
    let res = client
        .post("http://127.0.0.1:6969/v1/chat/completions")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Failed to send chat request: {}", e))?;

    let mut stream = res.bytes_stream();
    let mut accumulated = String::new();
    let mut current_channel = "none";
    let mut marker_accumulated = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        let s = String::from_utf8_lossy(&chunk);
        for line in s.lines() {
            if !line.starts_with("data: ") {
                continue;
            }

            let data = line.trim_start_matches("data: ");

            if data == "[DONE]" {
                println!(); // Extra newline at end
                return Ok(convert_to_chat_response(&accumulated));
            }

            // Parse JSON
            let v: Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };

            if let Some(delta) = v["choices"][0]["delta"]["content"].as_str() {
                accumulated.push_str(delta);
                marker_accumulated.push_str(delta);

                // Handle markers in the stream
                let mut suppress_token = false;

                if marker_accumulated.contains("<|channel|>analysis<|message|>") {
                    println!("\n{}", "**[Reasoning]**".bold());
                    current_channel = "analysis";
                    marker_accumulated.clear();
                    suppress_token = true;
                } else if marker_accumulated.contains("<|channel|>final<|message|>") {
                    println!("\n{}", "---".dimmed());
                    println!("{}", "**[Answer]**".bold());
                    current_channel = "final";
                    marker_accumulated.clear();
                    suppress_token = true;
                } else if marker_accumulated.contains("<|channel|>code<|message|>") {
                    println!("\n{}", "**[Executing Code]**".bold());
                    current_channel = "code";
                    marker_accumulated.clear();
                    suppress_token = true;
                } else if marker_accumulated.contains("<|end|>")
                    || marker_accumulated.contains("<|start|>")
                {
                    current_channel = "none";
                    marker_accumulated.clear();
                    suppress_token = true;
                }

                // Extra check for leftover marker parts or legacy markers
                if delta.contains("<|message|>")
                    || delta.contains("<|end|>")
                    || delta.contains("<|channel|>")
                {
                    suppress_token = true;
                }

                // Output content based on channel
                if !suppress_token {
                    if current_channel == "analysis" {
                        print!("{}", delta.dimmed());
                        io::stdout().flush().ok();
                    } else if current_channel == "final" {
                        print!("{}", delta);
                        io::stdout().flush().ok();
                    } else if current_channel == "code" {
                        // We don't necessarily want to stream raw code here if it's messy,
                        // but it's good for feedback.
                        print!("{}", delta.cyan());
                        io::stdout().flush().ok();
                    } else {
                        // Fallback for legacy format (mem-agent) - print dimmed
                        print!("{}", delta.dimmed());
                        io::stdout().flush().ok();
                    }
                }
            }
        }
    }

    if !accumulated.is_empty() {
        Ok(convert_to_chat_response(&accumulated))
    } else {
        Err(String::from("request failed"))
    }
}

fn convert_to_chat_response(content: &str) -> ChatResponse {
    ChatResponse {
        analysis: extract_analysis(content),
        reply: extract_reply(content),
        code: extract_python(content),
    }
}

fn extract_analysis(content: &str) -> String {
    // Try gpt-oss format first: <|channel|>analysis<|message|>...
    if let Some(analysis) = extract_channel_content(content, "analysis") {
        return analysis;
    }

    // Fallback to legacy format: <think>...</think>
    if content.contains("<think>") && content.contains("</think>") {
        let list_a = content.split("<think>").collect::<Vec<&str>>();
        let list_b = list_a[1].split("</think>").collect::<Vec<&str>>();
        list_b[0].to_owned()
    } else {
        "".to_owned()
    }
}

/// Extract content from a gpt-oss channel marker
/// Format: <|channel|>channel_name<|message|>content<|end|>
fn extract_channel_content(content: &str, channel: &str) -> Option<String> {
    let channel_marker = format!("<|channel|>{}<|message|>", channel);
    if let Some(start_idx) = content.find(&channel_marker) {
        let content_start = start_idx + channel_marker.len();
        let remaining = &content[content_start..];

        // Look for end marker or next channel marker
        let end_idx = remaining
            .find("<|end|>")
            .or_else(|| remaining.find("<|channel|>"))
            .unwrap_or(remaining.len());

        Some(remaining[..end_idx].trim().to_owned())
    } else {
        None
    }
}

fn extract_reply(content: &str) -> String {
    // Try gpt-oss format first: <|channel|>final<|message|>...
    if let Some(reply) = extract_channel_content(content, "final") {
        return reply;
    }

    // Fallback to legacy format: <reply>...</reply>
    if content.contains("<reply>") && content.contains("</reply>") {
        let list_a = content.split("<reply>").collect::<Vec<&str>>();
        let list_b = list_a[1].split("</reply>").collect::<Vec<&str>>();
        list_b[0].to_owned()
    } else {
        "".to_owned()
    }
}

fn extract_python(content: &str) -> String {
    // Try gpt-oss format first: <|channel|>code<|message|>...
    if let Some(code) = extract_channel_content(content, "code") {
        return code;
    }

    // Fallback to legacy format: <python>...</python>
    if content.contains("<python>") && content.contains("</python>") {
        let list_a = content.split("<python>").collect::<Vec<&str>>();
        let list_b = list_a[1].split("</python>").collect::<Vec<&str>>();
        list_b[0].to_owned()
    } else {
        "".to_owned()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_channel_content() {
        let content = "<|channel|>analysis<|message|>This is a test analysis.<|channel|>code<|message|>print('hello')<|channel|>final<|message|>Final answer.";

        assert_eq!(
            extract_channel_content(content, "analysis").unwrap(),
            "This is a test analysis."
        );
        assert_eq!(
            extract_channel_content(content, "code").unwrap(),
            "print('hello')"
        );
        assert_eq!(
            extract_channel_content(content, "final").unwrap(),
            "Final answer."
        );
        assert_eq!(extract_channel_content(content, "nonexistent"), None);
    }

    #[test]
    fn test_extract_reply() {
        // Test gpt-oss format
        let gpt_oss_content = "<|channel|>final<|message|>Hello world";
        assert_eq!(extract_reply(gpt_oss_content), "Hello world");

        // Test legacy format
        let legacy_content = "<reply>Legacy reply</reply>";
        assert_eq!(extract_reply(legacy_content), "Legacy reply");

        // Test empty
        assert_eq!(extract_reply("no markers"), "");
    }

    #[test]
    fn test_extract_python() {
        // Test gpt-oss format
        let gpt_oss_content = "<|channel|>code<|message|>x = 1";
        assert_eq!(extract_python(gpt_oss_content), "x = 1");

        // Test legacy format
        let legacy_content = "<python>y = 2</python>";
        assert_eq!(extract_python(legacy_content), "y = 2");

        // Test empty
        assert_eq!(extract_python("no markers"), "");
    }
}
