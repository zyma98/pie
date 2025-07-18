use anyhow::{Context, Result, anyhow};
use clap::{Args, Parser, Subcommand};
use comfy_table::{Table, presets::UTF8_FULL};
use daemonize::Daemonize;
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use pie::{
    Config as EngineConfig,
    auth::{create_jwt, init_secret},
    client::{self, Client},
};
use rand::{Rng, distributions::Alphanumeric};
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use std::{
    env,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};
use sysinfo::{Pid, System};

//================================================================================//
// SECTION: CLI Command & Config Structs
//================================================================================//

#[derive(Parser, Debug)]
#[command(author, version, about = "A CLI for the PIE system")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Start(StartArgs),
    Stop,
    Status,
    #[command(subcommand)]
    Model(ModelCommands),
    Run(RunArgs),
}

#[derive(Args, Debug)]
/// Start the PIE engine.
pub struct StartArgs {
    /// Path to a custom TOML configuration file.
    #[arg(long)]
    pub config: Option<PathBuf>,
    /// Run the engine in the foreground for interactive debugging.
    #[arg(long, short)]
    pub interactive: bool,
    /// The network host to bind to.
    #[arg(long)]
    pub host: Option<String>,
    /// The network port to use.
    #[arg(long)]
    pub port: Option<u16>,
    /// Disable authentication.
    #[arg(long)]
    pub no_auth: bool,
    /// A log file to write to.
    #[arg(long)]
    pub log: Option<PathBuf>,
    /// Enable verbose console logging.
    #[arg(long, short)]
    pub verbose: bool,
}

#[derive(Args, Debug)]
/// Submit an inferlet (Wasm program) to the engine.
pub struct RunArgs {
    /// Path to the .wasm inferlet file.
    pub wasm_path: PathBuf,
    /// Run the inferlet in the background and print its instance ID.
    #[arg(long, short)]
    pub detach: bool,
}

// Other command structs (ModelCommands, etc.) remain the same
#[derive(Subcommand, Debug)]
pub enum ModelCommands {
    List,
    Add(AddModelArgs),
    Remove(RemoveModelArgs),
    Serve(ModelServeArgs),
}
#[derive(Args, Debug)]
pub struct AddModelArgs {
    pub model_name: String,
}
#[derive(Args, Debug)]
pub struct RemoveModelArgs {
    pub model_name: String,
}

#[derive(Args, Debug)]
pub struct ModelServeArgs {
    pub model_name: String,
    /// Port to serve the model on
    #[arg(long, default_value = "8080")]
    pub port: u16,
}

// Helper struct for parsing the TOML config file
#[derive(Deserialize, Debug, Default)]
struct ConfigFile {
    host: Option<String>,
    port: Option<u16>,
    enable_auth: Option<bool>,
    auth_secret: Option<String>,
    cache_dir: Option<PathBuf>,
    verbose: Option<bool>,
    log: Option<PathBuf>,
}

// Helper struct for what client commands need to know
struct ClientConfig {
    host: String,
    port: u16,
    auth_secret: String,
}

//================================================================================//
// SECTION: Main Entrypoint
//================================================================================//

fn main() -> Result<()> {
    let cli = Cli::parse();

    // The 'start' command is special: it's synchronous and might daemonize.
    if let Commands::Start(args) = cli.command {
        let config = build_engine_config(&args)?;
        if args.interactive {
            println!("🚀 Starting PIE engine in interactive mode...");
            return pie::start(config); // This blocks until Ctrl+C
        } else {
            return handle_start_daemon(config);
        }
    }

    // All other commands are async clients, so we create a Tokio runtime.
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async {
            match cli.command {
                Commands::Start(_) => unreachable!(), // Already handled
                Commands::Stop => handle_stop().await,
                Commands::Status => handle_status().await,
                Commands::Run(args) => handle_run(args).await,
                Commands::Model(cmd) => handle_model_command(cmd).await,
            }
        })
}

//================================================================================//
// SECTION: Command Handlers
//================================================================================//

/// Handles the daemonization (background) start process.
fn handle_start_daemon(config: EngineConfig) -> Result<()> {
    if is_engine_running() {
        return Err(anyhow!(
            "Engine is already running. Use 'pie-cli stop' first."
        ));
    }
    println!("🚀 Starting PIE engine in background...");

    let pie_home = get_pie_home()?;
    let logs_dir = pie_home.join("logs");
    fs::create_dir_all(&logs_dir)?;
    let stdout = File::create(logs_dir.join("engine.out"))?;
    let stderr = File::create(logs_dir.join("engine.err"))?;

    let daemonize = Daemonize::new()
        .pid_file(get_pids_path()?) // Use a dedicated PID file
        .working_directory(&pie_home)
        .stdout(stdout)
        .stderr(stderr);

    match daemonize.start() {
        Ok(_) => {
            // This code runs ONLY in the forked DAEMON process.
            // The pie::start function is blocking and handles its own runtime.
            println!(
                "PIE daemon process started with PID: {}",
                std::process::id()
            );
            if let Err(e) = pie::start(config) {
                eprintln!("PIE Engine failed: {}", e);
            }
        }
        Err(e) => return Err(anyhow!("Failed to daemonize: {}", e)),
    }

    // The PARENT process exits here after successfully forking.
    println!("✅ Engine process launched in background.");
    Ok(())
}

async fn handle_run(args: RunArgs) -> Result<()> {
    let client_config = load_client_config()?;
    let url = format!("ws://{}:{}", client_config.host, client_config.port);
    let mut client = Client::connect(&url).await?;

    init_secret(&client_config.auth_secret);

    let token = create_jwt("default", pie::auth::Role::User)?;
    client.authenticate(&token).await?;
    //println!("🔐 Authenticated successfully.");

    let wasm_blob = fs::read(&args.wasm_path).context("Failed to read Wasm file")?;
    let hash = client::hash_program(&wasm_blob);
    println!("Program hash: {}", hash);

    if !client.program_exists(&hash).await? {
        //println!("Program not found on engine, uploading...");
        client.upload_program(&wasm_blob).await?;
        println!("✅ Program upload successful.");
    }

    println!("🚀 Launching instance...");
    let mut instance = client.launch_instance(&hash).await?;
    println!("✅ Instance launched with ID: {}", instance.id());

    if args.detach {
        return Ok(());
    }

    println!("Attaching to instance output... (Ctrl+C to exit)");
    while let Ok((event, message)) = instance.recv().await {
        if event == "terminated" {
            println!("Instance terminated. Reason: {}", message);
            break;
        } else {
            println!("[{}] {}", event, message);
        }
    }
    Ok(())
}

async fn handle_stop() -> Result<()> {
    println!("🔌 Stopping PIE services...");
    if let Ok(pid) = fs::read_to_string(get_pids_path()?) {
        if let Ok(pid_val) = pid.trim().parse::<u32>() {
            if let Some(process) = System::new_all().process(Pid::from_u32(pid_val)) {
                println!("- Stopping engine (PID: {})", pid_val);
                process.kill();
            }
        }
    }
    fs::remove_file(get_pids_path()?).ok();
    println!("✅ Engine stopped.");
    Ok(())
}

async fn handle_status() -> Result<()> {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_header(vec!["Service", "Status", "Details"]);

    if is_engine_running() {
        let pid = fs::read_to_string(get_pids_path()?).unwrap_or_default();
        table.add_row(vec![
            "PIE Engine",
            &format!("Running (PID: {})", pid.trim()),
            "",
        ]);
    } else {
        table.add_row(vec!["PIE Engine", "Stopped", ""]);
    }
    println!("{table}");
    Ok(())
}

async fn handle_model_command(command: ModelCommands) -> Result<()> {
    match command {
        ModelCommands::List => {
            println!("📚 Available local models:");
            let models_dir = get_pie_home()?.join("models");
            if !models_dir.exists() {
                println!("  No models found.");
                return Ok(());
            }
            for entry in fs::read_dir(models_dir)? {
                let entry = entry?;
                if entry.file_type()?.is_dir() {
                    println!("  - {}", entry.file_name().to_string_lossy());
                }
            }
        }
        ModelCommands::Add(args) => {
            println!("➕ Adding model: {}", args.model_name);

            let models_root = get_pie_home()?.join("models");
            let model_files_dir = models_root.join(&args.model_name);
            let metadata_path = models_root.join(format!("{}.toml", args.model_name));

            // Check if the model already exists and ask for confirmation to overwrite.
            if metadata_path.exists() || model_files_dir.exists() {
                print!(
                    "⚠️ Model '{}' already exists. Overwrite? [y/N] ",
                    args.model_name
                );
                std::io::stdout()
                    .flush()
                    .context("Failed to flush stdout")?;

                let mut confirmation = String::new();
                std::io::stdin()
                    .read_line(&mut confirmation)
                    .context("Failed to read user input")?;

                if confirmation.trim().to_lowercase() != "y" {
                    println!("Aborted by user.");
                    return Ok(());
                }
            }

            // --- Set up directories and save files with the new structure ---
            fs::create_dir_all(&model_files_dir)?;

            println!("Parameters will be stored at {:?}", model_files_dir);

            let model_index_base =
                "https://raw.githubusercontent.com/pie-project/model-index/refs/heads/main";
            let metadata_url = format!("{}/{}.toml", model_index_base, args.model_name);

            // --- Download metadata with specific 404 error handling ---
            let metadata_raw =
                match download_file_with_progress(&metadata_url, "Downloading metadata...").await {
                    Ok(data) => data,
                    Err(e) => {
                        if let Some(req_err) = e.downcast_ref::<reqwest::Error>() {
                            if req_err.status() == Some(reqwest::StatusCode::NOT_FOUND) {
                                anyhow::bail!(
                                    "Model '{}' not found in the official index.",
                                    args.model_name
                                );
                            }
                        }
                        return Err(e.context("Failed to download model metadata"));
                    }
                };

            let metadata_str = String::from_utf8(metadata_raw)
                .context("Failed to parse model metadata as UTF-8")?;
            let metadata: toml::Value = toml::from_str(&metadata_str)?;

            // Store the .toml metadata file
            fs::write(&metadata_path, &metadata_str)?;

            // Download all source files listed in the metadata
            if let Some(source) = metadata.get("source").and_then(|s| s.as_table()) {
                for (name, url_val) in source {
                    if let Some(url) = url_val.as_str() {
                        let file_data =
                            download_file_with_progress(url, &format!("Downloading {}...", name))
                                .await?;
                        fs::write(model_files_dir.join(name), file_data)?;
                    }
                }
            }
            println!("✅ Model '{}' added successfully!", args.model_name);
        }
        ModelCommands::Remove(args) => {
            println!("🗑️ Removing model: {}", args.model_name);

            // Define the base path for all models
            let models_root = get_pie_home()?.join("models");
            // Path to the subdirectory containing the model's files
            let model_files_dir = models_root.join(&args.model_name);
            // Path to the model's metadata .toml file
            let metadata_path = models_root.join(format!("{}.toml", args.model_name));

            let mut was_removed = false;

            // Attempt to remove the model's file directory if it exists
            if model_files_dir.exists() {
                fs::remove_dir_all(&model_files_dir)?;
                was_removed = true;
            }

            // Attempt to remove the model's metadata file if it exists
            if metadata_path.exists() {
                fs::remove_file(&metadata_path)?;
                was_removed = true;
            }

            if was_removed {
                println!("✅ Model '{}' removed.", args.model_name);
            } else {
                // If neither the directory nor the file was found, return an error
                anyhow::bail!("Model '{}' not found locally.", args.model_name);
            }
        }
        ModelCommands::Serve(args) => return Err(anyhow!("`model serve` is not yet implemented.")),
    }
    Ok(())
}

//================================================================================//
// SECTION: Helpers
//================================================================================//

/// Merges config from file and CLI args to create the final EngineConfig.
fn build_engine_config(args: &StartArgs) -> Result<EngineConfig> {
    let file_config = if let Some(path) = &args.config {
        let content = fs::read_to_string(path)?;
        toml::from_str(&content)?
    } else {
        ConfigFile::default()
    };

    let enable_auth = if args.no_auth {
        false
    } else {
        file_config.enable_auth.unwrap_or(true)
    };
    let auth_secret = file_config.auth_secret.unwrap_or_else(|| {
        rand::thread_rng()
            .sample_iter(&Alphanumeric)
            .take(32)
            .map(char::from)
            .collect()
    });

    Ok(EngineConfig {
        host: args
            .host
            .clone()
            .or(file_config.host)
            .unwrap_or_else(|| "127.0.0.1".to_string()),
        port: args.port.or(file_config.port).unwrap_or(9123),
        enable_auth,
        auth_secret,
        cache_dir: file_config
            .cache_dir
            .unwrap_or_else(|| get_pie_home().unwrap().join("cache")),
        verbose: args.verbose || file_config.verbose.unwrap_or(false),
        log: args.log.clone().or(file_config.log),
    })
}

/// Loads the minimal config needed for client commands to connect to the engine.
fn load_client_config() -> Result<ClientConfig> {
    let config_path = get_pie_home()?.join("runtime/engine.toml");
    let content = fs::read_to_string(&config_path).with_context(|| {
        format!(
            "Engine config not found at {:?}. Is the engine running?",
            config_path
        )
    })?;
    let config: ConfigFile = toml::from_str(&content)?;

    Ok(ClientConfig {
        host: config.host.context("'host' missing from engine config")?,
        port: config.port.context("'port' missing from engine config")?,
        auth_secret: config
            .auth_secret
            .context("'auth_secret' missing from engine config")?,
    })
}

fn get_pie_home() -> Result<PathBuf> {
    if let Ok(path) = env::var("PIE_HOME") {
        Ok(PathBuf::from(path))
    } else {
        dirs::home_dir()
            .map(|p| p.join(".pie"))
            .ok_or_else(|| anyhow!("Failed to find home dir"))
    }
}

fn get_pids_path() -> Result<PathBuf> {
    let dir = get_pie_home()?.join("runtime");
    fs::create_dir_all(&dir)?;
    Ok(dir.join("pie.pid"))
}

fn is_engine_running() -> bool {
    if let Ok(pid_path) = get_pids_path() {
        if let Ok(pid_str) = fs::read_to_string(pid_path) {
            if let Ok(pid) = pid_str.trim().parse::<u32>() {
                return System::new_all().process(Pid::from_u32(pid)).is_some();
            }
        }
    }
    false
}

async fn download_file_with_progress(url: &str, message: &str) -> Result<Vec<u8>> {
    let client = HttpClient::new();
    let res = client.get(url).send().await?.error_for_status()?;
    let total_size = res.content_length().unwrap_or(0);

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec})")?
            .progress_chars("##-"),
    );
    pb.set_message(message.to_string());

    let mut downloaded = 0;
    let mut stream = res.bytes_stream();
    let mut content = Vec::with_capacity(total_size as usize);

    while let Some(item) = stream.next().await {
        let chunk = item?;
        downloaded += chunk.len();
        content.extend_from_slice(&chunk);
        pb.set_position(downloaded as u64);
    }

    let file_name = url
        .rsplit('/')
        .next()
        .unwrap_or("file")
        .split('?')
        .next()
        .unwrap_or("file");

    pb.finish_with_message(format!("✅ Downloaded {}", file_name));
    Ok(content)
}
