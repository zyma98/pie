mod service;

pub mod auth;
mod backend;
mod batching;
mod bindings;
pub mod client;
mod instance;
mod l4m;
mod messaging;
mod object;
mod ping;
mod runtime;
mod server;
mod tokenizer;
mod utils;

//
use anyhow::Context;
use std::path::{Path, PathBuf};

use crate::auth::{create_jwt, init_secret};
use crate::messaging::{PubSub, PushPull};
use crate::ping::Ping;
use crate::runtime::Runtime;
use crate::server::Server;
use crate::service::install_service;
use std::fs;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

#[derive(Debug)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub enable_auth: bool,
    pub auth_secret: String,
    pub cache_dir: PathBuf,
    pub verbose: bool,
    pub log: Option<PathBuf>,
}

pub fn start(config: Config) -> anyhow::Result<()> {
    // 3. Setup logging
    // This guard must be kept alive for the duration of the program.
    // If it's dropped, the background logging thread will shut down.
    let _guard;

    let stdout_filter = if config.verbose {
        EnvFilter::new("info")
    } else {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"))
    };

    let stdout_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stdout)
        .with_filter(stdout_filter);

    let file_layer = if let Some(log_path) = &config.log {
        // Ensure the parent directory exists.
        if let Some(parent) = log_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create log directory at {:?}", parent))?;
        }

        // Use tracing_appender to create a non-blocking file writer.
        // We use `rolling::never` to create a single, non-rotating log file.
        let file_appender = tracing_appender::rolling::never(
            log_path.parent().unwrap_or_else(|| Path::new(".")),
            log_path.file_name().unwrap_or_else(|| "pie.log".as_ref()),
        );

        // The 'non_blocking' function spawns a dedicated thread for writing.
        let (non_blocking_writer, guard) = tracing_appender::non_blocking(file_appender);
        _guard = guard; // Store the guard to keep the thread alive.

        let layer = tracing_subscriber::fmt::layer()
            .with_writer(non_blocking_writer) // Use the non-blocking writer
            .with_ansi(false)
            .with_filter(EnvFilter::new("trace")); // Log everything to the file
        Some(layer)
    } else {
        None
    };

    tracing_subscriber::registry()
        .with(stdout_layer)
        .with(file_layer)
        .init();

    tracing::debug!("{:#?}", config);

    // 4. Build the Tokio runtime and start the runtime.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    rt.block_on(main(config))
}

async fn main(config: Config) -> anyhow::Result<()> {
    // Ensure the cache directory exists
    fs::create_dir_all(&config.cache_dir).map_err(|e| {
        tracing::error!(error = %e,"Setup failure: could not create cache dir");
        e
    })?;

    if config.enable_auth {
        tracing::info!("Authentication is enabled.");
        init_secret(&config.auth_secret);
        let token = create_jwt("default", auth::Role::User)?;
        tracing::info!("Use this token to authenticate: {}", token);
    } else {
        tracing::info!("Authentication is disabled.");
    }

    // Set up core services
    let runtime = Runtime::new(&config.cache_dir);
    runtime.load_existing_programs()?;

    let server_url = format!("{}:{}", config.host, config.port);

    let server = Server::new(&server_url, config.enable_auth);
    let messaging_inst2inst = PubSub::new();
    let messaging_user2inst = PushPull::new();

    // Set up test services
    //let dummy_l4m_backend = backend::SimulatedBackend::new(l4m::Simulator::new()).await;
    let dummy_ping_backend = backend::SimulatedBackend::new(ping::Simulator::new()).await;
    let ping = Ping::new(dummy_ping_backend.clone()).await;

    install_service("runtime", runtime);
    install_service("server", server);
    install_service("messaging-inst2inst", messaging_inst2inst);
    install_service("messaging-user2inst", messaging_user2inst);
    install_service("ping", ping);

    //l4m::attach_new_backend("model-test", dummy_l4m_backend).await;

    tracing::info!("Runtime started successfully.");

    tokio::signal::ctrl_c().await?;
    tracing::info!("Ctrl+C received, shutting down.");

    Ok(())
}
