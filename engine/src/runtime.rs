use dashmap::DashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc::{Sender, UnboundedReceiver, UnboundedSender};
use uuid::Uuid;
use wasmtime::{Config, Engine, Store, component::Component, component::Linker};
use wasmtime_wasi;

use crate::instance::{Id as InstanceId, InstanceState};
use crate::{bindings, service, tokenizer};

use crate::service::{Service, ServiceError};
use thiserror::Error;
use tokio::sync::oneshot;

#[derive(Debug, Error)]
pub enum RuntimeError {
    /// Wrap general I/O errors
    #[error("I/O error occurred: {0}")]
    Io(#[from] std::io::Error),

    /// Wrap Wasmtime errors
    #[error("Wasmtime error occurred: {0}")]
    Wasmtime(#[from] wasmtime::Error),

    /// No program found for a given hash
    #[error("No such program with hash={0}")]
    MissingProgram(String),

    /// Failed to compile a WASM component from disk
    #[error("Failed to compile program at path {path:?}: {source}")]
    CompileWasm {
        path: std::path::PathBuf,
        #[source]
        source: wasmtime::Error,
    },

    /// Fallback for unexpected cases
    #[error("Runtime error: {0}")]
    Other(String),
}

#[derive(Debug)]
pub enum Command {
    ProgramExists {
        hash: String,
        event: oneshot::Sender<bool>,
    },

    UploadProgram {
        hash: String,
        raw: Vec<u8>,
        event: oneshot::Sender<Result<String, RuntimeError>>,
    },

    LaunchInstance {
        hash: String,
        event: oneshot::Sender<Result<InstanceId, RuntimeError>>,
    },

    TerminateInstance {
        instance_id: InstanceId,
    },
}

impl Command {
    pub fn dispatch(self) -> Result<(), ServiceError> {
        service::dispatch(service::SERVICE_RUNTIME, self)
    }
}

/// Holds the “global” or “runtime” data that the controller needs to manage
/// instances, compiled programs, etc.
pub struct Runtime {
    /// The Wasmtime engine (global)
    engine: Engine,

    /// Pre-compiled WASM components, keyed by BLAKE3 hex string
    programs_in_memory: DashMap<String, Component>,

    /// Paths to compiled modules on disk
    programs_in_disk: DashMap<String, std::path::PathBuf>,

    /// Running instances
    running_instances: DashMap<InstanceId, InstanceHandle>,
}

pub struct InstanceHandle {
    pub hash: String,
    //pub to_origin: Sender<ServerMessage>,
    // pub evt_from_system: Sender<String>,
    // pub evt_from_origin: Sender<String>,
    // pub evt_from_peers: Sender<(String, String)>,
    pub join_handle: tokio::task::JoinHandle<()>,
}

impl Service for Runtime {
    type Command = Command;

    async fn handle(&mut self, cmd: Self::Command) {
        match cmd {
            Command::ProgramExists { hash, event } => {
                let exists = self.programs_in_memory.contains_key(&hash)
                    || self.programs_in_disk.contains_key(&hash);
                event.send(exists).unwrap();
            }

            Command::UploadProgram { raw, event } => {
                let hash = blake3::hash(&raw).to_hex().to_string();
                if self.programs_in_memory.contains_key(&hash) {
                    event.send(Ok(hash)).unwrap();
                } else {
                    let path = std::path::Path::new("cache").join(&hash);
                    std::fs::write(&path, &raw).unwrap();
                    self.programs_in_disk.insert(hash.clone(), path);
                    event.send(Ok(hash)).unwrap();
                }
            }

            Command::StartInstance { hash, event } => {
                let (to_origin, from_origin) = tokio::sync::mpsc::channel(1000);
                let instance_id = self.start_program(&hash, to_origin).await.unwrap();
                event.send(Ok(instance_id)).unwrap();
            }

            Command::TerminateInstance { instance_id, event } => {
                let success = self.terminate_program(instance_id, "terminated by user".to_string());
                event.send(success).unwrap();
            }
        }

        todo!()
    }
}

impl Runtime {
    /// Create a new `Runtime`
    pub fn new(
        command_dispatcher: CommandDispatcher,
        report_rx: UnboundedReceiver<Report>,
    ) -> Self {
        // Configure Wasmtime engine
        let mut config = Config::default();
        config.async_support(true);
        let engine = Engine::new(&config).unwrap();

        Self {
            engine,
            programs_in_memory: DashMap::new(),
            programs_in_disk: DashMap::new(),
            running_instances: DashMap::new(),
            command_dispatcher,
        }
    }

    /// If desired, you can add your “load programs from disk” logic here.
    pub fn load_existing_programs(&self, cache_dir: &std::path::Path) -> Result<(), RuntimeError> {
        let entries = std::fs::read_dir(cache_dir)?; // Will map to RuntimeError::Io automatically
        for entry in entries {
            let entry = entry?; // same here, auto-converted to RuntimeError::Io
            if entry.file_type()?.is_file() {
                let path = entry.path();
                let data = std::fs::read(&path)?; // also auto Io
                let hash = blake3::hash(&data).to_hex().to_string();
                self.programs_in_disk.insert(hash, path);
            }
        }
        Ok(())
    }

    /// Actually start a program instance
    pub async fn start_program(
        &self,
        hash: &str,
        to_origin: Sender<ServerMessage>,
    ) -> Result<InstanceId, RuntimeError> {
        // 1) Make sure the `Component` is loaded in memory
        if self.programs_in_memory.get(hash).is_none() {
            // load from disk if possible
            if let Some(path_entry) = self.programs_in_disk.get(hash) {
                // Use a custom error variant for compile errors
                let component =
                    Component::from_file(&self.engine, path_entry.value()).map_err(|err| {
                        RuntimeError::CompileWasm {
                            path: path_entry.value().to_path_buf(),
                            source: err,
                        }
                    })?;
                self.programs_in_memory.insert(hash.to_string(), component);
            } else {
                // If not on disk either, return a custom error
                return Err(RuntimeError::MissingProgram(hash.to_string()));
            }
        }

        // 2) Now we have a compiled component
        let component = match self.programs_in_memory.get(hash) {
            Some(c) => c.clone(),
            None => {
                return Err(RuntimeError::Other(
                    "Failed to get component from memory".into(),
                ));
            }
        };

        let instance_id = Uuid::new_v4();

        // 4) Build the InstanceState
        let inst_state = InstanceState::new(instance_id, self.inst2server.clone()).await;

        // 5) Instantiate and run in a task
        let engine_clone = self.engine.clone();
        let to_origin_clone = to_origin.clone();
        let join_handle = tokio::spawn(async move {
            // Wrap everything in a closure returning a Result,
            // so we can capture errors more systematically if desired:
            let result = async {
                let mut store = Store::new(&engine_clone, inst_state);
                let mut linker = Linker::new(&engine_clone);

                // Add to linker
                wasmtime_wasi::add_to_linker_async(&mut linker)
                    .map_err(|e| RuntimeError::Other(format!("Failed to link WASI: {e}")))?;
                wasmtime_wasi_http::add_only_http_to_linker_async(&mut linker)
                    .map_err(|e| RuntimeError::Other(format!("Failed to link WASI: {e}")))?;

                bindings::add_to_linker(&mut linker)?;

                // instance_old::spi::app::l4m::add_to_linker(&mut linker, |s| s)?;
                // instance_old::spi::app::l4m_vision::add_to_linker(&mut linker, |s| s)?;
                // instance_old::spi::app::system::add_to_linker(&mut linker, |s| s)?;
                // instance_old::spi::app::ping::add_to_linker(&mut linker, |s| s)?;

                // App::add_to_linker(&mut linker, |s| s)
                //     .map_err(|e| RuntimeError::Other(format!("Error adding to linker: {e}")))?;

                let instance = linker
                    .instantiate_async(&mut store, &component)
                    .await
                    .map_err(|e| RuntimeError::Other(format!("Instantiation error: {e}")))?;

                // Attempt to call “run”
                let run_export = instance.get_export(&mut store, None, "spi:app/run");
                let run_iface = run_export
                    .ok_or_else(|| RuntimeError::Other("No spi:app/run in the module".into()))?;

                let run_func_export = instance.get_export(&mut store, Some(&run_iface), "run");
                let run_func_export = run_func_export
                    .ok_or_else(|| RuntimeError::Other("No 'run' function found".into()))?;

                let run_func = instance
                    .get_typed_func::<(), (Result<(), String>,)>(&mut store, &run_func_export)
                    .map_err(|e| {
                        RuntimeError::Other(format!("Failed to get 'run' function: {e}"))
                    })?;

                match run_func.call_async(&mut store, ()).await {
                    Ok((Ok(()),)) => {
                        println!("Instance {instance_id} finished normally");
                        Ok(())
                    }
                    Ok((Err(runtime_err),)) => {
                        eprintln!("Instance {instance_id} returned an error");
                        Err(RuntimeError::Other(runtime_err))
                    }
                    Err(call_err) => {
                        eprintln!("Instance {instance_id} call error: {call_err}");
                        Err(RuntimeError::Other(format!("Call error: {call_err}")))
                    }
                }
            }
            .await;

            if let Err(err) = result {
                eprintln!("Instance {instance_id} error: {err}");
                to_origin_clone
                    .send(ServerMessage::ProgramTerminated {
                        instance_id: String::from(instance_id),
                        reason: format!("{err}"),
                    })
                    .await
                    .ok();
            } else {
                to_origin_clone
                    .send(ServerMessage::ProgramTerminated {
                        instance_id: String::from(instance_id),
                        reason: "the instance normally finished".to_string(),
                    })
                    .await
                    .ok();
            }
        });

        // 6) Record in the “running_instances” so we can manage it later
        let instance_handle = InstanceHandle {
            hash: hash.to_string(),
            to_origin,
            join_handle,
        };
        self.running_instances.insert(instance_id, instance_handle);

        Ok(instance_id)
    }

    /// Terminate (abort) a running instance
    pub fn terminate_program(&self, instance_id: InstanceId, reason: String) -> bool {
        if let Some((_, handle)) = self.running_instances.remove(&instance_id) {
            // TODO
            handle.join_handle.abort();
            true
        } else {
            false
        }
    }
}

// Error reporter

enum Report {
    Error(InstanceId, String),
    Warning(InstanceId, String),
    LogTrace(InstanceId, String, String, Instant),
}

#[derive(Clone, Debug)]
pub struct ExceptionDispatcher {
    tx: UnboundedSender<Report>,
}

impl ExceptionDispatcher {
    pub fn new(tx: UnboundedSender<Report>) -> Self {
        Self { tx }
    }

    pub fn error(&self, inst: InstanceId, msg: String) {
        self.tx.send(Report::Error(inst, msg)).unwrap()
    }

    pub fn warning(&self, inst: InstanceId, msg: String) {
        self.tx.send(Report::Warning(inst, msg)).unwrap()
    }

    pub fn log_trace(&self, inst: InstanceId, subject: String, tag: String) {
        self.tx
            .send(Report::LogTrace(inst, subject, tag, Instant::now()))
            .unwrap()
    }
}
