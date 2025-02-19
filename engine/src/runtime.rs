use anyhow::Context;
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::mpsc::Sender;
use uuid::Uuid;
use wasmtime::{component::Component, component::Linker, Engine, Store};
use wasmtime_wasi;

use crate::instance::{App, Command, Id as InstanceId, InstanceState};
use crate::server::ServerMessage;
use crate::tokenizer;

/// Holds the “global” or “runtime” data that the controller needs to manage
/// instances, compiled programs, etc.
pub struct Runtime {
    /// The Wasmtime engine (global)
    pub engine: Engine,

    /// Pre-compiled WASM components, keyed by BLAKE3 hex string
    pub programs_in_memory: DashMap<String, Component>,

    /// Paths to compiled modules on disk
    pub programs_in_disk: DashMap<String, std::path::PathBuf>,

    /// Running instances
    pub running_instances: DashMap<InstanceId, InstanceHandle>,

    /// The channel for (Instance -> controller) messages
    pub inst2server: Sender<(InstanceId, Command)>,

    /// For demonstration: a tokenizer or other utility
    pub tokenizer: Arc<tokenizer::BytePairEncoder>,
}

pub struct InstanceHandle {
    pub hash: String,
    pub to_origin: Sender<ServerMessage>,
    pub evt_from_origin: Sender<String>,
    pub evt_from_peers: Sender<(String, String)>,
    pub join_handle: tokio::task::JoinHandle<()>,
}

impl Runtime {
    /// Create a new `RuntimeState` plus a minimal `Controller` around it
    pub fn new(engine: Engine, inst2server: Sender<(InstanceId, Command)>) -> Runtime {
        // Here you might also load your tokenizer file
        // or do other initialization.
        let tokenizer = Arc::new(
            tokenizer::llama3_tokenizer(super::TOKENIZER_MODEL)
                .expect("Tokenizer load failed"),
        );

        Runtime {
            engine,
            programs_in_memory: DashMap::new(),
            programs_in_disk: DashMap::new(),
            running_instances: DashMap::new(),
            inst2server,
            tokenizer,
        }
    }

    /// If desired, you can add your “load programs from disk” logic here.
    pub fn load_existing_programs(&self, cache_dir: &std::path::Path) -> anyhow::Result<()> {
        let entries = std::fs::read_dir(cache_dir)?;
        for entry in entries {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                let path = entry.path();
                let data = std::fs::read(&path)?;
                let hash = blake3::hash(&data).to_hex().to_string();
                self.programs_in_disk.insert(hash, path);
            }
        }
        Ok(())
    }

    /// Actually start a program instance
    /// (moved out of the WebSocket side so that the server just calls into here).
    pub async fn start_program(
        &self,
        hash: &str,
        to_origin: Sender<ServerMessage>,
    ) -> anyhow::Result<InstanceId> {
        // 1) Make sure the `Component` is loaded in memory
        if self.programs_in_memory.get(hash).is_none() {
            // load from disk if possible
            if let Some(path_entry) = self.programs_in_disk.get(hash) {
                let component = Component::from_file(&self.engine, path_entry.value())
                    .context("Failed to compile program")?;
                self.programs_in_memory.insert(hash.to_string(), component);
            } else {
                anyhow::bail!("No such program with hash={}", hash);
            }
        }

        // 2) Now we have a compiled component
        let component = match self.programs_in_memory.get(hash) {
            Some(c) => c.clone(),
            None => anyhow::bail!("Failed to load component from memory"),
        };

        let instance_id = Uuid::new_v4();

        // 3) Create channels for event-sending
        let (server2inst_tx, server2inst_rx) = tokio::sync::mpsc::channel::<String>(32);
        let (peer_tx, peer_rx) = tokio::sync::mpsc::channel::<(String, String)>(32);

        // 4) Build the InstanceState
        let inst_state = InstanceState::new(
            instance_id,
            self.inst2server.clone(),
            server2inst_rx,
            peer_rx,
            crate::instance::InstanceUtils {
                tokenizer: self.tokenizer.clone(),
            },
        );

        // 5) Instantiate and run in a task
        let engine_clone = self.engine.clone();
        let join_handle = tokio::spawn(async move {
            let mut store = Store::new(&engine_clone, inst_state);
            let mut linker = Linker::new(&engine_clone);

            if let Err(e) = App::add_to_linker(&mut linker, |s| s) {
                eprintln!("Error adding to linker: {}", e);
                return;
            }
            if let Err(e) = wasmtime_wasi::add_to_linker_async(&mut linker) {
                eprintln!("Failed to link WASI: {}", e);
                return;
            }

            let instance = match linker.instantiate_async(&mut store, &component).await {
                Ok(i) => i,
                Err(e) => {
                    eprintln!("Instantiation error: {}", e);
                    return;
                }
            };

            // Attempt to call “run”
            let run_export = instance.get_export(&mut store, None, "spi:app/run");
            if run_export.is_none() {
                eprintln!("No spi:app/run in the module");
                return;
            }

            let run_iface = run_export.unwrap();
            let run_func_export = instance.get_export(&mut store, Some(&run_iface), "run");
            if run_func_export.is_none() {
                eprintln!("No run function found");
                return;
            }

            let run_func = match instance
                .get_typed_func::<(), (Result<(), ()>,)>(&mut store, &run_func_export.unwrap())
            {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("Failed to get run function: {}", e);
                    return;
                }
            };

            match run_func.call_async(&mut store, ()).await {
                Ok((Ok(()),)) => {
                    println!("Instance {} finished normally", instance_id);
                }
                Ok((Err(()),)) => {
                    eprintln!("Instance {} returned an error", instance_id);
                }
                Err(call_err) => {
                    eprintln!("Instance {} call error: {}", instance_id, call_err);
                }
            }
        });

        // 6) Record in the “running_instances” so we can manage it later (e.g. terminate)
        let instance_handle = InstanceHandle {
            hash: hash.to_string(),
            to_origin,
            evt_from_origin: server2inst_tx,
            evt_from_peers: peer_tx,
            join_handle,
        };
        self.running_instances.insert(instance_id, instance_handle);

        Ok(instance_id)
    }

    /// Terminate (abort) a running instance
    pub fn terminate_program(&self, instance_id: InstanceId) -> bool {
        if let Some((_, handle)) = self.running_instances.remove(&instance_id) {
            handle.join_handle.abort();
            true
        } else {
            false
        }
    }
}
