use super::instance::{InstanceId, InstanceState, OutputDelivery, OutputDeliveryCtrl};
use super::service::{CommandDispatcher, Service};
use super::{api, server};
use crate::model;
use crate::model::request::QueryResponse;
use crate::service::ServiceCommand;
use dashmap::DashMap;
use hyper::server::conn::http1;
use pie_client::message::{self, LibraryInfo};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, OnceLock};
use thiserror::Error;
use tokio::sync::oneshot;
use uuid::Uuid;
use wasmtime::component::{Func, Resource, ResourceAny, ResourceType, Val};
use wasmtime::component::types::ComponentItem;
use wasmtime::{Config, Engine, Store, component::Component, component::Linker};
use wasmtime_wasi_http::WasiHttpView;
use wasmtime_wasi_http::bindings::exports::wasi::http::incoming_handler::{
    IncomingRequest, ResponseOutparam,
};
use wasmtime_wasi_http::bindings::http::types::Scheme;
use wasmtime_wasi_http::body::HyperOutgoingBody;
use wasmtime_wasi_http::io::TokioIo;

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The sender of the command channel, which is used to send commands to the
/// handler task.
static COMMAND_DISPATCHER: OnceLock<CommandDispatcher<Command>> = OnceLock::new();

/// Marker type for stub resources used during library validation.
/// These resources are registered in the validation linker but never actually used.
struct StubResource;

/// Starts the runtime service. A daemon task will be spawned to handle the
/// commands dispatched from other services.
pub fn start_service<P: AsRef<std::path::Path>>(cache_dir: P) {
    let runtime = Runtime::new(cache_dir);

    // Loading existing programs should not fail.
    runtime.load_existing_programs().unwrap();
    runtime.start(&COMMAND_DISPATCHER);
}

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

    /// Library already exists
    #[error("Library '{0}' already exists")]
    LibraryExists(String),

    /// Library not found
    #[error("Library '{0}' not found")]
    LibraryNotFound(String),

    /// Library dependency not found
    #[error("Library dependency '{0}' not found")]
    MissingDependency(String),

    /// Library imports cannot be satisfied by its dependencies
    #[error("Library imports not satisfiable: {0}")]
    UnsatisfiableImports(String),

    /// Fallback for unexpected cases
    #[error("Runtime error: {0}")]
    Other(String),
}

pub enum Command {
    GetVersion {
        event: oneshot::Sender<String>,
    },

    ProgramExists {
        hash: String,
        event: oneshot::Sender<bool>,
    },

    UploadProgram {
        hash: String,
        raw: Vec<u8>,
        /// Names of libraries this program depends on
        dependencies: Vec<String>,
        event: oneshot::Sender<Result<String, RuntimeError>>,
    },

    LaunchInstance {
        username: String,
        program_hash: String,
        /// Names of libraries this program depends on (overrides upload-time dependencies if non-empty)
        dependencies: Vec<String>,
        arguments: Vec<String>,
        detached: bool,
        event: oneshot::Sender<Result<InstanceId, RuntimeError>>,
    },

    AttachInstance {
        inst_id: InstanceId,
        event: oneshot::Sender<AttachInstanceResult>,
    },

    DetachInstance {
        inst_id: InstanceId,
    },

    AllowOutput {
        inst_id: InstanceId,
    },

    LaunchServerInstance {
        username: String,
        program_hash: String,
        port: u32,
        arguments: Vec<String>,
        event: oneshot::Sender<Result<(), RuntimeError>>,
    },

    TerminateInstance {
        inst_id: InstanceId,
        notification_to_client: Option<TerminationCause>,
    },

    FinishInstance {
        inst_id: InstanceId,
        cause: TerminationCause,
    },

    SetOutputDelivery {
        inst_id: InstanceId,
        mode: OutputDelivery,
    },

    DebugQuery {
        query: String,
        event: oneshot::Sender<QueryResponse>,
    },

    ListInstances {
        username: String,
        event: oneshot::Sender<Vec<message::InstanceInfo>>,
    },

    /// Upload a library component and register its exports.
    /// The library can specify dependencies on other libraries.
    UploadLibrary {
        /// Unique name/identifier for the library
        name: String,
        /// Raw WASM component bytes
        raw: Vec<u8>,
        /// Names of libraries this library depends on
        dependencies: Vec<String>,
        event: oneshot::Sender<Result<String, RuntimeError>>,
    },

    /// List all loaded libraries
    ListLibraries {
        event: oneshot::Sender<Vec<LibraryInfo>>,
    },
}

impl ServiceCommand for Command {
    const DISPATCHER: &'static OnceLock<CommandDispatcher<Self>> = &COMMAND_DISPATCHER;
}

/// Holds the "global" or "runtime" data that the controller needs to manage
/// instances, compiled programs, etc.

/// Information about a loaded program
struct LoadedProgram {
    /// The compiled component
    component: Component,
    /// Names of libraries this program depends on
    dependencies: Vec<String>,
}

/// Information about a loaded library
struct LoadedLibrary {
    /// The library name
    name: String,
    /// The compiled component
    component: Component,
    /// Names of libraries this library depends on
    dependencies: Vec<String>,
}

struct Runtime {
    /// The Wasmtime engine (global)
    engine: Engine,
    /// Base linker with host-defined interfaces (WASI, HTTP, Pie API)
    base_linker: Arc<Linker<InstanceState>>,

    cache_dir: std::path::PathBuf,

    /// Pre-compiled WASM programs, keyed by BLAKE3 hex string
    programs_in_memory: DashMap<String, LoadedProgram>,

    /// Paths to compiled modules on disk (just the raw bytes, no metadata)
    programs_in_disk: DashMap<String, std::path::PathBuf>,

    /// Running instances
    running_instances: DashMap<InstanceId, InstanceHandle>,

    /// Finished instances
    finished_instances: DashMap<InstanceId, InstanceHandle>,

    /// Running server instances
    running_server_instances: DashMap<InstanceId, InstanceHandle>,

    /// Loaded libraries, keyed by library name
    loaded_libraries: HashMap<String, LoadedLibrary>,

    /// Order in which libraries were loaded (for displaying)
    library_load_order: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum TerminationCause {
    Normal(String),
    Signal,
    Exception(String),
    OutOfResources(String),
}

#[derive(Debug, Clone)]
pub enum InstanceRunningState {
    /// The instance is running and a client is attached to it.
    /// The output will be streamed to the client.
    ///
    /// Note that this enum does not directly affect the output streaming behavior.
    /// The output streaming behavior is determined by the output delivery mode.
    /// The `Attached` state merely prevents other clients from attaching to the same
    /// instance. Once an instance is set to `Attached`, the output delivery mode must
    /// be set to `Streamed` via the `SetOutputDelivery` command.
    Attached,
    /// The instance is running and not attached to a client.
    /// The output will be buffered.
    ///
    /// Note that this enum does not directly affect the output streaming behavior.
    /// The output streaming behavior is determined by the output delivery mode.
    /// The `Detached` state merely indicates that the instance is not attached to a
    /// client and other clients can attach to it. Before setting the instance to
    /// `Detached`, the output delivery mode must be set to `Buffered` via the
    /// `SetOutputDelivery` command, so to prevent the output from being lost.
    Detached,
    /// The instance has finished execution.
    /// The output is buffered and waiting to be streamed to the client.
    Finished(TerminationCause),
}

impl From<InstanceRunningState> for message::InstanceStatus {
    fn from(state: InstanceRunningState) -> Self {
        match state {
            InstanceRunningState::Attached => message::InstanceStatus::Attached,
            InstanceRunningState::Detached => message::InstanceStatus::Detached,
            InstanceRunningState::Finished(_) => message::InstanceStatus::Finished,
        }
    }
}

#[derive(Clone)]
pub enum AttachInstanceResult {
    /// The instance is running and the client has been attached to it successfully.
    AttachedRunning,
    /// The instance has finished execution and the client has been attached to it successfully.
    AttachedFinished(TerminationCause),
    /// The instance is not found.
    InstanceNotFound,
    /// Another client has already been attached to this instance.
    AlreadyAttached,
}

struct InstanceHandle {
    username: String,
    program_hash: String,
    arguments: Vec<String>,
    start_time: std::time::Instant,
    output_delivery_ctrl: OutputDeliveryCtrl,
    running_state: InstanceRunningState,
    join_handle: tokio::task::JoinHandle<()>,
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

            Command::UploadProgram {
                hash,
                raw,
                dependencies,
                event,
            } => {
                // Validate that all dependencies exist
                for dep in &dependencies {
                    if !self.loaded_libraries.contains_key(dep) {
                        event
                            .send(Err(RuntimeError::MissingDependency(dep.clone())))
                            .unwrap();
                        return;
                    }
                }

                if self.programs_in_memory.contains_key(&hash) {
                    // Update dependencies even if program already exists
                    if let Some(mut program) = self.programs_in_memory.get_mut(&hash) {
                        program.dependencies = dependencies;
                    }
                    event.send(Ok(hash)).unwrap();
                } else if let Ok(component) = Component::from_binary(&self.engine, raw.as_slice()) {
                    let loaded_program = LoadedProgram {
                        component,
                        dependencies,
                    };
                    self.programs_in_memory.insert(hash.to_string(), loaded_program);

                    // Write to disk
                    let file_path = std::path::Path::new(&self.cache_dir).join(&hash);
                    std::fs::write(&file_path, &raw).unwrap();
                    self.programs_in_disk.insert(hash.clone(), file_path);
                    event.send(Ok(hash)).unwrap();
                } else {
                    event
                        .send(Err(RuntimeError::Other("Failed to compile".into())))
                        .unwrap();
                }
            }

            Command::LaunchInstance {
                username,
                program_hash,
                dependencies,
                event,
                arguments,
                detached,
            } => {
                let res = self
                    .launch_instance(username, program_hash, dependencies, arguments, detached)
                    .await;
                event
                    .send(res)
                    .map_err(|_| "Failed to send instance ID after launching instance")
                    .unwrap();
            }

            Command::AttachInstance { inst_id, event } => {
                let res = self.attach_instance(inst_id);
                event
                    .send(res)
                    .map_err(|_| "Failed to send attach instance result")
                    .unwrap();
            }

            Command::DetachInstance { inst_id } => {
                self.detach_instance(inst_id);
            }

            Command::AllowOutput { inst_id } => {
                self.allow_output(inst_id);
            }

            Command::LaunchServerInstance {
                username,
                program_hash,
                port,
                arguments,
                event,
            } => {
                let _ = self
                    .launch_server_instance(username, program_hash, port, arguments)
                    .await;
                event.send(Ok(())).unwrap();
            }

            Command::TerminateInstance {
                inst_id,
                notification_to_client,
            } => {
                self.terminate_instance(inst_id, notification_to_client);
            }

            Command::FinishInstance { inst_id, cause } => {
                self.finish_instance(inst_id, cause);
            }

            Command::SetOutputDelivery { inst_id, mode } => {
                self.set_output_delivery(inst_id, mode);
            }

            Command::GetVersion { event } => {
                event.send(VERSION.to_string()).unwrap();
            }

            Command::DebugQuery { query, event } => {
                let res = match query.as_str() {
                    "ping" => {
                        format!("pong")
                    }
                    "get_instance_count" => {
                        format!("{}", self.running_instances.len())
                    }
                    "get_server_instance_count" => {
                        format!("{}", self.running_server_instances.len())
                    }
                    // Add the new queries here
                    "list_running_instances" => {
                        let instances: Vec<String> = self
                            .running_instances
                            .iter()
                            .map(|item| {
                                format!(
                                    "Instance ID: {}, Program Hash: {}",
                                    item.key(),
                                    item.value().program_hash
                                )
                            })
                            .collect();

                        format!("{}", instances.join("\n"))
                    }
                    "list_in_memory_programs" => {
                        let keys: Vec<String> = self
                            .programs_in_memory
                            .iter()
                            .map(|item| item.key().clone())
                            .collect();

                        format!("{}", keys.join("\n"))
                    }

                    _ => {
                        format!("Unknown query: {}", query)
                    }
                };

                event.send(QueryResponse { value: res }).unwrap();
            }
            Command::ListInstances { username, event } => {
                // Internal users (from monitor) can see all instances
                let show_all = username == "internal";
                let mut instances: Vec<message::InstanceInfo> = self
                    .running_instances
                    .iter()
                    .chain(self.finished_instances.iter())
                    .filter(|item| show_all || item.value().username == username)
                    .map(|item| message::InstanceInfo {
                        id: item.key().to_string(),
                        arguments: item.value().arguments.clone(),
                        status: item.value().running_state.clone().into(),
                        username: item.value().username.clone(),
                        elapsed_secs: item.value().start_time.elapsed().as_secs(),
                        kv_pages_used: 0, // TODO: query from resource_manager
                    })
                    .collect();
                
                // Sort by elapsed time (most recent first) and limit to 50 for performance
                instances.sort_by(|a, b| a.elapsed_secs.cmp(&b.elapsed_secs));
                instances.truncate(50);

                event.send(instances).unwrap();
            }

            Command::UploadLibrary {
                name,
                raw,
                dependencies,
                event,
            } => {
                let res = self.upload_library(name, raw, dependencies).await;
                event.send(res).unwrap();
            }

            Command::ListLibraries { event } => {
                let libraries = self.list_libraries();
                event.send(libraries).unwrap();
            }
        }
    }
}

impl Runtime {
    fn new<P: AsRef<std::path::Path>>(cache_dir: P) -> Self {
        // Configure Wasmtime engine
        let mut config = Config::default();
        config.async_support(true);

        // TODO: Adjust settings later: https://docs.wasmtime.dev/api/wasmtime/struct.PoolingAllocationConfig.html
        // let mut pooling_config = PoolingAllocationConfig::default();
        //config.allocation_strategy(InstanceAllocationStrategy::Pooling(pooling_config));

        let engine = Engine::new(&config).unwrap();

        let mut base_linker = Linker::<InstanceState>::new(&engine);

        // Add host-defined interfaces to the base linker
        wasmtime_wasi::p2::add_to_linker_async(&mut base_linker)
            .map_err(|e| RuntimeError::Other(format!("Failed to link WASI: {e}")))
            .unwrap();
        wasmtime_wasi_http::add_only_http_to_linker_async(&mut base_linker)
            .map_err(|e| RuntimeError::Other(format!("Failed to link WASI: {e}")))
            .unwrap();

        api::add_to_linker(&mut base_linker).unwrap();

        let cache_dir = cache_dir.as_ref().join("programs");
        // Ensure the cache directory exists
        std::fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");

        Self {
            engine,
            base_linker: Arc::new(base_linker),
            cache_dir,
            programs_in_memory: DashMap::new(),
            programs_in_disk: DashMap::new(),
            running_instances: DashMap::new(),
            finished_instances: DashMap::new(),
            running_server_instances: DashMap::new(),
            loaded_libraries: HashMap::new(),
            library_load_order: Vec::new(),
        }
    }

    fn load_existing_programs(&self) -> Result<(), RuntimeError> {
        let entries = std::fs::read_dir(&self.cache_dir)?; // Will map to RuntimeError::Io automatically
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

    /// Get the component and its dependencies for a program hash.
    /// If the program is not in memory, it will be loaded from disk with empty dependencies.
    fn get_program(&self, hash: &str) -> Result<(Component, Vec<String>), RuntimeError> {
        // 1) Make sure the program is loaded in memory
        if self.programs_in_memory.get(hash).is_none() {
            // load from disk if possible
            if let Some(path_entry) = self.programs_in_disk.get(hash) {
                let component =
                    Component::from_file(&self.engine, path_entry.value()).map_err(|err| {
                        RuntimeError::CompileWasm {
                            path: path_entry.value().to_path_buf(),
                            source: err,
                        }
                    })?;
                // Programs loaded from disk don't have stored dependencies
                // (dependencies can be provided at launch time)
                let loaded_program = LoadedProgram {
                    component,
                    dependencies: Vec::new(),
                };
                self.programs_in_memory
                    .insert(hash.to_string(), loaded_program);
            } else {
                // If not on disk either, return a custom error
                return Err(RuntimeError::MissingProgram(hash.to_string()));
            }
        }

        // 2) Now we have a compiled program
        let program = match self.programs_in_memory.get(hash) {
            Some(p) => (p.component.clone(), p.dependencies.clone()),
            None => {
                return Err(RuntimeError::Other(
                    "Failed to get program from memory".into(),
                ));
            }
        };

        Ok(program)
    }

    /// Actually start a program instance with dynamic linking support.
    /// If `dependencies` is non-empty, use those dependencies instead of the program's stored dependencies.
    async fn launch_instance(
        &mut self,
        username: String,
        program_hash: String,
        dependencies: Vec<String>,
        arguments: Vec<String>,
        detached: bool,
    ) -> Result<InstanceId, RuntimeError> {
        let (component, stored_dependencies) = self.get_program(&program_hash)?;
        let instance_id = Uuid::new_v4();

        // Use provided dependencies if non-empty, otherwise use stored dependencies
        let dependencies = if dependencies.is_empty() {
            stored_dependencies
        } else {
            dependencies
        };

        // Validate that all dependencies exist
        for dep in &dependencies {
            if !self.loaded_libraries.contains_key(dep) {
                return Err(RuntimeError::MissingDependency(dep.clone()));
            }
        }

        // Collect all dependencies (recursively) in topological order
        let all_deps = self.collect_recursive_dependencies(&dependencies)?;

        // Validate that all recursive dependencies exist
        for dep in &all_deps {
            if !self.loaded_libraries.contains_key(dep) {
                return Err(RuntimeError::MissingDependency(dep.clone()));
            }
        }

        // Instantiate and run in a task
        let engine = self.engine.clone();

        // Collect library components for the dependencies
        let library_components: Vec<(String, Component)> = all_deps
            .iter()
            .filter_map(|name| {
                self.loaded_libraries
                    .get(name)
                    .map(|lib| (name.clone(), lib.component.clone()))
            })
            .collect();

        // Create a oneshot channel to signal when the task can start
        let (start_tx, start_rx) = oneshot::channel();
        // Create a oneshot channel to receive the output delivery controller
        let (output_delivery_ctrl_tx, output_delivery_ctrl_rx) = oneshot::channel();

        let join_handle = tokio::spawn(Self::launch_with_linking(
            instance_id,
            username.clone(),
            component,
            library_components,
            arguments.clone(),
            detached,
            engine,
            start_rx,
            output_delivery_ctrl_tx,
        ));

        // Wait for the output delivery controller to be sent back
        let output_delivery_ctrl = output_delivery_ctrl_rx.await.unwrap();

        let running_state = if detached {
            InstanceRunningState::Detached
        } else {
            InstanceRunningState::Attached
        };

        // Record in the "running_instances" so we can manage it later
        let instance_handle = InstanceHandle {
            username,
            program_hash,
            arguments,
            start_time: std::time::Instant::now(),
            output_delivery_ctrl,
            running_state,
            join_handle,
        };
        self.running_instances.insert(instance_id, instance_handle);

        // Signal the task to start now that the join_handle is in the map
        let _ = start_tx.send(());

        Ok(instance_id)
    }

    /// Set the instance as attached. Its output will be streamed to the client.
    fn attach_instance(&self, inst_id: InstanceId) -> AttachInstanceResult {
        // Check if the instance is still running.
        if let Some(mut handle) = self.running_instances.get_mut(&inst_id) {
            // An instance cannot be attached if it is already attached.
            if let InstanceRunningState::Attached = handle.running_state {
                return AttachInstanceResult::AlreadyAttached;
            }

            handle.running_state = InstanceRunningState::Attached;
            return AttachInstanceResult::AttachedRunning;
        }

        // Check if the instance has finished execution.
        if let Some(mut handle) = self.finished_instances.get_mut(&inst_id) {
            // Set the instance to attached to prevent other clients from attaching to it.
            // Take the termination cause from the finished state and return it. The code that
            // takes the termination cause should be responsible for later cleaning up the instance.
            if matches!(&handle.running_state, InstanceRunningState::Finished(_)) {
                if let InstanceRunningState::Finished(cause) =
                    std::mem::replace(&mut handle.running_state, InstanceRunningState::Attached)
                {
                    return AttachInstanceResult::AttachedFinished(cause);
                }
            }
        }

        return AttachInstanceResult::InstanceNotFound;
    }

    /// Set the instance as detached. Prior to calling this method, the instance output delivery
    /// mode must be set to buffered.
    fn detach_instance(&self, inst_id: InstanceId) {
        if let Some(mut handle) = self.running_instances.get_mut(&inst_id) {
            handle.running_state = InstanceRunningState::Detached;
        }
    }

    /// Allow output for a running instance
    fn allow_output(&self, inst_id: InstanceId) {
        if let Some(handle) = self.running_instances.get(&inst_id) {
            handle.output_delivery_ctrl.allow_output();
        }
    }

    /// Actually start a server program instance (HTTP server mode).
    /// Note: Server instances currently don't support dynamic linking - they use the base linker.
    async fn launch_server_instance(
        &self,
        username: String,
        program_hash: String,
        port: u32,
        arguments: Vec<String>,
    ) -> Result<InstanceId, RuntimeError> {
        let instance_id = Uuid::new_v4();
        let (component, _dependencies) = self.get_program(&program_hash)?;

        // Instantiate and run in a task
        let engine = self.engine.clone();
        let linker = self.base_linker.clone();
        let addr = SocketAddr::from(([127, 0, 0, 1], port as u16));

        // Create a oneshot channel to signal when the task can start
        let (start_tx, start_rx) = oneshot::channel();

        let join_handle = tokio::spawn(Self::launch_server(
            addr,
            username.clone(),
            component,
            arguments.clone(),
            engine,
            linker,
            start_rx,
        ));

        // Create a dummy output delivery controller for server instances (not used since each request gets its own instance)
        let (dummy_state, output_delivery_ctrl) =
            InstanceState::new(Uuid::new_v4(), username.clone(), vec![]).await;
        drop(dummy_state); // We don't actually use this

        // Record in the "running_instances" so we can manage it later
        let instance_handle = InstanceHandle {
            username,
            program_hash,
            arguments,
            start_time: std::time::Instant::now(),
            output_delivery_ctrl,
            running_state: InstanceRunningState::Detached,
            join_handle,
        };
        self.running_server_instances
            .insert(instance_id, instance_handle);

        // Signal the task to start now that the join_handle is in the map
        let _ = start_tx.send(());

        Ok(instance_id)
    }

    /// Terminate a running instance, and optionally notify the client.
    fn terminate_instance(
        &self,
        instance_id: InstanceId,
        notification_to_client: Option<TerminationCause>,
    ) {
        let instance = self
            .running_instances
            .remove(&instance_id)
            .or(self.finished_instances.remove(&instance_id));

        if let Some((_, handle)) = instance {
            handle.join_handle.abort();

            model::cleanup_instance(instance_id.clone());

            if let Some(cause) = notification_to_client {
                server::InstanceEvent::Terminate {
                    inst_id: instance_id,
                    cause,
                }
                .dispatch();
            }
        }
    }

    /// Finish a running instance. If the instance is attached, it will notify the client and
    /// clean up the instance. If the instance is detached, it will mark the instance as finished
    /// and add it to the finished instances map. If the instance is already finished, it will
    /// panic.
    fn finish_instance(&self, instance_id: InstanceId, cause: TerminationCause) {
        if let Some((_, mut handle)) = self.running_instances.remove(&instance_id) {
            match handle.running_state {
                // For an attached instance, its output must have been streamed to the client,
                // so we can clean up the instance and notify the client about the termination.
                InstanceRunningState::Attached => {
                    handle.join_handle.abort();
                    model::cleanup_instance(instance_id.clone());

                    server::InstanceEvent::Terminate {
                        inst_id: instance_id,
                        cause,
                    }
                    .dispatch();
                }
                // For a detached instance, we can just mark it as finished and add it to the
                // finished instances map. Its output is buffered and waiting to be streamed to
                //the client.
                InstanceRunningState::Detached => {
                    handle.running_state = InstanceRunningState::Finished(cause);
                    self.finished_instances.insert(instance_id, handle);
                }
                // If the instance is already finished, we can't finish it again.
                // This should never happen.
                InstanceRunningState::Finished(_) => {
                    panic!("Instance {instance_id} is already finished and cannot be sealed again")
                }
            }
        }
    }

    /// Set the output delivery for a running instance
    fn set_output_delivery(&self, instance_id: InstanceId, output_delivery: OutputDelivery) {
        if let Some(handle) = self.running_instances.get(&instance_id) {
            handle
                .output_delivery_ctrl
                .set_output_delivery(output_delivery);
        }

        if let Some(handle) = self.finished_instances.get(&instance_id) {
            handle
                .output_delivery_ctrl
                .set_output_delivery(output_delivery);
        }
    }

    /// Upload a library component and register its exports.
    /// The library can specify dependencies on other libraries.
    async fn upload_library(
        &mut self,
        name: String,
        raw: Vec<u8>,
        dependencies: Vec<String>,
    ) -> Result<String, RuntimeError> {
        // Check if library already exists
        if self.loaded_libraries.contains_key(&name) {
            return Err(RuntimeError::LibraryExists(name));
        }

        // Check that all dependencies exist by name
        for dep in &dependencies {
            if !self.loaded_libraries.contains_key(dep) {
                return Err(RuntimeError::MissingDependency(dep.clone()));
            }
        }

        // Compile the component to validate it's a proper WASM component
        let component = Component::from_binary(&self.engine, &raw)
            .map_err(|e| RuntimeError::Other(format!("Failed to compile library: {e}")))?;

        // Perform rigorous dependency validation using instantiate_pre
        self.validate_library_imports(&component, &dependencies)?;

        // Store the loaded library
        let loaded_library = LoadedLibrary {
            name: name.clone(),
            component,
            dependencies,
        };

        self.loaded_libraries.insert(name.clone(), loaded_library);
        self.library_load_order.push(name.clone());

        tracing::info!("Library '{}' loaded successfully", name);
        Ok(name)
    }

    /// Validate that a library component's imports can be satisfied by the host
    /// interfaces and the declared dependencies.
    ///
    /// This creates a validation linker with:
    /// 1. Host-defined interfaces (WASI, HTTP, Pie API)
    /// 2. Stub definitions for all exports from dependencies (recursively)
    ///
    /// Then uses `instantiate_pre` to verify that all imports are satisfiable.
    fn validate_library_imports(
        &self,
        component: &Component,
        dependencies: &[String],
    ) -> Result<(), RuntimeError> {
        // Create a fresh linker for validation
        let mut validation_linker = Linker::<InstanceState>::new(&self.engine);

        // Add host-defined interfaces
        wasmtime_wasi::p2::add_to_linker_async(&mut validation_linker)
            .map_err(|e| RuntimeError::Other(format!("Failed to link WASI: {e}")))?;
        wasmtime_wasi_http::add_only_http_to_linker_async(&mut validation_linker)
            .map_err(|e| RuntimeError::Other(format!("Failed to link WASI HTTP: {e}")))?;
        api::add_to_linker(&mut validation_linker)
            .map_err(|e| RuntimeError::Other(format!("Failed to link Pie API: {e}")))?;

        // Collect all dependencies (recursively) in topological order
        let all_deps = self.collect_recursive_dependencies(dependencies)?;

        // Add stub definitions for all dependency exports
        for dep_name in &all_deps {
            if let Some(dep_lib) = self.loaded_libraries.get(dep_name) {
                self.add_stub_definitions_for_component(&mut validation_linker, &dep_lib.component)
                    .map_err(|e| {
                        RuntimeError::Other(format!(
                            "Failed to add stub definitions for '{}': {e}",
                            dep_name
                        ))
                    })?;
            }
        }

        // Try to create an InstancePre to verify all imports are satisfiable
        validation_linker.instantiate_pre(component).map_err(|e| {
            RuntimeError::UnsatisfiableImports(format!(
                "Library imports not satisfiable. \
                 Declared dependencies: [{}]. \
                 Available dependencies: [{}]. \
                 Error: {e}",
                dependencies.join(", "),
                all_deps.join(", ")
            ))
        })?;

        Ok(())
    }

    /// Collect all dependencies recursively in topological order (dependencies before dependents).
    fn collect_recursive_dependencies(
        &self,
        direct_deps: &[String],
    ) -> Result<Vec<String>, RuntimeError> {
        let mut result = Vec::new();
        let mut visited = std::collections::HashSet::new();

        fn visit(
            dep_name: &str,
            loaded_libraries: &HashMap<String, LoadedLibrary>,
            result: &mut Vec<String>,
            visited: &mut std::collections::HashSet<String>,
        ) -> Result<(), RuntimeError> {
            if visited.contains(dep_name) {
                return Ok(());
            }
            visited.insert(dep_name.to_string());

            if let Some(lib) = loaded_libraries.get(dep_name) {
                // Visit transitive dependencies first
                for transitive_dep in &lib.dependencies {
                    visit(transitive_dep, loaded_libraries, result, visited)?;
                }
            }

            result.push(dep_name.to_string());
            Ok(())
        }

        for dep in direct_deps {
            visit(dep, &self.loaded_libraries, &mut result, &mut visited)?;
        }

        Ok(result)
    }

    /// Add stub definitions for all exports of a component to the linker.
    ///
    /// This scans the component's exports and creates stub functions with matching
    /// signatures. The stubs are never actually called - they're only used for
    /// validation via `instantiate_pre`.
    fn add_stub_definitions_for_component(
        &self,
        linker: &mut Linker<InstanceState>,
        component: &Component,
    ) -> Result<(), wasmtime::Error> {
        // Get the component's type to iterate exports
        let component_type = linker.substituted_component_type(component)?;

        // Iterate directly over interface exports without collecting into a vector
        for (export_name, export_item) in component_type.exports(&self.engine) {
            if let ComponentItem::ComponentInstance(instance_type) = export_item {
                self.add_stub_definitions_for_interface(linker, export_name, &instance_type)?;
            }
        }

        Ok(())
    }

    /// Add stub definitions for an interface to the linker.
    fn add_stub_definitions_for_interface(
        &self,
        linker: &mut Linker<InstanceState>,
        interface_name: &str,
        instance_type: &wasmtime::component::types::ComponentInstance,
    ) -> Result<(), wasmtime::Error> {
        // Try to get or create the instance; if it fails, the interface is already defined
        let mut root = linker.root();
        let mut inst = match root.instance(interface_name) {
            Ok(inst) => inst,
            Err(_) => return Ok(()), // Interface already fully defined, skip
        };

        // Register both resources and functions
        for (export_name, export_item) in instance_type.exports(&self.engine) {
            match export_item {
                ComponentItem::Resource(_) => {
                    // Register a stub resource with a no-op destructor
                    // Silently skip if already defined
                    let _ = inst.resource_async(
                        export_name,
                        wasmtime::component::ResourceType::host::<StubResource>(),
                        |_store, _rep| Box::new(async { Ok(()) }),
                    );
                }
                ComponentItem::ComponentFunc(_) => {
                    // Create a stub function that matches the signature.
                    // Since we're only validating via instantiate_pre, these stubs will never
                    // actually be called - they just need to exist with matching signatures.
                    let func_name = export_name.to_string();

                    // Silently skip if already defined
                    let _ =
                        inst.func_new_async(export_name, move |_store, _func, _args, _results| {
                            let func_name = func_name.clone();
                            Box::new(async move {
                                // This stub should never be called during validation.
                                // instantiate_pre only checks that imports are defined,
                                // it doesn't actually call them.
                                unreachable!(
                                    "Stub function '{}' was unexpectedly called during validation",
                                    func_name
                                );
                            })
                        });
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// List all loaded libraries
    fn list_libraries(&self) -> Vec<LibraryInfo> {
        self.library_load_order
            .iter()
            .enumerate()
            .filter_map(|(index, name)| {
                self.loaded_libraries.get(name).map(|lib| LibraryInfo {
                    name: lib.name.clone(),
                    dependencies: lib.dependencies.clone(),
                    load_order: index as u32,
                })
            })
            .collect()
    }

    async fn handle_server_request(
        engine: Engine,
        linker: Arc<Linker<InstanceState>>,
        username: String,
        component: Component,
        arguments: Vec<String>,
        req: hyper::Request<hyper::body::Incoming>,
    ) -> anyhow::Result<hyper::Response<HyperOutgoingBody>> {
        let inst_id = Uuid::new_v4();
        let (inst_state, _output_delivery_ctrl) =
            InstanceState::new(inst_id, username, arguments).await;

        let mut store = Store::new(&engine, inst_state);
        let (sender, receiver) = oneshot::channel();

        let req = store.data_mut().new_incoming_request(Scheme::Http, req)?;
        let out = store.data_mut().new_response_outparam(sender)?;

        let instance = linker
            .instantiate_async(&mut store, &component)
            .await
            .map_err(|e| RuntimeError::Other(format!("Instantiation error: {e}")))?;

        let (_, serve_export) = instance
            .get_export(&mut store, None, "wasi:http/incoming-handler@0.2.4")
            .ok_or_else(|| RuntimeError::Other("No 'serve' function found".into()))?;

        let (_, handle_func_export) = instance
            .get_export(&mut store, Some(&serve_export), "handle")
            .ok_or_else(|| RuntimeError::Other("No 'handle' function found".into()))?;

        let handle_func = instance
            .get_typed_func::<(Resource<IncomingRequest>, Resource<ResponseOutparam>), ()>(
                &mut store,
                &handle_func_export,
            )
            .map_err(|e| RuntimeError::Other(format!("Failed to get 'handle' function: {e}")))?;

        let task = tokio::task::spawn(async move {
            if let Err(e) = handle_func.call_async(&mut store, (req, out)).await {
                eprintln!("error: {e:?}");
                return Err(e);
            }
            Ok(())
        });

        match receiver.await {
            Ok(Ok(resp)) => Ok(resp),
            Ok(Err(e)) => Err(e.into()),
            Err(_) => {
                let e = match task.await {
                    Ok(r) => {
                        r.expect_err("if the receiver has an error, the task must have failed")
                    }
                    Err(e) => e.into(),
                };
                Err(e.context("guest never invoked `response-outparam::set` method"))
            }
        }
    }

    async fn launch_server(
        addr: SocketAddr,
        username: String,
        component: Component,
        arguments: Vec<String>,
        engine: Engine,
        linker: Arc<Linker<InstanceState>>,
        start_rx: oneshot::Receiver<()>,
    ) {
        // Wait for the signal to start
        let _ = start_rx.await;

        let result = async {
            let socket = tokio::net::TcpSocket::new_v4()?;
            socket.set_reuseaddr(!cfg!(windows))?;
            socket.bind(addr)?;
            let listener = socket.listen(100)?;
            eprintln!("Serving HTTP on http://{}/", listener.local_addr()?);
            //store.data_mut().w
            tokio::task::spawn(async move {
                loop {
                    let (stream, _) = listener.accept().await.unwrap();
                    let stream = TokioIo::new(stream);
                    let engine_ = engine.clone();
                    let linker_ = linker.clone();
                    let component_ = component.clone();
                    let arguments_ = arguments.clone();
                    let username_ = username.clone();
                    tokio::task::spawn(async move {
                        if let Err(e) = http1::Builder::new()
                            .keep_alive(true)
                            .serve_connection(
                                stream,
                                hyper::service::service_fn(move |req| {
                                    Self::handle_server_request(
                                        engine_.clone(),
                                        linker_.clone(),
                                        username_.clone(),
                                        component_.clone(),
                                        arguments_.clone(),
                                        req,
                                    )
                                }),
                            )
                            .await
                        {
                            eprintln!("error: {e:?}");
                        }
                    });
                }
            });
            anyhow::Ok(())
        };
        if let Err(e) = result.await {
            eprintln!("error: {e}");
        }
    }

    async fn launch(
        instance_id: InstanceId,
        username: String,
        component: Component,
        arguments: Vec<String>,
        detached: bool,
        engine: Engine,
        linker: Arc<Linker<InstanceState>>,
        start_rx: oneshot::Receiver<()>,
        output_delivery_ctrl_tx: oneshot::Sender<OutputDeliveryCtrl>,
    ) {
        // Create the instance state and output delivery controller
        let (inst_state, output_delivery_ctrl) =
            InstanceState::new(instance_id, username, arguments).await;

        let output_delivery = if detached {
            OutputDelivery::Buffered
        } else {
            OutputDelivery::Streamed
        };

        // Set the initial output delivery mode
        output_delivery_ctrl.set_output_delivery(output_delivery);

        // Send the output delivery controller back before starting
        output_delivery_ctrl_tx
            .send(output_delivery_ctrl)
            .map_err(|_| "Failed to send output delivery controller")
            .unwrap();

        // Wait for the signal to start
        start_rx.await.unwrap();

        // Wrap everything in a closure returning a Result,
        // so we can capture errors more systematically if desired:
        let result = async {
            let mut store = Store::new(&engine, inst_state);

            let instance = linker
                .instantiate_async(&mut store, &component)
                .await
                .map_err(|e| RuntimeError::Other(format!("Instantiation error: {e}")))?;

            // Attempt to call "run"
            let (_, run_export) = instance
                .get_export(&mut store, None, "inferlet:core/run")
                .ok_or_else(|| RuntimeError::Other("No 'run' function found".into()))?;

            let (_, run_func_export) = instance
                .get_export(&mut store, Some(&run_export), "run")
                .ok_or_else(|| RuntimeError::Other("No 'run' function found".into()))?;

            let run_func = instance
                .get_typed_func::<(), (Result<(), String>,)>(&mut store, &run_func_export)
                .map_err(|e| RuntimeError::Other(format!("Failed to get 'run' function: {e}")))?;

            return match run_func.call_async(&mut store, ()).await {
                Ok((Ok(()),)) => {
                    let return_value = store.data().return_value();
                    //println!("Instance {instance_id} finished normally");
                    Ok(return_value)
                }
                Ok((Err(runtime_err),)) => {
                    //eprintln!("Instance {instance_id} returned an error");
                    Err(RuntimeError::Other(runtime_err))
                }
                Err(call_err) => {
                    //eprintln!("Instance {instance_id} call error: {call_err}");
                    Err(RuntimeError::Other(format!("Call error: {call_err}")))
                }
            };
        }
        .await;

        match result {
            Ok(return_value) => {
                Command::FinishInstance {
                    inst_id: instance_id,
                    cause: TerminationCause::Normal(return_value.unwrap_or_default()),
                }
                .dispatch();
            }
            Err(err) => {
                tracing::info!("Instance {instance_id} failed: {err}");
                Command::FinishInstance {
                    inst_id: instance_id,
                    cause: TerminationCause::Exception(err.to_string()),
                }
                .dispatch();
            }
        }
    }

    /// Launch an instance with dynamic linking support.
    /// This creates a fresh linker, instantiates dependencies, and registers forwarding implementations.
    async fn launch_with_linking(
        instance_id: InstanceId,
        username: String,
        component: Component,
        library_components: Vec<(String, Component)>,
        arguments: Vec<String>,
        detached: bool,
        engine: Engine,
        start_rx: oneshot::Receiver<()>,
        output_delivery_ctrl_tx: oneshot::Sender<OutputDeliveryCtrl>,
    ) {
        // Create the instance state and output delivery controller
        let (inst_state, output_delivery_ctrl) =
            InstanceState::new(instance_id, username, arguments).await;

        let output_delivery = if detached {
            OutputDelivery::Buffered
        } else {
            OutputDelivery::Streamed
        };

        // Set the initial output delivery mode
        output_delivery_ctrl.set_output_delivery(output_delivery);

        // Send the output delivery controller back before starting
        output_delivery_ctrl_tx
            .send(output_delivery_ctrl)
            .map_err(|_| "Failed to send output delivery controller")
            .unwrap();

        // Wait for the signal to start
        start_rx.await.unwrap();

        // Wrap everything in a closure returning a Result
        let result = async {
            // Create the store with the instance state
            let mut store = Store::new(&engine, inst_state);

            // Create a fresh linker and add host-defined interfaces
            let mut linker = Linker::<InstanceState>::new(&engine);
            wasmtime_wasi::p2::add_to_linker_async(&mut linker)
                .map_err(|e| RuntimeError::Other(format!("Failed to link WASI: {e}")))?;
            wasmtime_wasi_http::add_only_http_to_linker_async(&mut linker)
                .map_err(|e| RuntimeError::Other(format!("Failed to link WASI HTTP: {e}")))?;
            api::add_to_linker(&mut linker)
                .map_err(|e| RuntimeError::Other(format!("Failed to link Pie API: {e}")))?;

            // Instantiate each library in dependency order and register forwarding implementations
            for (lib_name, lib_component) in library_components {
                let lib_instance = linker
                    .instantiate_async(&mut store, &lib_component)
                    .await
                    .map_err(|e| {
                        RuntimeError::Other(format!(
                            "Failed to instantiate library '{}': {e}",
                            lib_name
                        ))
                    })?;

                // Register forwarding implementations for this library's exports
                Self::register_library_exports(&engine, &mut linker, &mut store, &lib_component, lib_instance)
                    .map_err(|e| {
                        RuntimeError::Other(format!(
                            "Failed to register exports for library '{}': {e}",
                            lib_name
                        ))
                    })?;
            }

            // Instantiate the main program
            let instance = linker
                .instantiate_async(&mut store, &component)
                .await
                .map_err(|e| RuntimeError::Other(format!("Instantiation error: {e}")))?;

            // Attempt to call "run"
            let (_, run_export) = instance
                .get_export(&mut store, None, "inferlet:core/run")
                .ok_or_else(|| RuntimeError::Other("No 'run' function found".into()))?;

            let (_, run_func_export) = instance
                .get_export(&mut store, Some(&run_export), "run")
                .ok_or_else(|| RuntimeError::Other("No 'run' function found".into()))?;

            let run_func = instance
                .get_typed_func::<(), (Result<(), String>,)>(&mut store, &run_func_export)
                .map_err(|e| RuntimeError::Other(format!("Failed to get 'run' function: {e}")))?;

            return match run_func.call_async(&mut store, ()).await {
                Ok((Ok(()),)) => {
                    let return_value = store.data().return_value();
                    Ok(return_value)
                }
                Ok((Err(runtime_err),)) => Err(RuntimeError::Other(runtime_err)),
                Err(call_err) => Err(RuntimeError::Other(format!("Call error: {call_err}"))),
            };
        }
        .await;

        match result {
            Ok(return_value) => {
                Command::FinishInstance {
                    inst_id: instance_id,
                    cause: TerminationCause::Normal(return_value.unwrap_or_default()),
                }
                .dispatch();
            }
            Err(err) => {
                tracing::info!("Instance {instance_id} failed: {err}");
                Command::FinishInstance {
                    inst_id: instance_id,
                    cause: TerminationCause::Exception(err.to_string()),
                }
                .dispatch();
            }
        }
    }

    /// Register forwarding implementations for a library's exports.
    /// This scans the library component's exports and registers functions that forward calls
    /// to the library instance.
    fn register_library_exports(
        engine: &Engine,
        linker: &mut Linker<InstanceState>,
        store: &mut Store<InstanceState>,
        library_component: &Component,
        library_instance: wasmtime::component::Instance,
    ) -> Result<(), wasmtime::Error> {
        // Get the component's type to iterate exports
        let component_type = linker.substituted_component_type(library_component)?;

        // Iterate over interface exports
        for (export_name, export_item) in component_type.exports(engine) {
            if let ComponentItem::ComponentInstance(instance_type) = export_item {
                Self::register_interface_exports(
                    engine,
                    linker,
                    store,
                    &export_name,
                    &instance_type,
                    library_instance,
                )?;
            }
        }

        Ok(())
    }

    /// Register forwarding implementations for an interface.
    fn register_interface_exports(
        engine: &Engine,
        linker: &mut Linker<InstanceState>,
        store: &mut Store<InstanceState>,
        interface_name: &str,
        instance_type: &wasmtime::component::types::ComponentInstance,
        library_instance: wasmtime::component::Instance,
    ) -> Result<(), wasmtime::Error> {
        // Get the interface export index from the library
        let (_, interface_idx) = match library_instance.get_export(&mut *store, None, interface_name) {
            Some(idx) => idx,
            None => return Ok(()), // Interface not exported, skip
        };

        let mut root = linker.root();
        let mut inst = match root.instance(interface_name) {
            Ok(inst) => inst,
            Err(_) => return Ok(()), // Interface already fully defined, skip
        };

        // Collect resources first
        let mut resources: Vec<Arc<str>> = Vec::new();
        let mut functions = Vec::new();

        for (export_name, export_item) in instance_type.exports(engine) {
            match export_item {
                ComponentItem::Resource(_) => {
                    let resource_name_arc: Arc<str> = export_name.into();
                    resources.push(resource_name_arc.clone());

                    // Register a stub resource with a destructor that forwards to the library
                    let iface = Arc::<str>::from(interface_name);
                    let res = resource_name_arc;

                    let _ = inst.resource_async(
                        export_name,
                        ResourceType::host::<DynamicResource>(),
                        move |mut store, rep| {
                            let iface = iface.clone();
                            let res = res.clone();

                            Box::new(async move {
                                tracing::debug!(
                                    "Resource destructor called: {}::{} rep={}",
                                    iface, res, rep
                                );

                                // Look up the provider resource from the resource map
                                let provider_resource = store.data_mut().dynamic_resource_map.remove(&rep);

                                if let Some(resource_any) = provider_resource {
                                    resource_any.resource_drop_async(&mut store).await?;
                                }

                                Ok(())
                            })
                        },
                    );
                }
                ComponentItem::ComponentFunc(func_type) => {
                    functions.push((export_name.to_string(), func_type));
                }
                _ => {}
            }
        }

        // Process functions
        for (export_name, func_type) in functions {
            // Look up the function export index
            let (_, func_idx) = match library_instance.get_export(&mut *store, Some(&interface_idx), &export_name) {
                Some(idx) => idx,
                None => continue, // Function not found, skip
            };

            // Resolve the Func handle
            let provider_func = match library_instance.get_func(&mut *store, &func_idx) {
                Some(f) => f,
                None => continue, // Not a function, skip
            };

            // Categorize the function
            let func_category = categorize_function(&export_name, &resources);

            match func_category {
                FuncCategory::Constructor { resource: _ } => {
                    Self::register_constructor_forwarding(&mut inst, &export_name, provider_func)?;
                }
                FuncCategory::Method { resource: _ } => {
                    Self::register_method_forwarding(&mut inst, &export_name, &func_type, provider_func)?;
                }
                FuncCategory::StaticMethod { .. } | FuncCategory::FreeFunction => {
                    Self::register_static_function_forwarding(&mut inst, &export_name, &func_type, provider_func)?;
                }
            }
        }

        Ok(())
    }

    /// Register a constructor function that forwards to the library
    fn register_constructor_forwarding(
        inst: &mut wasmtime::component::LinkerInstance<'_, InstanceState>,
        func_name: &str,
        provider_func: Func,
    ) -> Result<(), wasmtime::Error> {
        let func_name_for_log: Arc<str> = func_name.into();

        inst.func_new_async(func_name, move |mut store, _func, args, results| {
            let func_name_for_log = func_name_for_log.clone();

            Box::new(async move {
                tracing::debug!("Constructor {} called with {} args", func_name_for_log, args.len());

                // Call provider's constructor
                let mut ctor_results = vec![Val::Bool(false)];
                provider_func.call_async(&mut store, args, &mut ctor_results).await?;
                provider_func.post_return_async(&mut store).await?;

                // Get the ResourceAny from the result
                let provider_resource = match &ctor_results[0] {
                    Val::Resource(r) => r.clone(),
                    _ => return Err(wasmtime::Error::msg("constructor did not return resource")),
                };

                // Allocate a host rep and store the mapping
                let rep = store.data_mut().alloc_dynamic_rep();
                store.data_mut().dynamic_resource_map.insert(rep, provider_resource);

                // Create host resource and convert to ResourceAny for the return value
                let host_resource = Resource::<DynamicResource>::new_own(rep);
                let host_resource_any = ResourceAny::try_from_resource(host_resource, &mut store)?;
                results[0] = Val::Resource(host_resource_any);

                Ok(())
            })
        })?;

        Ok(())
    }

    /// Register a method function that forwards to the library
    fn register_method_forwarding(
        inst: &mut wasmtime::component::LinkerInstance<'_, InstanceState>,
        func_name: &str,
        func_type: &wasmtime::component::types::ComponentFunc,
        provider_func: Func,
    ) -> Result<(), wasmtime::Error> {
        let _has_results = func_type.results().len() > 0;
        let func_name_for_log: Arc<str> = func_name.into();

        inst.func_new_async(func_name, move |mut store, _func, args, results| {
            let func_name_for_log = func_name_for_log.clone();
            let num_results = results.len();

            Box::new(async move {
                tracing::debug!("Method {} called with {} args", func_name_for_log, args.len());

                // First arg is the resource handle
                let host_resource_any = match &args[0] {
                    Val::Resource(r) => r.clone(),
                    _ => return Err(wasmtime::Error::msg("expected resource as first argument")),
                };

                // Convert to typed Resource to get the rep
                let host_resource: Resource<DynamicResource> =
                    Resource::try_from_resource_any(host_resource_any, &mut store)?;
                let rep = host_resource.rep();

                // Get the provider resource
                let provider_resource = store
                    .data()
                    .dynamic_resource_map
                    .get(&rep)
                    .cloned()
                    .ok_or_else(|| wasmtime::Error::msg(format!("unknown resource rep={}", rep)))?;

                // Build args for provider call: first element is the provider resource, rest are from args[1..]
                let provider_args: Vec<Val> = std::iter::once(Val::Resource(provider_resource))
                    .chain(args[1..].iter().cloned())
                    .collect();

                // Call provider's method
                if num_results > 0 {
                    let mut method_results = vec![Val::Bool(false); num_results];
                    provider_func.call_async(&mut store, &provider_args, &mut method_results).await?;
                    provider_func.post_return_async(&mut store).await?;
                    for (i, r) in method_results.into_iter().enumerate() {
                        results[i] = r;
                    }
                } else {
                    provider_func.call_async(&mut store, &provider_args, &mut []).await?;
                    provider_func.post_return_async(&mut store).await?;
                }

                Ok(())
            })
        })?;

        Ok(())
    }

    /// Register a static function that forwards to the library
    fn register_static_function_forwarding(
        inst: &mut wasmtime::component::LinkerInstance<'_, InstanceState>,
        func_name: &str,
        func_type: &wasmtime::component::types::ComponentFunc,
        provider_func: Func,
    ) -> Result<(), wasmtime::Error> {
        let _has_results = func_type.results().len() > 0;
        let func_name_for_log: Arc<str> = func_name.into();

        inst.func_new_async(func_name, move |mut store, _func, args, results| {
            let func_name_for_log = func_name_for_log.clone();
            let num_results = results.len();

            Box::new(async move {
                tracing::debug!("Static function {} called with {} args", func_name_for_log, args.len());

                // Call provider's function directly
                if num_results > 0 {
                    let mut func_results = vec![Val::Bool(false); num_results];
                    provider_func.call_async(&mut store, args, &mut func_results).await?;
                    provider_func.post_return_async(&mut store).await?;
                    for (i, r) in func_results.into_iter().enumerate() {
                        results[i] = r;
                    }
                } else {
                    provider_func.call_async(&mut store, args, &mut []).await?;
                    provider_func.post_return_async(&mut store).await?;
                }

                Ok(())
            })
        })?;

        Ok(())
    }
}

/// Dynamic marker type for host-defined resources used in dynamic linking.
/// This is a phantom type used to create host resource handles.
struct DynamicResource;

/// Categories of functions in the component model
enum FuncCategory {
    Constructor { resource: Arc<str> },
    Method { resource: Arc<str> },
    StaticMethod { _resource: Arc<str> },
    FreeFunction,
}

/// Categorize a function based on its name and the known resources
fn categorize_function(func_name: &str, resources: &[Arc<str>]) -> FuncCategory {
    // Check for constructor: [constructor]resource-name
    if let Some(resource_name) = func_name.strip_prefix("[constructor]") {
        let resource = resources
            .iter()
            .find(|r| r.as_ref() == resource_name)
            .cloned()
            .unwrap_or_else(|| resource_name.into());
        return FuncCategory::Constructor { resource };
    }

    // Check for method: [method]resource-name.method-name
    if let Some(rest) = func_name.strip_prefix("[method]") {
        if let Some(dot_pos) = rest.find('.') {
            let resource_name = &rest[..dot_pos];
            let resource = resources
                .iter()
                .find(|r| r.as_ref() == resource_name)
                .cloned()
                .unwrap_or_else(|| resource_name.into());
            return FuncCategory::Method { resource };
        }
    }

    // Check for static method: [static]resource-name.method-name
    if let Some(rest) = func_name.strip_prefix("[static]") {
        if let Some(dot_pos) = rest.find('.') {
            let resource_name = &rest[..dot_pos];
            let resource = resources
                .iter()
                .find(|r| r.as_ref() == resource_name)
                .cloned()
                .unwrap_or_else(|| resource_name.into());
            return FuncCategory::StaticMethod { _resource: resource };
        }
    }

    // Otherwise it's a free function
    FuncCategory::FreeFunction
}
