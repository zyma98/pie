use crate::batching::{BatchingConfig, MultiStreamBatcher};
use crate::handler::{Handler, get_batching_config};
use crate::instance::InstanceId;
use crate::resource::{ResourceId, ResourceManager, ResourceTypeId};
use crate::runtime::trap_exception;
use crate::service::{self, Service, ServiceError};
use crate::tokenizer::BytePairEncoder;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

pub type HandlerId = u32;

pub type CmdQueueId = u32;

static SERVICE_ID_KVS: OnceLock<usize> = OnceLock::new();

/// Dispatches a command to the key-value store service.
pub fn dispatch(command: Command) -> Result<(), ServiceError> {
    let service_id = *SERVICE_ID_KVS.get_or_init(|| service::get_service_id("resource").unwrap());
    service::dispatch(service_id, command)
}
static REGISTERED_MODELS: std::sync::LazyLock<boxcar::Vec<(String, usize)>> =
    std::sync::LazyLock::new(boxcar::Vec::new);

pub fn registered_models() -> Vec<String> {
    REGISTERED_MODELS
        .iter()
        .map(|(_, (model_name, _))| model_name.clone())
        .collect()
}

pub fn register_model(model_name: String, service_id: usize) {
    REGISTERED_MODELS.push((model_name, service_id));
}

pub fn model_service_id(model_name: &str) -> Option<usize> {
    REGISTERED_MODELS
        .iter()
        .find(|(_, (name, _))| name == model_name)
        .map(|(_, (_, service_id))| *service_id)
}

pub fn cleanup_instance(inst_id: InstanceId) {
    REGISTERED_MODELS.iter().for_each(|(_, (_, service_id))| {
        Command::Cleanup { inst_id }.dispatch(*service_id).ok();
    })
}

pub async fn query(service_id: usize, query: &'static str) -> String {
    let (tx, rx) = oneshot::channel();
    Command::Query {
        query,
        response: tx,
    }
    .dispatch(service_id)
    .unwrap();
    rx.await.unwrap()
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HandshakeRequest {
    version: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HandshakeResponse {
    // backend metadata
    version: String,

    // model metadata
    model_name: String,
    model_traits: Vec<String>,
    model_description: String,
    model_template: String,
    model_template_type: String,

    // resources
    kv_page_size: u32,
    resources: Vec<(u32, u32)>, // (id, capacity)

    // tokenizer
    tokenizer_merge_table: Vec<(u32, Vec<u8>)>,
    tokenizer_special_tokens: Vec<(String, u32)>,
    tokenizer_split_regex: String,
    tokenizer_escape_non_printable: bool,
}

/// Defines the set of operations available for the key-value store.
#[derive(Debug)]
pub enum Command {
    Submit {
        inst_id: InstanceId,
        cmd_queue_id: CmdQueueId,
        handler: Handler,
        data: Bytes,
        response: Option<oneshot::Sender<Bytes>>,
    },

    Query {
        query: &'static str,
        response: oneshot::Sender<String>,
    },

    Allocate {
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        count: usize,
        response: oneshot::Sender<Vec<ResourceId>>, // physical ids
    },

    Deallocate {
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        ptrs: Vec<ResourceId>, // physical ids
    },

    Cleanup {
        inst_id: InstanceId,
    },

    GetAllExported {
        type_id: ResourceTypeId,
        response: oneshot::Sender<Vec<(String, Vec<ResourceId>)>>,
    },

    Export {
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        ptrs: Vec<ResourceId>,
        name: String,
    },

    Import {
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        name: String,
        response: oneshot::Sender<Vec<ResourceId>>,
    },

    ReleaseExported {
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        name: String,
    },
}

impl Command {
    pub fn dispatch(self, service_id: usize) -> Result<(), ServiceError> {
        service::dispatch(service_id, self)
    }
}

/// An in-memory key-value store service.
#[derive(Debug)]
pub struct Model {
    // model info
    model_name: String,
    model_traits: Vec<String>,
    model_description: String,
    model_template: String,
    model_template_type: String,

    // resources
    resource_manager: ResourceManager,

    // tokenizer
    tokenizer: Arc<BytePairEncoder>,

    // page size
    kv_page_size: u32,

    // event loops
    scheduler_tx: UnboundedSender<(Handler, CmdQueueId, Option<oneshot::Sender<Bytes>>, Bytes)>,
    scheduling_worker_handle: JoinHandle<()>,
    backend_worker_handle: JoinHandle<()>,
}

impl Model {
    pub async fn new(endpoint: &str) -> anyhow::Result<Self> {
        // --- This setup logic is largely the same, but initializes ResourceManager ---
        let mut socket = DealerSocket::new();
        socket.connect(endpoint).await?;

        let mut batching_config = get_batching_config();
        let mut batch_triggers = HashMap::new();
        for (handler, cfg) in batching_config.iter_mut() {
            if let BatchingConfig::Triggered { trigger, .. } = cfg {
                let t = Arc::new(AtomicBool::new(true));
                batch_triggers.insert(handler.get_handler_id(), t.clone());
                *trigger = Some(t.clone());
            }
        }

        let (backend_tx, backend_rx) = unbounded_channel();
        let (scheduler_tx, scheduler_rx) = unbounded_channel();

        let backend_worker_handle =
            tokio::spawn(Self::backend_worker(socket, backend_rx, batch_triggers));
        let scheduling_worker_handle = tokio::spawn(Self::scheduling_worker(
            batching_config,
            scheduler_rx,
            backend_tx,
        ));

        let (handshake_tx, handshake_rx) = oneshot::channel();
        scheduler_tx.send((
            Handler::Handshake,
            0,
            Some(handshake_tx),
            Bytes::from(
                rmp_serde::to_vec_named(&HandshakeRequest {
                    version: "0.1.0".to_string(),
                })
                .unwrap(),
            ),
        ))?;

        let handshake_info =
            rmp_serde::from_slice::<HandshakeResponse>(&handshake_rx.await?.to_vec())?;

        let merge_table = handshake_info.tokenizer_merge_table.into_iter().collect();
        let special_tokens = handshake_info
            .tokenizer_special_tokens
            .into_iter()
            .collect();
        let pattern = handshake_info.tokenizer_split_regex;
        let escape_non_printable = handshake_info.tokenizer_escape_non_printable;

        let tokenizer = Arc::new(BytePairEncoder::new(
            merge_table,
            special_tokens,
            &pattern,
            escape_non_printable,
        ));

        // Initialize the ResourceManager with information from the handshake
        let resource_manager =
            ResourceManager::new(handshake_info.resources.iter().cloned().collect());

        Ok(Model {
            model_name: handshake_info.model_name,
            model_traits: handshake_info.model_traits,
            model_description: handshake_info.model_description,
            model_template: handshake_info.model_template,
            model_template_type: handshake_info.model_template_type,
            resource_manager,
            tokenizer,
            kv_page_size: handshake_info.kv_page_size,
            scheduler_tx,
            scheduling_worker_handle,
            backend_worker_handle,
        })
    }

    async fn scheduling_worker(
        config: HashMap<Handler, BatchingConfig>,
        mut rx: UnboundedReceiver<(Handler, CmdQueueId, Option<oneshot::Sender<Bytes>>, Bytes)>,
        backend_tx: UnboundedSender<(Handler, Vec<(Option<oneshot::Sender<Bytes>>, Bytes)>)>,
    ) {
        let mut batcher = MultiStreamBatcher::new(config.clone());
        let mut next_poll_duration: Option<Duration> = None;

        // A default timeout to use when the batcher has no pending items.
        // This prevents a busy-loop while still allowing the scheduler to be responsive.
        const IDLE_TIMEOUT: Duration = Duration::from_secs(1);

        loop {
            // Use the specific duration from the batcher, or fall back to the idle timeout.
            let sleep_duration = next_poll_duration.unwrap_or(IDLE_TIMEOUT);

            tokio::select! {
                // Branch 1: A new message is received.
                maybe_msg = rx.recv() => {
                    if let Some((handler, cmd_queue_id, sender, msg)) = maybe_msg {
                        // Eagerly send if no batching config exists for this handler.
                        if !config.contains_key(&handler) {
                            let _ = backend_tx.send((handler, vec![(sender, msg)]));
                            continue;
                        }

                        // Otherwise, add the command to the batcher.
                        batcher.push(cmd_queue_id, handler, (sender, msg), Instant::now());
                    } else {
                        // Channel closed, shut down the scheduler.
                        println!("[Info] Command channel closed, shutting down scheduler handler.");
                        break;
                    }
                },
                // Branch 2: The timeout expires.
                _ = tokio::time::sleep(sleep_duration) => {
                    // Timeout fired. Proceed to the polling logic below.
                }
            }

            // Poll the batcher for any ready batches after either event.
            let batch = batcher.poll(Instant::now());

            for (batch_handler, batch_payload) in batch {
                // check if the handler is sync
                if batch_handler == Handler::Synchronize {
                    let (sender, _) = batch_payload.into_iter().next().unwrap();
                    if let Some(sender) = sender {
                        let _ = sender.send(Bytes::default());
                    }
                    continue;
                }

                if !batch_payload.is_empty() {
                    let _ = backend_tx.send((batch_handler, batch_payload));
                }
            }

            // Recalculate the next required timeout for the next loop iteration.
            next_poll_duration = batcher.next_poll_in(Instant::now());
        }
    }

    /// Manages communication with a backend service over a ZMQ DEALER socket.
    ///
    /// This function runs an event loop that performs two main tasks concurrently:
    /// 1.  Receives commands from an MPSC channel, bundles them into a ZMQ message,
    ///     and sends them to the backend. It registers reply handlers (`oneshot::Sender`)
    ///     to await the responses.
    /// 2.  Receives multipart ZMQ messages from the backend, parses them, and uses a
    ///     correlation ID to route the reply payloads to the correct waiting handlers.
    ///
    /// # Message Formats
    ///
    /// ## Outgoing (Client -> Server)
    /// * Frame 0: `corr_id` (4 bytes, u32 big-endian)
    /// * Frame 1: `handler_id` (4 bytes, u32 big-endian)
    /// * Frame 2...N: `payload_bytes`
    ///
    /// ## Incoming (Server -> Client)
    /// * Frame 0: `corr_id` (4 bytes, u32 big-endian)
    /// * Frame 1...N: `reply_payload_bytes`
    ///
    async fn backend_worker(
        mut socket: DealerSocket,
        mut rx: UnboundedReceiver<(Handler, Vec<(Option<oneshot::Sender<Bytes>>, Bytes)>)>,
        batch_triggers: HashMap<HandlerId, Arc<AtomicBool>>,
    ) {
        // A correlation ID to match outgoing requests with incoming replies.
        // wrapping_add ensures it safely wraps around on overflow instead of panicking.
        let mut corr_id: u32 = 0;

        // Stores the `oneshot` senders to send replies back to the original requesters.
        // The key is a tuple of (correlation_id, batch_index).
        let mut event_table: HashMap<(u32, usize), oneshot::Sender<Bytes>> = HashMap::new();

        loop {
            tokio::select! {
                // Branch 1: Handle outgoing commands received from other parts of the application.
                maybe_command = rx.recv() => {
                    if let Some((handler, payload)) = maybe_command {
                        let current_corr_id = corr_id;

                        // Unzip payload into two parts
                        let (senders, batch): (Vec<_>, Vec<_>) = payload.into_iter().unzip();

                        // Construct the multipart ZMQ message.
                        let mut frames: VecDeque<Bytes> = VecDeque::with_capacity(2 + batch.len());
                        frames.push_back(Bytes::copy_from_slice(&current_corr_id.to_be_bytes()));
                        frames.push_back(Bytes::copy_from_slice(&handler.get_handler_id().to_be_bytes()));
                        frames.extend(batch);

                        let zmq_msg = match ZmqMessage::try_from(frames) {
                            Ok(msg) => msg,
                            Err(e) => {
                                eprintln!("Failed to construct ZMQ message: {:?}", e);
                                continue;
                            }
                        };

                        // Send the message. If it fails, the `senders` are dropped,
                        // which notifies the receivers that the operation failed.
                        if let Err(e) = socket.send(zmq_msg).await {
                            eprintln!("[Error] Socket send failed for corr_id {}: {:?}", current_corr_id, e);
                            continue; // Do not proceed to register handlers.
                        }

                        // If sending was successful, register the reply handlers.
                        for (idx, sender) in senders.into_iter().enumerate() {
                            if let Some(sender) = sender {
                                event_table.insert((current_corr_id, idx), sender);
                            }
                        }

                        // Increment the correlation ID for the next message.
                        corr_id = corr_id.wrapping_add(1);

                    } else {
                        // The channel is closed, indicating the application is shutting down.
                        println!("[Info] Command channel closed, shutting down backend handler.");
                        break;
                    }
                },

                // Branch 2: Handle incoming replies from the backend server.
                result = socket.recv() => {
                    match result {
                        Ok(zmq_msg) => {
                            // We expect at least a correlation ID frame.
                            if zmq_msg.is_empty() {
                                eprintln!("[Warn] Received an empty ZMQ message. Ignoring.");
                                continue;
                            }
                            let mut frames = zmq_msg.into_vecdeque();

                            // The first frame is the correlation ID. ZmqMessage acts like a VecDeque.
                            let corr_id_bytes = frames.pop_front().unwrap();
                            let handler_id_bytes = frames.pop_front().unwrap();

                            // Parse the correlation ID from the first frame.
                            let received_corr_id = u32::from_be_bytes(corr_id_bytes.as_ref().try_into().unwrap());
                            let received_handler_id = u32::from_be_bytes(handler_id_bytes.as_ref().try_into().unwrap());

                            // The remaining frames are payloads. We iterate through them and
                            // find the corresponding `oneshot::Sender` to send the reply back.
                            for (idx, payload) in frames.into_iter().enumerate() {
                                let key = (received_corr_id, idx);
                                if let Some(sender) = event_table.remove(&key) {
                                    // Send the payload. We don't care if the receiver was dropped.
                                    let _ = sender.send(payload);
                                }
                            }

                            // Update the trigger if it exists
                            if let Some(trigger) = batch_triggers.get(&received_handler_id) {
                                trigger.store(true, std::sync::atomic::Ordering::SeqCst);
                            }

                        },
                        Err(e) => {
                            eprintln!("[Error] Socket receive error: {}. Shutting down.", e);
                            break;
                        }
                    }
                }
            }
        }
    }

    pub fn submit(
        &self,
        handler: Handler,
        cmd_queue_id: CmdQueueId,
        sender: Option<oneshot::Sender<Bytes>>,
        msg: Bytes,
    ) {
        let _ = self.scheduler_tx.send((handler, cmd_queue_id, sender, msg));
    }
}
impl Service for Model {
    type Command = Command;

    async fn handle(&mut self, cmd: Self::Command) {
        match cmd {
            Command::Submit {
                inst_id,
                handler,
                cmd_queue_id,
                data,
                response,
            } => {
                self.submit(handler, cmd_queue_id, response, data);
            }

            Command::Query { query, response } => {
                let _ = match query {
                    "name" => response.send(self.model_name.clone()),
                    "license" => response.send(self.model_name.clone()),
                    "traits" => response.send(self.model_traits.join(",")),
                    "description" => response.send(self.model_description.clone()),
                    "prompt_template" => response.send(self.model_template.clone()),
                    "prompt_template_type" => response.send(self.model_template_type.clone()),
                    "kv_page_size" => response.send(self.kv_page_size.to_string()),
                    _ => Ok(()),
                };
            }

            Command::Allocate {
                inst_id,
                type_id,
                count,
                response,
            } => {
                let result = self
                    .resource_manager
                    .allocate_with_oom(inst_id, type_id, count);
                match result {
                    Ok(allocated_ids) => {
                        let _ = response.send(allocated_ids);
                    }
                    Err(e) => {
                        trap_exception(inst_id, e);
                    }
                }
            }

            Command::Deallocate {
                inst_id,
                type_id,
                ptrs,
            } => {
                if let Err(e) = self.resource_manager.deallocate(inst_id, type_id, ptrs) {
                    trap_exception(inst_id, e);
                }
            }

            Command::Cleanup { inst_id } => {
                if let Err(e) = self.resource_manager.cleanup(inst_id) {
                    trap_exception(inst_id, e);
                }
            }

            Command::GetAllExported { type_id, response } => {
                let list = self.resource_manager.get_all_exported(type_id);
                let _ = response.send(list);
            }

            Command::Export {
                inst_id,
                type_id,
                ptrs,
                name,
            } => {
                if let Err(e) = self.resource_manager.export(inst_id, type_id, ptrs, name) {
                    trap_exception(inst_id, e);
                }
            }

            Command::Import {
                inst_id,
                type_id,
                name,
                response,
            } => match self.resource_manager.import(type_id, name) {
                Ok(ptrs) => {
                    let _ = response.send(ptrs);
                }
                Err(e) => trap_exception(inst_id, e),
            },

            Command::ReleaseExported {
                inst_id,
                type_id,
                name,
            } => {
                if let Err(e) = self.resource_manager.release_exported(type_id, name) {
                    trap_exception(inst_id, e);
                }
            }
        }
    }
}
