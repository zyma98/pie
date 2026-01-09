//! iceoryx2 transport layer for PyCrust.
//!
//! This module handles the actual IPC communication using iceoryx2.
//! The transport runs in a dedicated OS thread that does busy-spinning
//! for minimum latency, directly dispatching responses to waiting futures.

use crate::error::{Result, RpcError};
use crate::protocol::{RpcRequest, RpcResponse};
use dashmap::DashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use tokio::sync::oneshot;

/// Maximum message size in bytes (4MB).
const MAX_MESSAGE_SIZE: usize = 4194304;

/// Maximum buffer size for subscribers (handles concurrent requests).
// const SUBSCRIBER_BUFFER_SIZE: usize = 256;

/// Number of spin iterations per poll cycle.
/// With ~1ns per spin_loop hint, 100 spins ≈ 100ns of spinning.
const SPIN_ITERATIONS: u32 = 300;



/// Message to send to the transport thread.
pub enum TransportCommand {
    /// Send a request to the worker.
    SendRequest(RpcRequest),
    /// Shutdown the transport thread.
    Shutdown,
}

/// Shared state between client and transport thread.
pub struct TransportShared {
    /// Pending requests waiting for responses.
    pub pending: DashMap<u64, oneshot::Sender<Result<RpcResponse>>>,
    /// Flag to signal shutdown.
    pub running: AtomicBool,
}

/// Handle to the transport running in a dedicated thread.
pub struct TransportHandle {
    /// Channel to send commands to the transport thread.
    cmd_tx: mpsc::Sender<TransportCommand>,
    /// Shared state with the transport thread.
    shared: Arc<TransportShared>,
    /// Handle to the transport thread.
    thread_handle: Option<thread::JoinHandle<()>>,
}

impl TransportHandle {
    /// Connect to a PyCrust service.
    ///
    /// Spawns a dedicated thread for the transport that does busy-spinning
    /// and directly dispatches responses to waiting futures.
    pub fn connect(service_name: &str) -> Result<Self> {
        let (cmd_tx, cmd_rx) = mpsc::channel::<TransportCommand>();

        let shared = Arc::new(TransportShared {
            pending: DashMap::new(),
            running: AtomicBool::new(true),
        });

        let service_name = service_name.to_string();
        let thread_shared = Arc::clone(&shared);
        let thread_handle = thread::spawn(move || {
            if let Err(e) = run_transport_loop(&service_name, cmd_rx, thread_shared) {
                eprintln!("[pycrust] Transport error: {}", e);
            }
        });

        Ok(Self {
            cmd_tx,
            shared,
            thread_handle: Some(thread_handle),
        })
    }

    /// Get shared state for registering pending requests.
    pub fn shared(&self) -> &Arc<TransportShared> {
        &self.shared
    }

    /// Send a request to the worker.
    pub fn send_request(&self, request: RpcRequest) -> Result<()> {
        self.cmd_tx
            .send(TransportCommand::SendRequest(request))
            .map_err(|_| RpcError::ChannelClosed)
    }

    /// Shutdown the transport.
    pub fn shutdown(&mut self) {
        self.shared.running.store(false, Ordering::Relaxed);
        let _ = self.cmd_tx.send(TransportCommand::Shutdown);
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for TransportHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}


/// Run the transport loop in a dedicated thread.
///
/// This thread does busy-spinning on iceoryx2 and directly dispatches
/// responses to the waiting oneshot channels, completely bypassing Tokio.
fn run_transport_loop(
    service_name: &str,
    cmd_rx: mpsc::Receiver<TransportCommand>,
    shared: Arc<TransportShared>,
) -> Result<()> {
    // use iceoryx2::port::client::Client;
    use iceoryx2::prelude::*;

    let node = NodeBuilder::new()
        .create::<ipc::Service>()
        .map_err(|e| RpcError::ConnectionFailed(format!("Failed to create node: {}", e)))?;

    // Create Request-Response Service
    let service = node
        .service_builder(&service_name.try_into().unwrap())
        .request_response::<[u8], [u8]>()
        .max_active_requests_per_client(256)
        .max_loaned_requests(256)
        .open_or_create()
        .map_err(|e| {
            RpcError::ConnectionFailed(format!("Failed to create service: {}", e))
        })?;

    let client: iceoryx2::port::client::Client<
        iceoryx2::service::ipc::Service,
        [u8],
        (),
        [u8],
        (),
    > = service.client_builder()
        .initial_max_slice_len(1024)
        .allocation_strategy(iceoryx2::prelude::AllocationStrategy::PowerOfTwo)
        .create().map_err(|e| {
        RpcError::ConnectionFailed(format!("Failed to create client: {}", e))
    })?;

    // Store active pending requests we are polling
    let mut active_requests: Vec<(
        u64,
        iceoryx2::pending_response::PendingResponse<
            iceoryx2::service::ipc::Service,
            [u8],
            (),
            [u8],
            (),
        >,
    )> = Vec::with_capacity(256);

    // Reusable buffer to prevent heap thrashing
    let mut serial_buf = Vec::with_capacity(4096); 

    // Track consecutive idle iterations for adaptive sleeping
    let mut idle_count: u32 = 0;

    while shared.running.load(Ordering::Relaxed) {
        let mut had_activity = false;

        // 1. Process Commands
        loop {
            match cmd_rx.try_recv() {
                Ok(TransportCommand::SendRequest(request)) => {
                    serial_buf.clear();
                    // Serialize directly to reusable buffer
                    if let Err(e) = rmp_serde::encode::write_named(&mut serial_buf, &request) {
                        eprintln!("[pycrust] Failed to serialize request: {}", e);
                         if let Some((_, sender)) = shared.pending.remove(&request.id) {
                            let _ = sender.send(Err(RpcError::SerializationError(e)));
                         }
                    } else {
                        if serial_buf.len() > MAX_MESSAGE_SIZE {
                            eprintln!("[pycrust] Request too large: {}", serial_buf.len());
                            if let Some((_, sender)) = shared.pending.remove(&request.id) {
                                let _ = sender.send(Err(RpcError::TransportError(format!("Request too large: {}", serial_buf.len()))));
                            }
                        } else {
                            match client.loan_slice_uninit(serial_buf.len()) {
                                Ok(sample) => {
                                    let sample = sample.write_from_slice(&serial_buf);
                                    match sample.send() {
                                        Ok(pending) => {
                                            if request.id != 0 {
                                                active_requests.push((request.id, pending));
                                            }
                                            // If id == 0, it's a notification. Worker won't reply. 
                                            // We drop 'pending' immediately.
                                            had_activity = true;
                                        }
                                        Err(e) => {
                                            eprintln!("[pycrust] Failed to send request: {}", e);
                                            if let Some((_, sender)) = shared.pending.remove(&request.id) {
                                                let _ = sender.send(Err(RpcError::TransportError(format!("Failed to send request: {}", e))));
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                     eprintln!("[pycrust] Failed to loan sample: {}", e);
                                     if let Some((_, sender)) = shared.pending.remove(&request.id) {
                                         let _ = sender.send(Err(RpcError::TransportError(format!("Failed to loan sample: {}", e))));
                                     }
                                }
                            }
                        }
                    }
                }
                Ok(TransportCommand::Shutdown) => {
                    return Ok(());
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => return Ok(()),
            }
        }

        // 2. Poll Active Requests
        if !active_requests.is_empty() {
            active_requests.retain_mut(|(req_id, pending_request)| {
                // Orphan Check: If client timed out, req_id is removed from pending.
                // We must stop polling to avoid unlimited zombie buildup.
                if !shared.pending.contains_key(req_id) {
                    return false;
                }
                match pending_request.receive() {
                    Ok(Some(sample)) => {
                         let payload: &[u8] = sample.payload();
                         match rmp_serde::from_slice::<RpcResponse>(payload) {
                            Ok(response) => {
                                // Dispatch
                                if let Some((_, sender)) = shared.pending.remove(&response.id) {
                                    let _ = sender.send(Ok(response));
                                }
                            }
                            Err(e) => {
                                eprintln!("[pycrust] Failed to deserialize response: {}", e);
                                if let Some((_, sender)) = shared.pending.remove(req_id) {
                                    let _ = sender.send(Err(RpcError::ProtocolError(format!("Deserialization failed: {}", e))));
                                }
                            }
                        }
                        had_activity = true;
                        false // Remove from list, request completed
                    },
                    Ok(None) => true, // Keep polling
                    Err(e) => {
                        eprintln!("[pycrust] Receive error for req {}: {}", req_id, e);
                         if let Some((_, sender)) = shared.pending.remove(req_id) {
                            let _ = sender.send(Err(RpcError::TransportError(format!("Receive error: {}", e))));
                         }
                        false // Remove on error
                    }
                }
            });
        }

        // 3. Adaptive Spin/Sleep
        if had_activity {
            idle_count = 0;
        } else {
            idle_count = idle_count.saturating_add(1);

            if idle_count < SPIN_ITERATIONS {
                // Spin-wait: tight loop for minimum latency
                std::hint::spin_loop();
            } else {
                // Short sleep to yield CPU if we are truly idle
                std::thread::sleep(std::time::Duration::from_micros(50));
            }
        }
    }

    Ok(())
}


