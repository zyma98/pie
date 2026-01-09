//! Worker loop implementation for PyCrust.

use crate::convert::{msgpack_to_pyobject, pyobject_to_msgpack};
use iceoryx2::prelude::*;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Maximum message size in bytes (4MB).
const MAX_MESSAGE_SIZE: usize = 4194304;

/// Maximum buffer size for subscribers (handles concurrent requests).
// const SUBSCRIBER_BUFFER_SIZE: usize = 256;

/// Number of spin iterations per idle loop.
const SPIN_ITERATIONS: u32 = 1000;

/// Number of idle loops before sleeping (spin budget).
const IDLE_THRESHOLD: u32 = 100;

/// Check Python signals every N idle loops.
const SIGNAL_CHECK_INTERVAL: u32 = 50;

/// Status codes for RPC responses.
mod status {
    pub const OK: u8 = 0;
    pub const METHOD_NOT_FOUND: u8 = 1;
    pub const INVALID_PARAMS: u8 = 2;
    pub const INTERNAL_ERROR: u8 = 3;
}

/// Run the worker loop, dispatching RPC calls to the Python callback.
/// Run the worker loop, dispatching RPC calls to the Python callback.
#[pyfunction]
#[pyo3(signature = (service_name, dispatch_callback))]
pub fn run_worker(py: Python<'_>, service_name: &str, dispatch_callback: Py<PyAny>) -> PyResult<()> {

    // Create iceoryx2 node
    let node = NodeBuilder::new()
        .create::<ipc::Service>()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create node: {}", e)))?;

    // Create Request-Response Service
    let service = node
        .service_builder(&service_name.try_into().unwrap())
        .request_response::<[u8], [u8]>()
        .max_active_requests_per_client(256)
        .max_loaned_requests(256)
        .open_or_create()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create service: {}", e)))?;

    let server = service
        .server_builder()
        .initial_max_slice_len(1024)
        .allocation_strategy(iceoryx2::prelude::AllocationStrategy::PowerOfTwo)
        .create()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create server: {}", e)))?;

    eprintln!("[pycrust-worker] Listening on service: {}", service_name);

    let mut idle_count: u32 = 0;
    
    // ✅ OPTIMIZATION: Reusable buffer to prevent heap allocation churn
    let mut serial_buf = Vec::with_capacity(4096);

    // Main worker loop
    loop {
        // ... (signal checks) ...
        if idle_count % SIGNAL_CHECK_INTERVAL == 0 {
            py.check_signals()?;
        }

        // Poll for incoming requests
        match server.receive() {
            Ok(Some(active_request)) => {
                // ✅ OPTIMIZATION: Direct access to SHM (Zero-Copy Parse)
                // We parse directly from the shared memory slice.
                // No `to_vec()`, no `allow_threads` overhead.
                // Note: `unwrap_or_default` is theoretically risky if payload is empty, but payload() returns valid slice.
                let payload_shm = active_request.payload();

                match rmp_serde::from_slice::<Request>(payload_shm) {
                    Ok(req) => {
                         // `req.payload` is now a slice pointing directly into SHM
                        // We construct the Python object directly from this slice.
                        match msgpack_to_pyobject(py, req.payload) {
                            Ok(py_args) => {
                                // Dispatch to Python
                                // The GIL is held here, which is fine as dispatching to Python requires it anyway.
                                let result = dispatch_callback.call1(py, (req.method, py_args));

                                // Handle Response
                                // If ID is 0, this is a notification (fire-and-forget), so we skip response
                                if req.id != 0 {
                                    let (status, response_bytes_ref) = match result {
                                        Ok(py_res) => {
                                            // TODO: Further optimization - pass &mut serial_buf to avoid intermediate Vec
                                            // For now, we still rely on `pyobject_to_msgpack` returning a Vec<u8>
                                            match pyobject_to_msgpack(py_res.bind(py)) {
                                                Ok(b) => (status::OK, b),
                                                Err(e) => (status::INTERNAL_ERROR, rmp_serde::to_vec_named(&e.to_string()).unwrap_or_default()),
                                            }
                                        },
                                        Err(e) => {
                                             // "Method not found" check
                                            let error_str = e.to_string();
                                            let (status_code, error_msg) = if error_str.contains("Method not found") {
                                                (status::METHOD_NOT_FOUND, error_str)
                                            } else {
                                                (status::INTERNAL_ERROR, error_str)
                                            };
                                            (status_code, rmp_serde::to_vec_named(&error_msg).unwrap_or_default())
                                        }
                                    };

                                    // ✅ OPTIMIZATION: Serialize direct struct to buffer
                                    serial_buf.clear();
                                    let response_struct = Response {
                                        id: req.id,
                                        status,
                                        payload: &response_bytes_ref,
                                    };
                                    
                                    if let Ok(_) = rmp_serde::encode::write_named(&mut serial_buf, &response_struct) {
                                         if serial_buf.len() <= MAX_MESSAGE_SIZE {
                                             if let Ok(sample) = active_request.loan_slice_uninit(serial_buf.len()) {
                                                 let sample = sample.write_from_slice(&serial_buf);
                                                 let _ = sample.send();
                                             } else {
                                                 eprintln!("[pycrust-worker] Failed to loan response slice");
                                             }
                                         } else {
                                             eprintln!("[pycrust-worker] Response too large: {}", serial_buf.len());
                                         }
                                    }
                                } else {
                                    // Notification: consume result but send nothing
                                    if let Err(e) = result {
                                         // Log error for notifications since caller won't see it
                                         eprintln!("[pycrust-worker] Notification error: {}", e);
                                    }
                                }
                            },
                            Err(e) => {
                                eprintln!("[pycrust-worker] Failed to convert payload: {}", e);
                                
                                if req.id != 0 {
                                    // 1. Prepare error message
                                    let error_msg = format!("Invalid arguments: {}", e);
                                    let error_bytes = rmp_serde::to_vec_named(&error_msg).unwrap_or_default();
                                    
                                    // 2. Serialize response to buffer
                                    serial_buf.clear();
                                    let response_struct = Response {
                                        id: req.id,
                                        status: status::INVALID_PARAMS,
                                        payload: &error_bytes,
                                    };

                                    // 3. Send error response so client doesn't hang
                                    if let Ok(_) = rmp_serde::encode::write_named(&mut serial_buf, &response_struct) {
                                         if serial_buf.len() <= MAX_MESSAGE_SIZE {
                                             if let Ok(sample) = active_request.loan_slice_uninit(serial_buf.len()) {
                                                 let sample = sample.write_from_slice(&serial_buf);
                                                 let _ = sample.send();
                                             }
                                         }
                                    }
                                }
                            }
                        }
                    },
                    Err(e) => {
                         eprintln!("[pycrust-worker] Failed to deserialize request header: {}", e);
                    }
                }
                
                idle_count = 0;
            }
            Ok(None) => {
                idle_count = idle_count.saturating_add(1);
             
                if idle_count < 100_000 {
                   std::hint::spin_loop();
                } else {
                   // ✅ OPTIMIZATION: Release GIL and Sleep
                   // If we are truly idle, release GIL so other Python threads can run,
                   // and sleep to save CPU.
                   py.allow_threads(|| {
                       std::thread::sleep(std::time::Duration::from_micros(50));
                   });
                   // Don't reset idle count so we continue to sleep until activity?
                   // No, we should probably loop in sleep/check. 
                   // But `server.receive()` is fast check. 
                   // Effectively we enter a "sleepy polling" mode.
                }
            }
            Err(e) => {
                eprintln!("[pycrust-worker] Receive error: {}", e);
            }
        }
    }
}

#[derive(Deserialize)]
struct Request<'a> {
    id: u64,
    method: &'a str,
    #[serde(with = "serde_bytes")]
    payload: &'a [u8],
}

#[derive(Serialize)]
struct Response<'a> {
    id: u64,
    status: u8,
    #[serde(with = "serde_bytes")]
    payload: &'a [u8],
}


