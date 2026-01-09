//! Worker loop implementation for PyCrust.

use crate::convert::{msgpack_to_pyobject, pyobject_to_msgpack};
use iceoryx2::port::publisher::Publisher;
use iceoryx2::prelude::*;
use pyo3::prelude::*;

/// Maximum message size in bytes (64KB).
const MAX_MESSAGE_SIZE: usize = 65536;

/// Number of spin iterations per idle loop.
/// With ~10ns per spin iteration, 1000 spins ≈ 10µs of spinning.
const SPIN_ITERATIONS: u32 = 1000;

/// Number of idle loops before sleeping (spin budget).
/// After this many consecutive empty polls, we sleep briefly.
const IDLE_THRESHOLD: u32 = 100;

/// How often to check Python signals during spinning (every N idle loops).
const SIGNAL_CHECK_INTERVAL: u32 = 50;

/// Status codes for RPC responses.
mod status {
    pub const OK: u8 = 0;
    pub const METHOD_NOT_FOUND: u8 = 1;
    #[allow(dead_code)]
    pub const INVALID_PARAMS: u8 = 2;
    pub const INTERNAL_ERROR: u8 = 3;
}

/// Run the worker loop, dispatching RPC calls to the Python callback.
///
/// This function blocks and processes incoming RPC requests until interrupted.
/// The dispatch_callback should be a Python callable that takes (method: str, args: Any)
/// and returns the result.
#[pyfunction]
#[pyo3(signature = (service_name, dispatch_callback))]
pub fn run_worker(py: Python<'_>, service_name: &str, dispatch_callback: PyObject) -> PyResult<()> {
    // Create iceoryx2 node
    let node = NodeBuilder::new()
        .create::<ipc::Service>()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create node: {}", e)))?;

    // Subscribe to request topic
    let req_service_name = format!("{}_req", service_name);
    let req_service = node
        .service_builder(&req_service_name.as_str().try_into().unwrap())
        .publish_subscribe::<[u8]>()
        .open_or_create()
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to create request service: {}",
                e
            ))
        })?;

    let req_subscriber = req_service
        .subscriber_builder()
        .create()
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to create request subscriber: {}",
                e
            ))
        })?;

    // Publish to response topic
    let res_service_name = format!("{}_res", service_name);
    let res_service = node
        .service_builder(&res_service_name.as_str().try_into().unwrap())
        .publish_subscribe::<[u8]>()
        .open_or_create()
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to create response service: {}",
                e
            ))
        })?;

    let res_publisher = res_service
        .publisher_builder()
        .initial_max_slice_len(MAX_MESSAGE_SIZE)
        .create()
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to create response publisher: {}",
                e
            ))
        })?;

    eprintln!("[pycrust-worker] Listening on service: {}", service_name);

    // Track consecutive idle iterations for adaptive sleeping
    let mut idle_count: u32 = 0;

    // Main worker loop
    loop {
        // Periodically check Python signals (Ctrl+C)
        if idle_count % SIGNAL_CHECK_INTERVAL == 0 {
            py.check_signals()?;
        }

        // Poll for incoming requests
        match req_subscriber.receive() {
            Ok(Some(sample)) => {
                let payload = sample.payload();
                let (id, response_status, response_payload) =
                    process_request(py, payload, &dispatch_callback);

                // Send response
                send_response(&res_publisher, id, response_status, &response_payload);

                // Reset idle counter on activity
                idle_count = 0;
            }
            Ok(None) => {
                // No message available - use hybrid spinning
                idle_count = idle_count.saturating_add(1);

                if idle_count < IDLE_THRESHOLD {
                    // Spin-wait: release GIL and spin
                    py.allow_threads(|| {
                        for _ in 0..SPIN_ITERATIONS {
                            std::hint::spin_loop();
                        }
                    });
                } else {
                    // After many idle iterations, sleep briefly to save CPU
                    py.allow_threads(|| {
                        std::thread::sleep(std::time::Duration::from_micros(10));
                    });
                }
            }
            Err(e) => {
                eprintln!("[pycrust-worker] Receive error: {}", e);
            }
        }
    }
}

/// Process a single RPC request and return (id, status, payload).
fn process_request(
    py: Python<'_>,
    sample: &[u8],
    dispatch_callback: &PyObject,
) -> (u64, u8, Vec<u8>) {
    // Deserialize the request
    let request: rmpv::Value = match rmp_serde::from_slice(sample) {
        Ok(v) => v,
        Err(e) => {
            let error_msg = format!("Failed to deserialize request: {}", e);
            let error_bytes = rmp_serde::to_vec_named(&error_msg).unwrap_or_default();
            return (0, status::INTERNAL_ERROR, error_bytes);
        }
    };

    // Extract fields from the request
    let id = extract_u64(&request, "id").unwrap_or(0);
    let method = match extract_string(&request, "method") {
        Ok(m) => m,
        Err(e) => {
            let error_bytes = rmp_serde::to_vec_named(&e).unwrap_or_default();
            return (id, status::INTERNAL_ERROR, error_bytes);
        }
    };
    let payload_bytes = match extract_bytes(&request, "payload") {
        Ok(p) => p,
        Err(e) => {
            let error_bytes = rmp_serde::to_vec_named(&e).unwrap_or_default();
            return (id, status::INTERNAL_ERROR, error_bytes);
        }
    };

    // Convert payload to Python object
    let py_args = match msgpack_to_pyobject(py, &payload_bytes) {
        Ok(args) => args,
        Err(e) => {
            let error_msg = format!("Failed to convert payload: {}", e);
            let error_bytes = rmp_serde::to_vec_named(&error_msg).unwrap_or_default();
            return (id, status::INTERNAL_ERROR, error_bytes);
        }
    };

    // Call the dispatch callback: dispatch_callback(method, args) -> result
    let result = dispatch_callback.call1(py, (method.as_str(), py_args));

    match result {
        Ok(py_result) => {
            match pyobject_to_msgpack(py_result.bind(py)) {
                Ok(response_bytes) => (id, status::OK, response_bytes),
                Err(e) => {
                    let error_msg = format!("Failed to serialize result: {}", e);
                    let error_bytes = rmp_serde::to_vec_named(&error_msg).unwrap_or_default();
                    (id, status::INTERNAL_ERROR, error_bytes)
                }
            }
        }
        Err(e) => {
            // Check if it's a "method not found" error
            let error_str = e.to_string();
            let (status_code, error_msg) = if error_str.contains("Method not found") {
                (status::METHOD_NOT_FOUND, error_str)
            } else {
                (status::INTERNAL_ERROR, error_str)
            };
            let error_bytes = rmp_serde::to_vec_named(&error_msg).unwrap_or_default();
            (id, status_code, error_bytes)
        }
    }
}

/// Send a response back to the client.
fn send_response(
    publisher: &Publisher<ipc::Service, [u8], ()>,
    id: u64,
    status: u8,
    payload: &[u8],
) {
    // Build response as MessagePack
    let response = rmpv::Value::Map(vec![
        (
            rmpv::Value::String("id".into()),
            rmpv::Value::Integer(id.into()),
        ),
        (
            rmpv::Value::String("status".into()),
            rmpv::Value::Integer(status.into()),
        ),
        (
            rmpv::Value::String("payload".into()),
            rmpv::Value::Binary(payload.to_vec()),
        ),
    ]);

    if let Ok(encoded) = rmp_serde::to_vec_named(&response) {
        if let Ok(sample) = publisher.loan_slice_uninit(encoded.len()) {
            let sample = sample.write_from_slice(&encoded);
            let _ = sample.send();
        }
    }
}

/// Extract a u64 value from a MessagePack map.
fn extract_u64(value: &rmpv::Value, key: &str) -> Result<u64, String> {
    if let rmpv::Value::Map(map) = value {
        for (k, v) in map {
            if let rmpv::Value::String(s) = k {
                if s.as_str() == Some(key) {
                    if let rmpv::Value::Integer(i) = v {
                        if let Some(n) = i.as_u64() {
                            return Ok(n);
                        }
                        if let Some(n) = i.as_i64() {
                            return Ok(n as u64);
                        }
                    }
                }
            }
        }
    }
    Err(format!("Missing or invalid field: {}", key))
}

/// Extract a String value from a MessagePack map.
fn extract_string(value: &rmpv::Value, key: &str) -> Result<String, String> {
    if let rmpv::Value::Map(map) = value {
        for (k, v) in map {
            if let rmpv::Value::String(s) = k {
                if s.as_str() == Some(key) {
                    if let rmpv::Value::String(s) = v {
                        if let Some(str_val) = s.as_str() {
                            return Ok(str_val.to_string());
                        }
                    }
                }
            }
        }
    }
    Err(format!("Missing or invalid field: {}", key))
}

/// Extract a bytes value from a MessagePack map.
fn extract_bytes(value: &rmpv::Value, key: &str) -> Result<Vec<u8>, String> {
    if let rmpv::Value::Map(map) = value {
        for (k, v) in map {
            if let rmpv::Value::String(s) = k {
                if s.as_str() == Some(key) {
                    if let rmpv::Value::Binary(b) = v {
                        return Ok(b.clone());
                    }
                }
            }
        }
    }
    Err(format!("Missing or invalid field: {}", key))
}
