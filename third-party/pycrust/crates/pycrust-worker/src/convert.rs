//! Conversion utilities between MessagePack and Python objects using pythonize.

use pyo3::prelude::*;
use pythonize::{depythonize, pythonize};

/// Convert MessagePack bytes to a Python object.
///
/// Uses rmpv for MessagePack deserialization and pythonize for conversion to Python.
pub fn msgpack_to_pyobject(py: Python<'_>, data: &[u8]) -> PyResult<PyObject> {
    // First deserialize MessagePack to rmpv::Value
    let value: rmpv::Value = rmp_serde::from_slice(data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid msgpack: {}", e)))?;

    // Convert rmpv::Value to serde_json::Value for pythonize compatibility
    let json_value = rmpv_to_json(&value);

    // Use pythonize to convert to Python object
    pythonize(py, &json_value)
        .map(|bound| bound.unbind())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Pythonize error: {}", e)))
}

/// Convert a Python object to MessagePack bytes.
///
/// Uses pythonize for Python to Rust conversion and rmp-serde for MessagePack serialization.
pub fn pyobject_to_msgpack(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    // Use depythonize to convert Python object to serde_json::Value
    let json_value: serde_json::Value = depythonize(obj)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Depythonize error: {}", e)))?;

    // Convert to rmpv::Value and serialize
    let rmpv_value = json_to_rmpv(json_value);

    rmp_serde::to_vec_named(&rmpv_value)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Serialization error: {}", e)))
}

/// Convert rmpv::Value to serde_json::Value.
fn rmpv_to_json(value: &rmpv::Value) -> serde_json::Value {
    match value {
        rmpv::Value::Nil => serde_json::Value::Null,
        rmpv::Value::Boolean(b) => serde_json::Value::Bool(*b),
        rmpv::Value::Integer(i) => {
            if let Some(n) = i.as_i64() {
                serde_json::Value::Number(n.into())
            } else if let Some(n) = i.as_u64() {
                serde_json::Value::Number(n.into())
            } else {
                serde_json::Value::Null
            }
        }
        rmpv::Value::F32(f) => {
            serde_json::Number::from_f64(*f as f64)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null)
        }
        rmpv::Value::F64(f) => {
            serde_json::Number::from_f64(*f)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null)
        }
        rmpv::Value::String(s) => {
            serde_json::Value::String(s.as_str().unwrap_or("").to_string())
        }
        rmpv::Value::Binary(b) => {
            // Encode binary as base64 string for JSON compatibility
            use base64::Engine;
            let encoded = base64::engine::general_purpose::STANDARD.encode(b);
            serde_json::Value::String(encoded)
        }
        rmpv::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(rmpv_to_json).collect())
        }
        rmpv::Value::Map(map) => {
            let obj: serde_json::Map<String, serde_json::Value> = map
                .iter()
                .filter_map(|(k, v)| {
                    let key = match k {
                        rmpv::Value::String(s) => s.as_str().map(|s| s.to_string()),
                        _ => Some(format!("{}", k)),
                    };
                    key.map(|k| (k, rmpv_to_json(v)))
                })
                .collect();
            serde_json::Value::Object(obj)
        }
        rmpv::Value::Ext(_, data) => {
            // Encode extension data as base64
            use base64::Engine;
            let encoded = base64::engine::general_purpose::STANDARD.encode(data);
            serde_json::Value::String(encoded)
        }
    }
}

/// Convert serde_json::Value to rmpv::Value.
fn json_to_rmpv(value: serde_json::Value) -> rmpv::Value {
    match value {
        serde_json::Value::Null => rmpv::Value::Nil,
        serde_json::Value::Bool(b) => rmpv::Value::Boolean(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                rmpv::Value::Integer(i.into())
            } else if let Some(u) = n.as_u64() {
                rmpv::Value::Integer(u.into())
            } else if let Some(f) = n.as_f64() {
                rmpv::Value::F64(f)
            } else {
                rmpv::Value::Nil
            }
        }
        serde_json::Value::String(s) => rmpv::Value::String(s.into()),
        serde_json::Value::Array(arr) => {
            rmpv::Value::Array(arr.into_iter().map(json_to_rmpv).collect())
        }
        serde_json::Value::Object(obj) => {
            rmpv::Value::Map(
                obj.into_iter()
                    .map(|(k, v)| (rmpv::Value::String(k.into()), json_to_rmpv(v)))
                    .collect(),
            )
        }
    }
}
