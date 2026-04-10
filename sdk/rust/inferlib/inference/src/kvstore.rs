/// Retrieves a value from the persistent store for a given key.
///
/// Returns `Some(value)` if the key exists, or `None` if it does not.
pub(crate) fn store_get(key: String) -> Option<String> {
    inferlib_engine_bindings::inferlet::core::kvs::store_get(&key)
}

/// Sets a value in the persistent store for a given key.
///
/// This will create a new entry or overwrite an existing one.
pub(crate) fn store_set(key: String, value: String) {
    inferlib_engine_bindings::inferlet::core::kvs::store_set(&key, &value);
}

/// Deletes a key-value pair from the store.
///
/// If the key does not exist, this function does nothing.
pub(crate) fn store_delete(key: String) {
    inferlib_engine_bindings::inferlet::core::kvs::store_delete(&key);
}

/// Checks if a key exists in the store.
pub(crate) fn store_exists(key: String) -> bool {
    inferlib_engine_bindings::inferlet::core::kvs::store_exists(&key)
}

/// Returns a list of all keys currently in the store.
pub(crate) fn store_list_keys() -> Vec<String> {
    inferlib_engine_bindings::inferlet::core::kvs::store_list_keys()
}
