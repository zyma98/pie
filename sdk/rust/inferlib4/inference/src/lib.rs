mod brle;
mod chat;
mod context;
mod formatter;
mod models;
mod queues;

wit_bindgen::generate!({
    path: "wit",
    world: "inference-provider",
    generate_all,
});

use context::ContextImpl;
use formatter::ChatFormatterImpl;
use models::{ModelImpl, TokenizerImpl};
use queues::{ForwardPassImpl, QueueImpl};

use exports::inferlib4::inference::formatter::Guest as FormatterGuest;
use exports::inferlib4::inference::inference::Guest as InferenceGuest;
use exports::inferlib4::inference::kvstore::Guest as KvstoreGuest;
use exports::inferlib4::inference::messaging::Guest as MessagingGuest;
use exports::inferlib4::inference::models::Guest as ModelsGuest;
use exports::inferlib4::inference::queues::Guest as QueuesGuest;
use exports::inferlib4::inference::runtime::Guest as RuntimeGuest;

use wstd::runtime::{AsyncPollable, block_on};

struct InferenceComponentImpl;

impl ModelsGuest for InferenceComponentImpl {
    type Model = ModelImpl;
    type Tokenizer = TokenizerImpl;
}

impl QueuesGuest for InferenceComponentImpl {
    type Queue = QueueImpl;
    type ForwardPass = ForwardPassImpl;
}

impl RuntimeGuest for InferenceComponentImpl {
    /// Returns the runtime version string.
    fn get_version() -> String {
        inferlib4_engine_bindings::inferlet::core::runtime::get_version()
    }

    /// Returns a unique identifier for the running instance.
    fn get_instance_id() -> String {
        inferlib4_engine_bindings::inferlet::core::runtime::get_instance_id()
    }

    /// Retrieves POSIX-style CLI arguments passed to the inferlet from the remote user client.
    fn get_arguments() -> Vec<String> {
        inferlib4_engine_bindings::inferlet::core::runtime::get_arguments()
    }

    fn set_return(value: String) {
        inferlib4_engine_bindings::inferlet::core::runtime::set_return(&value);
    }

    /// Get names of models that have all specified traits (e.g. "input_text", "tokenize").
    fn get_all_models_with_traits(traits: Vec<String>) -> Vec<String> {
        inferlib4_engine_bindings::inferlet::core::runtime::get_all_models_with_traits(&traits)
    }

    /// Executes a debug command and returns the result as a string.
    fn debug_query(query: String) -> String {
        let future = inferlib4_engine_bindings::inferlet::core::runtime::debug_query(&query);
        let pollable = future.pollable();
        block_on(async {
            AsyncPollable::new(pollable).wait_for().await;
        });
        future.get().unwrap()
    }
}

impl InferenceGuest for InferenceComponentImpl {
    type Context = ContextImpl;
}

impl FormatterGuest for InferenceComponentImpl {
    type ChatFormatter = ChatFormatterImpl;
}

impl MessagingGuest for InferenceComponentImpl {
    /// Sends a message to the remote user client.
    fn send(message: String) {
        inferlib4_engine_bindings::inferlet::core::message::send(&message);
    }

    /// Receives an incoming message from the remote user client.
    fn receive() -> String {
        let future = inferlib4_engine_bindings::inferlet::core::message::receive();
        let pollable = future.pollable();
        block_on(async {
            AsyncPollable::new(pollable).wait_for().await;
        });
        future.get().unwrap()
    }

    /// Sends a blob to the remote user client.
    fn send_blob(data: Vec<u8>) {
        use inferlib4_engine_bindings::inferlet::core::common::Blob;
        let blob = Blob::new(&data);
        inferlib4_engine_bindings::inferlet::core::message::send_blob(blob);
    }

    /// Receives an incoming blob from the remote user client.
    fn receive_blob() -> Vec<u8> {
        let future = inferlib4_engine_bindings::inferlet::core::message::receive_blob();
        let pollable = future.pollable();
        block_on(async {
            AsyncPollable::new(pollable).wait_for().await;
        });
        let blob = future.get().unwrap();
        blob.read(0, blob.size())
    }

    /// Publishes a message to a topic, broadcasting it to all subscribers.
    fn broadcast(topic: String, message: String) {
        inferlib4_engine_bindings::inferlet::core::message::broadcast(&topic, &message);
    }

    /// Subscribes to a topic and waits for the next message.
    fn subscribe(topic: String) -> String {
        let subscription = inferlib4_engine_bindings::inferlet::core::message::subscribe(&topic);
        let pollable = subscription.pollable();
        block_on(async {
            AsyncPollable::new(pollable).wait_for().await;
        });
        subscription.get().unwrap()
    }
}

impl KvstoreGuest for InferenceComponentImpl {
    /// Retrieves a value from the persistent store for a given key.
    ///
    /// Returns `Some(value)` if the key exists, or `None` if it does not.
    fn store_get(key: String) -> Option<String> {
        inferlib4_engine_bindings::inferlet::core::kvs::store_get(&key)
    }

    /// Sets a value in the persistent store for a given key.
    ///
    /// This will create a new entry or overwrite an existing one.
    fn store_set(key: String, value: String) {
        inferlib4_engine_bindings::inferlet::core::kvs::store_set(&key, &value);
    }

    /// Deletes a key-value pair from the store.
    ///
    /// If the key does not exist, this function does nothing.
    fn store_delete(key: String) {
        inferlib4_engine_bindings::inferlet::core::kvs::store_delete(&key);
    }

    /// Checks if a key exists in the store.
    fn store_exists(key: String) -> bool {
        inferlib4_engine_bindings::inferlet::core::kvs::store_exists(&key)
    }

    /// Returns a list of all keys currently in the store.
    fn store_list_keys() -> Vec<String> {
        inferlib4_engine_bindings::inferlet::core::kvs::store_list_keys()
    }
}

export!(InferenceComponentImpl);
