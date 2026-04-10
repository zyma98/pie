/// Sends a message to the remote user client.
pub(crate) fn send(message: String) {
    inferlib_engine_bindings::inferlet::core::message::send(&message);
}

/// Receives an incoming message from the remote user client.
pub(crate) fn receive() -> String {
    let future = inferlib_engine_bindings::inferlet::core::message::receive();
    crate::wait_for_pollable(future.pollable());
    future.get().unwrap()
}

/// Sends a blob to the remote user client.
pub(crate) fn send_blob(data: Vec<u8>) {
    use inferlib_engine_bindings::inferlet::core::common::Blob;

    let blob = Blob::new(&data);
    inferlib_engine_bindings::inferlet::core::message::send_blob(blob);
}

/// Receives an incoming blob from the remote user client.
pub(crate) fn receive_blob() -> Vec<u8> {
    let future = inferlib_engine_bindings::inferlet::core::message::receive_blob();
    crate::wait_for_pollable(future.pollable());
    let blob = future.get().unwrap();
    blob.read(0, blob.size())
}

/// Publishes a message to a topic, broadcasting it to all subscribers.
pub(crate) fn broadcast(topic: String, message: String) {
    inferlib_engine_bindings::inferlet::core::message::broadcast(&topic, &message);
}

/// Subscribes to a topic and waits for the next message.
pub(crate) fn subscribe(topic: String) -> String {
    let subscription = inferlib_engine_bindings::inferlet::core::message::subscribe(&topic);
    crate::wait_for_pollable(subscription.pollable());
    subscription.get().unwrap()
}
