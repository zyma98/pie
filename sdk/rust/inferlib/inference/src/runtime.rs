use crate::InferenceComponentImpl;
use inferlib_macros::guest_interface;

#[guest_interface]
impl InferenceComponentImpl {
    /// Returns the runtime version string.
    fn get_version() -> String {
        inferlib_engine_bindings::inferlet::core::runtime::get_version()
    }

    /// Returns a unique identifier for the running instance.
    fn get_instance_id() -> String {
        inferlib_engine_bindings::inferlet::core::runtime::get_instance_id()
    }

    /// Retrieves POSIX-style CLI arguments passed to the inferlet from the remote user client.
    fn get_arguments() -> Vec<String> {
        inferlib_engine_bindings::inferlet::core::runtime::get_arguments()
    }

    fn set_return(value: String) {
        inferlib_engine_bindings::inferlet::core::runtime::set_return(&value);
    }

    /// Get names of models that have all specified traits (e.g. "input_text", "tokenize").
    fn get_all_models_with_traits(traits: Vec<String>) -> Vec<String> {
        inferlib_engine_bindings::inferlet::core::runtime::get_all_models_with_traits(&traits)
    }

    /// Executes a debug command and returns the result as a string.
    fn debug_query(query: String) -> String {
        let future = inferlib_engine_bindings::inferlet::core::runtime::debug_query(&query);
        crate::wait_for_pollable(future.pollable());
        future.get().unwrap()
    }
}
