// Generate WIT bindings - import from host, export our interface
wit_bindgen::generate!({
    path: "wit",
    world: "environment-provider",
    generate_all,
});

use exports::inferlib::environment::runtime::Guest;

struct RuntimeImpl;

impl Guest for RuntimeImpl {
    fn get_version() -> String {
        inferlet::core::runtime::get_version()
    }

    fn get_instance_id() -> String {
        inferlet::core::runtime::get_instance_id()
    }

    fn get_arguments() -> Vec<String> {
        inferlet::core::runtime::get_arguments()
    }

    fn set_return(value: String) {
        inferlet::core::runtime::set_return(&value);
    }
}

export!(RuntimeImpl);
