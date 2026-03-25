wit_bindgen::generate!({
    path: "wit",
    world: "importer",
    generate_all,
});

pub use self::microbench::host_echo::host_echo::echo;
