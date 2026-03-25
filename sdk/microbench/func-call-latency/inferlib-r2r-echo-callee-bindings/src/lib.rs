wit_bindgen::generate!({
    path: "wit",
    world: "importer",
    generate_all,
});

pub use self::microbench::echo_callee::echo::echo;
