wit_bindgen::generate!({
    path: "wit",
    world: "importer",
    generate_all,
});

pub use self::microbench::h2r_benchmark::h2r_benchmark::run_benchmark;
