wit_bindgen::generate!({
    path: "wit",
    world: "importer",
    generate_all,
});

pub use self::microbench::snapshot_benchmark::snapshot_benchmark::run_benchmark;
