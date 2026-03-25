wit_bindgen::generate!({
    path: "wit",
    world: "importer",
    generate_all,
});

pub use self::microbench::resecho_callee::resecho::{Dummy, resecho};
