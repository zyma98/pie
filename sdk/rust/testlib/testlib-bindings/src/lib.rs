wit_bindgen::generate!({
    path: "wit",
    world: "importer",
    generate_all,
});

pub use self::testlib::testlib::models::Model;
pub use self::testlib::testlib::inference::Context;
