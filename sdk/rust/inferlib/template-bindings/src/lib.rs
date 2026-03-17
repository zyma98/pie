wit_bindgen::generate!({
    path: "wit",
    world: "importer",
    generate_all,
});

pub use self::inferlib::template::template_rendering::TemplateRenderer;
