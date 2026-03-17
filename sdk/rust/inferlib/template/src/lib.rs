wit_bindgen::generate!({
    path: "wit",
    world: "template-provider",
    generate_all,
});

use exports::inferlib::template::template_rendering::{Guest, GuestTemplateRenderer};
use minijinja::Environment;

struct Component;

export!(Component);

impl Guest for Component {
    type TemplateRenderer = TemplateRendererImpl;
}

pub struct TemplateRendererImpl {
    env: Environment<'static>,
    name: String,
}

impl GuestTemplateRenderer for TemplateRendererImpl {
    fn new(name: String, template_str: String) -> Self {
        let mut env = Environment::new();
        env.add_template_owned(name.clone(), template_str)
            .expect("invalid template");
        TemplateRendererImpl { env, name }
    }

    fn render(&self, json_data: String) -> Result<String, String> {
        let data: serde_json::Value =
            serde_json::from_str(&json_data).map_err(|e| format!("JSON parse error: {}", e))?;

        let tmpl = self.env.get_template(&self.name).unwrap();
        tmpl.render(&data)
            .map_err(|e| format!("Template rendering error: {}", e))
    }
}
