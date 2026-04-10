use inferlib_macros::rc_resource;
use minijinja::Environment;

inferlib_macros::component!();

pub(crate) struct TemplateRenderer {
    env: Environment<'static>,
    name: String,
}

#[rc_resource]
impl TemplateRenderer {
    pub(crate) fn new(name: String, template_str: String) -> Self {
        let mut env = Environment::new();
        env.add_template_owned(name.clone(), template_str)
            .expect("invalid template");
        TemplateRenderer { env, name }
    }

    pub(crate) fn render(&self, json_data: String) -> Result<String, String> {
        let data: serde_json::Value =
            serde_json::from_str(&json_data).map_err(|e| format!("JSON parse error: {}", e))?;

        let tmpl = self.env.get_template(&self.name).unwrap();
        tmpl.render(&data)
            .map_err(|e| format!("Template rendering error: {}", e))
    }
}
