wit_bindgen::generate!({
    path: "wit",
    world: "schema-provider",
    generate_all,
});

use exports::inferlib::schema::json_schema::{Guest, GuestSchemaValidator};

struct Component;

export!(Component);

impl Guest for Component {
    type SchemaValidator = SchemaValidatorImpl;
}

pub struct SchemaValidatorImpl {
    validator: jsonschema::Validator,
}

impl GuestSchemaValidator for SchemaValidatorImpl {
    fn new(schema_str: String) -> Self {
        let schema: serde_json::Value =
            serde_json::from_str(&schema_str).expect("invalid schema JSON");
        let validator = jsonschema::validator_for(&schema).expect("invalid JSON Schema");
        SchemaValidatorImpl { validator }
    }

    fn validate(&self, json_str: String) -> Result<String, String> {
        let value: serde_json::Value =
            serde_json::from_str(&json_str).map_err(|e| format!("JSON parse error: {}", e))?;

        let errors: Vec<String> = self
            .validator
            .iter_errors(&value)
            .map(|e| format!("- {}", e))
            .collect();

        if errors.is_empty() {
            Ok(serde_json::to_string_pretty(&value).unwrap_or(json_str))
        } else {
            Err(errors.join("\n"))
        }
    }
}
