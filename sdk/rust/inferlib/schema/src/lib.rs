use inferlib_macros::rc_resource;

inferlib_macros::component!();

pub(crate) struct SchemaValidator {
    validator: jsonschema::Validator,
}

#[rc_resource]
impl SchemaValidator {
    pub(crate) fn new(schema_str: String) -> Self {
        let schema: serde_json::Value =
            serde_json::from_str(&schema_str).expect("invalid schema JSON");
        let validator = jsonschema::validator_for(&schema).expect("invalid JSON Schema");
        SchemaValidator { validator }
    }

    pub(crate) fn validate(&self, json_str: String) -> Result<String, String> {
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
