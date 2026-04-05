use boa_engine::{Context, Source};

/// Executes the given JavaScript code using the Boa engine.
pub fn execute_js_code(code: &str) -> String {
    let mut context = Context::default();
    match context.eval(Source::from_bytes(code)) {
        Ok(res) => res
            .to_string(&mut context)
            .unwrap_or_else(|_| "undefined".into())
            .to_std_string()
            .unwrap(),
        Err(e) => format!("Execution Error: {}", e),
    }
}
