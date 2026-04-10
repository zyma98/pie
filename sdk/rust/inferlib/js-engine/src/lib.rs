use boa_engine::{Context, Source};
inferlib_macros::component!();

pub(crate) fn execute(code: String) -> String {
    let mut context = Context::default();
    match context.eval(Source::from_bytes(code.as_bytes())) {
        Ok(res) => res
            .to_string(&mut context)
            .unwrap_or_else(|_| "undefined".into())
            .to_std_string()
            .unwrap(),
        Err(e) => format!("Execution Error: {}", e),
    }
}
