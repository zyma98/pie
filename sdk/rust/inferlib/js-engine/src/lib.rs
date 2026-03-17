wit_bindgen::generate!({
    path: "wit",
    world: "js-engine-provider",
    generate_all,
});

use boa_engine::{Context, Source};
use exports::inferlib::js_engine::js_engine::Guest;

struct Component;

export!(Component);

impl Guest for Component {
    fn execute(code: String) -> String {
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
}
