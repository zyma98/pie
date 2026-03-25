wit_bindgen::generate!({
    path: "wit",
    world: "echo-provider",
    generate_all,
});

struct Component;

export!(Component);

impl exports::microbench::echo_callee::echo::Guest for Component {
    fn echo(s: String) -> String {
        s
    }
}
