wit_bindgen::generate!({
    path: "wit",
    world: "resecho-provider",
    generate_all,
});

struct Component;

export!(Component);

struct DummyImpl;

impl exports::microbench::resecho_callee::resecho::GuestDummy for DummyImpl {
    fn new() -> Self {
        DummyImpl
    }
}

impl exports::microbench::resecho_callee::resecho::Guest for Component {
    type Dummy = DummyImpl;

    fn resecho(
        s: String,
        _resources: Vec<exports::microbench::resecho_callee::resecho::DummyBorrow<'_>>,
    ) -> String {
        s
    }
}
