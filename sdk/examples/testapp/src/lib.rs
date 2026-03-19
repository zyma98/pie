use testlib_bindings::{Model, Context};

mod __pie_export {
    ::inferlib_old_run_bindings::wit_bindgen::generate!({
        inline: r#"
package pie:testapp;

interface run {
    run: func() -> result<_, string>;
}

world inferlet {
    export run;
}
"#,
        world: "inferlet",
        pub_export_macro: true,
        runtime_path: "::inferlib_old_run_bindings::wit_bindgen::rt",
    });
}

async fn __pie_main_inner(_args: inferlib_old_run_bindings::Args) -> inferlib_old_run_bindings::Result<String> {
    let model = Model::get_auto();
    let _ctx = Context::new(&model);
    Ok("done".to_string())
}

struct __PieMain;

impl __pie_export::exports::pie::testapp::run::Guest for __PieMain {
    fn run() -> core::result::Result<(), String> {
        let result = inferlib_old_run_bindings::block_on(async {
            let args = inferlib_old_run_bindings::Args::from_vec(Vec::new());
            __pie_main_inner(args).await
        });

        match result {
            Ok(_) => {
                Ok(())
            }
            Err(e) => Err(format!("{:?}", e)),
        }
    }
}

__pie_export::export!(__PieMain with_types_in __pie_export);
