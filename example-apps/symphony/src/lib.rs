use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

/// The attribute macro: #[symphony::main]
#[proc_macro_attribute]
pub fn main(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Parse the input tokens into a Rust AST (Abstract Syntax Tree).
    // We expect the user to write: `async fn name(...) -> ... { ... }`
    let mut ast = parse_macro_input!(item as ItemFn);

    // Ensure the function is actually `async` (you can provide a more detailed error message).
    if ast.sig.asyncness.is_none() {
        return syn::Error::new_spanned(
            &ast.sig.fn_token,
            "`#[symphony::main]` requires an async function"
        )
            .to_compile_error()
            .into();
    }

    // Rename the user’s function from e.g. `fn main` to `fn async_main`.
    // This is just an example rename; adjust as needed.
    let old_name = ast.sig.ident.clone();
    let new_name = syn::Ident::new("async_main", old_name.span());
    ast.sig.ident = new_name;

    // Pull out things we need to insert back in a quote! block.
    let vis = &ast.vis;           // e.g. `pub` or nothing
    let sig = &ast.sig;           // The function signature
    let block = &ast.block;       // The body `{ ... }`

    // Generate the expanded code.
    //
    // We'll insert:
    // 1) `wit_bindgen::generate!` call
    // 2) `use crate::exports::spi::app::run::Guest;`
    // 3) A struct `App` that implements `Guest::run()`, which
    //    launches our newly renamed `async_main` on a Tokio runtime.
    // 4) Reinsert the user’s async function (renamed to `async_main`)
    // 5) `export!(App);`
    //
    // The user’s original `async fn main(...) { ... }` gets replaced with
    // `async fn async_main(...) { ... }`.
    let generated = quote! {
        // 1) Generate WIT code
        wit_bindgen::generate!({
            path: "../../api/app/wit",
            world: "app",
            generate_all,
        });

        // 2) Use statements
        use crate::exports::spi::app::run::Guest;

        // 3) The App struct + Guest impl

        struct App;
        
        impl exports::spi::app::run::Guest for App {
            fn run() -> core::result::Result<(), String> {
                let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        
                // Run the async main function inside the runtime
                let result = runtime.block_on(async_main());
                
                if let Err(e) = result {
                    return Err(format!("{:?}", e));
                }
                
                Ok(())
            }
        }

        // 4) The user’s async function (renamed to `async_main`)
        #vis #sig #block

        // 5) Finally, export the App as the WASI entry point
        export!(App);
    };

    // Hand the output tokens back to the compiler
    generated.into()
}