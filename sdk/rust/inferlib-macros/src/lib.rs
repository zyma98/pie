use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

/// Attribute macro for defining the main entry point of an inferlib application.
///
/// This macro generates the boilerplate code needed to implement the `Guest` trait
/// and export the application entry point.
///
/// # Example
///
/// ```rust,no_run
/// use inferlib_run_bindings::{Args, Result};
///
/// #[inferlib_macros::main]
/// async fn main(args: Args) -> Result<String> {
///     Ok("Hello, world!".to_string())
/// }
/// ```
#[proc_macro_attribute]
pub fn main(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let mut input_fn = parse_macro_input!(item as ItemFn);
    let original_fn_name = input_fn.sig.ident.clone();
    let inner_fn_name = syn::Ident::new("__pie_main_inner", original_fn_name.span());

    // Ensure the function is async
    if input_fn.sig.asyncness.is_none() {
        return syn::Error::new_spanned(
            input_fn.sig.ident,
            "The #[inferlib_macros::main] attribute can only be used on async functions",
        )
        .to_compile_error()
        .into();
    }

    // Rename the user's function so that we can call it from our generated code.
    input_fn.sig.ident = inner_fn_name.clone();

    // Generate a wrapper type that implements `inferlib_run_bindings::Guest`.
    // It calls the inner async function and maps the error to String.
    let expanded = quote! {
        #input_fn

        struct __PieMain;

        impl ::inferlib_run_bindings::Guest for __PieMain {
            fn run() -> ::core::result::Result<(), ::std::string::String> {
                let args = ::inferlib_run_bindings::Args::from_vec(
                    ::inferlib_environment_bindings::get_arguments()
                        .into_iter()
                        .map(::std::ffi::OsString::from)
                        .collect(),
                );

                let result = ::inferlib_run_bindings::block_on(async { #inner_fn_name(args).await });

                match result {
                    ::core::result::Result::Ok(r) => {
                        let r_any: &dyn ::std::any::Any = &r;
                        let output = if let ::core::option::Option::Some(s) = r_any.downcast_ref::<::std::string::String>() {
                            s.clone()
                        } else if let ::core::option::Option::Some(s) = r_any.downcast_ref::<&str>() {
                            ::std::string::ToString::to_string(s)
                        } else {
                            // Fallback for all other types
                            ::std::format!("{:?}", r)
                        };

                        ::inferlib_environment_bindings::set_return(&output);
                        ::core::result::Result::Ok(())
                    }
                    ::core::result::Result::Err(e) => ::core::result::Result::Err(::std::format!("{:?}", e)),
                }
            }
        }

        ::inferlib_run_bindings::export!(__PieMain with_types_in ::inferlib_run_bindings);
    };

    expanded.into()
}
