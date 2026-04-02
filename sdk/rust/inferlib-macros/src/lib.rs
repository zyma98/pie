use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{
    braced, parse_macro_input, Error, Ident, ItemEnum, ItemFn, Path, Result, Token, Type,
};

fn manifest_dir() -> std::result::Result<std::path::PathBuf, String> {
    std::env::var("CARGO_MANIFEST_DIR")
        .map(std::path::PathBuf::from)
        .map_err(|_| "CARGO_MANIFEST_DIR not set".to_string())
}

fn read_package_name() -> std::result::Result<String, String> {
    let manifest_dir = manifest_dir()?;
    let pie_toml_path = manifest_dir.join("Pie.toml");
    let pie_toml_content = std::fs::read_to_string(&pie_toml_path).map_err(|_| {
        "Failed to read Pie.toml - make sure it exists next to Cargo.toml".to_string()
    })?;

    let pie_config: toml::Value = pie_toml_content
        .parse()
        .map_err(|e| format!("Failed to parse Pie.toml: {e}"))?;

    pie_config["package"]["name"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| "Missing [package].name in Pie.toml".to_string())
}

fn to_rust_ident(name: &str) -> syn::Ident {
    let sanitized = name.replace('-', "_");
    syn::Ident::new(&sanitized, Span::call_site())
}

fn read_wit_exports_path() -> std::result::Result<Path, String> {
    let manifest_dir = manifest_dir()?;
    let world_wit_path = manifest_dir.join("wit/world.wit");
    let world_wit = std::fs::read_to_string(&world_wit_path).map_err(|_| {
        "Failed to read wit/world.wit - make sure it exists next to Cargo.toml".to_string()
    })?;

    let package_line = world_wit
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("package ") && line.ends_with(';'))
        .ok_or_else(|| "Failed to find `package ...;` in wit/world.wit".to_string())?;

    let package = package_line
        .trim_start_matches("package ")
        .trim_end_matches(';');
    let (namespace, name) = package
        .split_once(':')
        .ok_or_else(|| format!("Invalid WIT package `{package}` in wit/world.wit"))?;

    let exports_path = format!(
        "crate::exports::{}::{}",
        namespace.replace('-', "_"),
        name.replace('-', "_")
    );
    syn::parse_str(&exports_path)
        .map_err(|e| format!("Failed to build exports path from `{package}`: {e}"))
}

fn to_upper_camel(name: &str) -> String {
    let mut result = String::new();
    for part in name.split(['-', '_']) {
        if part.is_empty() {
            continue;
        }
        let mut chars = part.chars();
        if let Some(first) = chars.next() {
            result.push(first.to_ascii_uppercase());
            result.extend(chars);
        }
    }
    result
}

fn parse_world_exports() -> std::result::Result<Vec<String>, String> {
    let manifest_dir = manifest_dir()?;
    let world_wit_path = manifest_dir.join("wit/world.wit");
    let world_wit = std::fs::read_to_string(&world_wit_path).map_err(|_| {
        "Failed to read wit/world.wit - make sure it exists next to Cargo.toml".to_string()
    })?;

    Ok(world_wit
        .lines()
        .map(str::trim)
        .filter_map(|line| {
            line.strip_prefix("export ")
                .and_then(|rest| rest.strip_suffix(';'))
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .map(ToOwned::to_owned)
        })
        .collect())
}

fn parse_interface_resources(interface: &str) -> std::result::Result<Vec<String>, String> {
    let manifest_dir = manifest_dir()?;
    let interface_wit_path = manifest_dir.join("wit").join(format!("{interface}.wit"));
    let interface_wit = std::fs::read_to_string(&interface_wit_path).map_err(|_| {
        format!(
            "Failed to read wit/{interface}.wit - make sure it exists next to Cargo.toml"
        )
    })?;

    Ok(interface_wit
        .lines()
        .map(str::trim)
        .filter_map(|line| {
            line.strip_prefix("resource ")
                .and_then(|rest| rest.split_once('{').map(|(name, _)| name.trim()))
                .filter(|name| !name.is_empty())
                .map(ToOwned::to_owned)
        })
        .collect())
}

fn infer_component_bindings_from_wit() -> std::result::Result<Vec<InterfaceBindings>, String> {
    let mut interfaces = Vec::new();
    for interface in parse_world_exports()? {
        let resources = parse_interface_resources(&interface)?;
        if resources.is_empty() {
            continue;
        }

        let bindings = resources
            .into_iter()
            .map(|resource| {
                let assoc_name = to_upper_camel(&resource);
                let impl_name = format!("{assoc_name}Impl");
                let name = Ident::new(&assoc_name, Span::call_site());
                let ty = syn::parse_str::<Type>(&impl_name)
                    .map_err(|e| format!("Failed to build inferred type `{impl_name}`: {e}"))?;
                Ok(AssociatedTypeBinding { name, ty })
            })
            .collect::<std::result::Result<Vec<_>, String>>()?;

        interfaces.push(InterfaceBindings {
            interface: Ident::new(&interface, Span::call_site()),
            bindings,
        });
    }

    Ok(interfaces)
}

struct WitEnumInput {
    interface: String,
    name: Option<String>,
}

impl Parse for WitEnumInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let mut interface = None;
        let mut name = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            let value: syn::LitStr = input.parse()?;
            match key.to_string().as_str() {
                "interface" => interface = Some(value.value()),
                "name" => name = Some(value.value()),
                other => {
                    return Err(Error::new(
                        key.span(),
                        format!("unsupported wit_enum argument `{other}`"),
                    ));
                }
            }

            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        let interface =
            interface.ok_or_else(|| Error::new(Span::call_site(), "missing `interface = \"...\"`"))?;

        Ok(Self { interface, name })
    }
}

#[proc_macro_attribute]
pub fn main(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut input_fn = parse_macro_input!(item as ItemFn);
    let original_fn_name = input_fn.sig.ident.clone();
    let inner_fn_name = syn::Ident::new("__pie_main_inner", original_fn_name.span());

    if input_fn.sig.asyncness.is_none() {
        return syn::Error::new_spanned(
            input_fn.sig.ident,
            "The #[inferlib_macros::main] attribute can only be used on async functions",
        )
        .to_compile_error()
        .into();
    }

    let package_name = match read_package_name() {
        Ok(name) => name,
        Err(e) => {
            return syn::Error::new(Span::call_site(), e)
                .to_compile_error()
                .into();
        }
    };
    let package_ident = to_rust_ident(&package_name);

    let export_wit = format!(
        r#"
package pie:{package_name};

interface run {{
    run: func() -> result<_, string>;
}}

world inferlet {{
    export run;
}}
"#
    );

    input_fn.sig.ident = inner_fn_name.clone();

    let expanded = quote! {
        mod __pie_export {
            ::inferlib_run_bindings::wit_bindgen::generate!({
                inline: #export_wit,
                world: "inferlet",
                pub_export_macro: true,
                runtime_path: "::inferlib_run_bindings::wit_bindgen::rt",
            });
        }

        #input_fn

        struct __PieMain;

        impl __pie_export::exports::pie::#package_ident::run::Guest for __PieMain {
            fn run() -> ::core::result::Result<(), ::std::string::String> {
                let args = ::inferlib_run_bindings::Args::from_vec(
                    ::inferlib_inference_bindings::get_arguments()
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
                            ::std::format!("{:?}", r)
                        };

                        ::inferlib_inference_bindings::set_return(&output);
                        ::core::result::Result::Ok(())
                    }
                    ::core::result::Result::Err(e) => ::core::result::Result::Err(::std::format!("{:?}", e)),
                }
            }
        }

        __pie_export::export!(__PieMain with_types_in __pie_export);
    };

    expanded.into()
}

#[proc_macro_attribute]
pub fn wit_enum(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as WitEnumInput);
    let item_enum = parse_macro_input!(item as ItemEnum);

    if item_enum
        .variants
        .iter()
        .any(|variant| !matches!(variant.fields, syn::Fields::Unit))
    {
        return Error::new_spanned(
            &item_enum.ident,
            "#[inferlib_macros::wit_enum] currently supports only unit enums",
        )
        .to_compile_error()
        .into();
    }

    let exports = match read_wit_exports_path() {
        Ok(path) => path,
        Err(message) => {
            return Error::new(Span::call_site(), message)
                .to_compile_error()
                .into();
        }
    };

    let enum_ident = item_enum.ident.clone();
    let interface_ident = to_rust_ident(&args.interface);
    let wit_type_ident = Ident::new(
        &args
            .name
            .as_deref()
            .map(to_upper_camel)
            .unwrap_or_else(|| enum_ident.to_string()),
        Span::call_site(),
    );
    let variants = item_enum
        .variants
        .iter()
        .map(|variant| variant.ident.clone())
        .collect::<Vec<_>>();

    quote! {
        #item_enum

        impl ::core::convert::From<#enum_ident> for #exports::#interface_ident::#wit_type_ident {
            fn from(value: #enum_ident) -> Self {
                match value {
                    #( #enum_ident::#variants => Self::#variants, )*
                }
            }
        }

        impl ::core::convert::From<#exports::#interface_ident::#wit_type_ident> for #enum_ident {
            fn from(value: #exports::#interface_ident::#wit_type_ident) -> Self {
                match value {
                    #( #exports::#interface_ident::#wit_type_ident::#variants => Self::#variants, )*
                }
            }
        }
    }
    .into()
}

struct ComponentBindingsInput {
    component: Ident,
    exports: Path,
    interfaces: Vec<InterfaceBindings>,
}

struct InterfaceBindings {
    interface: Ident,
    bindings: Vec<AssociatedTypeBinding>,
}

struct AssociatedTypeBinding {
    name: Ident,
    ty: Type,
}

impl Parse for ComponentBindingsInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let fork = input.fork();
        let first_ident: Ident = fork.parse()?;
        if first_ident == "component" && fork.peek(Token![:]) {
            return parse_component_bindings_legacy(input);
        }
        parse_component_bindings_shorthand(input)
    }
}

fn parse_component_bindings_legacy(input: ParseStream<'_>) -> Result<ComponentBindingsInput> {
    let mut component = None;
    let mut exports = None;
    let mut interfaces = Vec::new();

    while !input.is_empty() {
        let key: Ident = input.parse()?;
        let key_name = key.to_string();
        match key_name.as_str() {
            "component" => {
                input.parse::<Token![:]>()?;
                component = Some(input.parse()?);
            }
            "exports" => {
                input.parse::<Token![:]>()?;
                exports = Some(input.parse()?);
            }
            _ => {
                input.parse::<Token![=>]>()?;
                let content;
                braced!(content in input);
                let mut bindings = Vec::new();
                while !content.is_empty() {
                    let name: Ident = content.parse()?;
                    content.parse::<Token![=]>()?;
                    let ty: Type = content.parse()?;
                    bindings.push(AssociatedTypeBinding { name, ty });
                    if content.peek(Token![,]) {
                        content.parse::<Token![,]>()?;
                    }
                }
                interfaces.push(InterfaceBindings {
                    interface: key,
                    bindings,
                });
            }
        }

        if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
        }
    }

    let component = component.ok_or_else(|| {
        Error::new(
            Span::call_site(),
            "missing `component: ComponentType` entry",
        )
    })?;
    let exports = exports.ok_or_else(|| {
        Error::new(
            Span::call_site(),
            "missing `exports: crate::exports::...` entry",
        )
    })?;

    Ok(ComponentBindingsInput {
        component,
        exports,
        interfaces,
    })
}

fn parse_component_bindings_shorthand(input: ParseStream<'_>) -> Result<ComponentBindingsInput> {
    let component: Ident = input.parse()?;
    let exports =
        read_wit_exports_path().map_err(|message| Error::new(Span::call_site(), message))?;

    if input.is_empty() {
        let interfaces = infer_component_bindings_from_wit()
            .map_err(|message| Error::new(Span::call_site(), message))?;
        return Ok(ComponentBindingsInput {
            component,
            exports,
            interfaces,
        });
    }

    let content;
    braced!(content in input);

    let mut interfaces = Vec::new();
    while !content.is_empty() {
        let interface: Ident = content.parse()?;
        let interface_content;
        braced!(interface_content in content);
        let mut bindings = Vec::new();
        while !interface_content.is_empty() {
            let binding = parse_associated_type_binding(&interface_content)?;
            bindings.push(binding);
            if interface_content.peek(Token![,]) {
                interface_content.parse::<Token![,]>()?;
            }
        }

        interfaces.push(InterfaceBindings {
            interface,
            bindings,
        });

        if content.peek(Token![,]) {
            content.parse::<Token![,]>()?;
        }
    }

    Ok(ComponentBindingsInput {
        component,
        exports,
        interfaces,
    })
}

fn parse_associated_type_binding(input: ParseStream<'_>) -> Result<AssociatedTypeBinding> {
    let fork = input.fork();
    let _: Ident = fork.parse()?;
    if fork.peek(Token![=]) {
        let name: Ident = input.parse()?;
        input.parse::<Token![=]>()?;
        let ty: Type = input.parse()?;
        return Ok(AssociatedTypeBinding { name, ty });
    }

    let ty: Type = input.parse()?;
    let Type::Path(type_path) = &ty else {
        return Err(Error::new_spanned(
            ty,
            "shorthand component_bindings entries must be simple type paths",
        ));
    };
    let last = type_path
        .path
        .segments
        .last()
        .ok_or_else(|| Error::new_spanned(type_path, "expected a type path"))?
        .ident
        .to_string();
    let assoc_name = last.strip_suffix("Impl").ok_or_else(|| {
        Error::new_spanned(
            type_path,
            "shorthand component_bindings types must end with `Impl` or use `Name = Type`",
        )
    })?;

    Ok(AssociatedTypeBinding {
        name: Ident::new(assoc_name, Span::call_site()),
        ty,
    })
}

#[proc_macro]
pub fn component_bindings(item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ComponentBindingsInput);
    let component = input.component;
    let exports = input.exports;

    let interface_impls = input.interfaces.into_iter().map(|interface| {
        let interface_name = interface.interface;
        let bindings = interface.bindings.into_iter().map(|binding| {
            let name = binding.name;
            let ty = binding.ty;
            quote! {
                type #name = #ty;
            }
        });

        quote! {
            impl #exports::#interface_name::Guest for #component {
                #(#bindings)*
            }
        }
    });

    quote! {
        struct #component;

        #(#interface_impls)*

        export!(#component);
    }
    .into()
}
