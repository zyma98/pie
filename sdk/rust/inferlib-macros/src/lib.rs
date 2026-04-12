use std::collections::BTreeSet;

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{
    braced, parse_macro_input, Error, Fields, FnArg, GenericArgument, Ident, ImplItem, Item,
    ItemEnum, ItemFn, ItemImpl, ItemStruct, Pat, Path, PathArguments, Result, ReturnType, Token,
    Type, TypePath,
};

mod wit;

use self::wit::{
    find_interface_for_functions, find_interface_for_symbol, parse_interface_resources,
    parse_interface_function_names, parse_resource_method_names, parse_wit_interface_symbols,
    parse_world_exports,
    read_package_name, read_wit_exports_path, read_wit_world_name, read_wit_world_with_entries,
};

fn manifest_dir() -> std::result::Result<std::path::PathBuf, String> {
    std::env::var("CARGO_MANIFEST_DIR")
        .map(std::path::PathBuf::from)
        .map_err(|_| "CARGO_MANIFEST_DIR not set".to_string())
}

fn to_rust_ident(name: &str) -> syn::Ident {
    let sanitized = name.replace('-', "_");
    syn::Ident::new(&sanitized, Span::call_site())
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

fn default_component_ident() -> Ident {
    Ident::new("__InferlibComponent", Span::call_site())
}

fn resource_name_from_ident(ident: &Ident) -> String {
    let ident_str = ident.to_string();
    ident_str
        .strip_suffix("Impl")
        .unwrap_or(&ident_str)
        .to_string()
}

fn resource_name_from_type(ty: &Type) -> syn::Result<String> {
    let Type::Path(type_path) = ty else {
        return Err(Error::new_spanned(
            ty,
            "#[inferlib_macros::guest_resource] requires a concrete resource type",
        ));
    };
    let self_ident = &type_path.path.segments.last().expect("segment").ident;
    Ok(resource_name_from_ident(self_ident))
}

fn has_attr_named(attrs: &[syn::Attribute], name: &str) -> bool {
    attrs.iter().any(|attr| {
        attr.path()
            .segments
            .last()
            .map(|segment| segment.ident == name)
            .unwrap_or(false)
    })
}

fn has_rc_resource_attr(attrs: &[syn::Attribute]) -> bool {
    has_attr_named(attrs, "rc_resource")
}

fn hidden_shared_wrapper_ident(name: &Ident) -> Ident {
    Ident::new(&format!("__Shared{}", name), Span::call_site())
}

fn hidden_shared_state_ident(name: &Ident) -> Ident {
    Ident::new(&format!("__SharedState{}", name), Span::call_site())
}

fn has_rc_resource_struct(name: &Ident) -> std::result::Result<bool, String> {
    let src_dir = manifest_dir()?.join("src");
    let mut files = Vec::new();
    collect_rust_files(&src_dir, &mut files);

    for file in files {
        let source = std::fs::read_to_string(&file)
            .map_err(|e| format!("Failed to read `{}`: {e}", file.display()))?;
        let syntax = syn::parse_file(&source)
            .map_err(|e| format!("Failed to parse `{}`: {e}", file.display()))?;

        if syntax.items.iter().any(|item| {
            matches!(
                item,
                Item::Struct(item_struct)
                    if item_struct.ident == *name && has_rc_resource_attr(&item_struct.attrs)
            )
        }) {
            return Ok(true);
        }
    }

    Ok(false)
}

fn collect_rust_files(dir: &std::path::Path, files: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rust_files(&path, files);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            files.push(path);
        }
    }
}

fn module_path_for_source(
    src_dir: &std::path::Path,
    file: &std::path::Path,
) -> std::result::Result<Vec<String>, String> {
    let relative = file
        .strip_prefix(src_dir)
        .map_err(|e| format!("Failed to relativize source file `{}`: {e}", file.display()))?;
    let mut segments = relative
        .components()
        .filter_map(|component| component.as_os_str().to_str())
        .map(str::to_string)
        .collect::<Vec<_>>();

    if let Some(last) = segments.last_mut() {
        if last == "lib.rs" || last == "main.rs" || last == "mod.rs" {
            segments.pop();
        } else if let Some(stripped) = last.strip_suffix(".rs") {
            *last = stripped.to_string();
        }
    }

    Ok(segments)
}

fn source_interface_for_file(file: &std::path::Path) -> std::result::Result<Option<String>, String> {
    let src_dir = manifest_dir()?.join("src");
    let module_path = module_path_for_source(&src_dir, file)?;
    let exports = parse_world_exports()?;
    if let Some(interface) = module_path.last() {
        return Ok(exports.into_iter().find(|export| export == interface));
    }

    if exports.len() == 1 {
        return Ok(exports.into_iter().next());
    }

    Ok(None)
}

fn interface_has_dedicated_module(interface: &str) -> std::result::Result<bool, String> {
    let src_dir = manifest_dir()?.join("src");
    let interface_module = interface.replace('-', "_");
    Ok(src_dir.join(format!("{interface_module}.rs")).exists()
        || src_dir.join(interface_module).join("mod.rs").exists())
}

fn current_wit_resource_name_for_ident(
    interface: &str,
    ident: &Ident,
) -> std::result::Result<Option<String>, String> {
    let target = ident.to_string();
    let resources = parse_interface_resources(interface)?;
    let mut matches = resources
        .into_iter()
        .filter(|resource| to_upper_camel(resource) == target)
        .collect::<Vec<_>>();
    if matches.len() > 1 {
        return Err(format!(
            "resource type `{target}` matches multiple resources in interface `{interface}`"
        ));
    }
    Ok(matches.pop())
}

fn current_wit_named_type_for_ident(
    interface: &str,
    ident: &Ident,
) -> std::result::Result<Option<String>, String> {
    let target = ident.to_string();
    let symbols = parse_wit_interface_symbols(interface)?;
    let mut matches = symbols
        .named_types
        .into_iter()
        .filter(|name| to_upper_camel(name) == target)
        .collect::<Vec<_>>();
    if matches.len() > 1 {
        return Err(format!(
            "type `{target}` matches multiple named WIT types in interface `{interface}`"
        ));
    }
    Ok(matches.pop())
}

fn rc_resource_wrapper_type(name: &Ident) -> std::result::Result<Option<Type>, String> {
    let src_dir = manifest_dir()?.join("src");
    let mut files = Vec::new();
    collect_rust_files(&src_dir, &mut files);

    for file in files {
        let source = std::fs::read_to_string(&file)
            .map_err(|e| format!("Failed to read `{}`: {e}", file.display()))?;
        let syntax = syn::parse_file(&source)
            .map_err(|e| format!("Failed to parse `{}`: {e}", file.display()))?;

        if syntax.items.iter().any(|item| {
            matches!(
                item,
                Item::Struct(item_struct)
                    if item_struct.ident == *name && has_rc_resource_attr(&item_struct.attrs)
            )
        }) {
            return Ok(None);
        }

        for item in syntax.items {
            let syn::Item::Impl(item_impl) = item else {
                continue;
            };
            if !has_rc_resource_attr(&item_impl.attrs) {
                continue;
            }
            let Type::Path(type_path) = &*item_impl.self_ty else {
                continue;
            };
            let Some(self_ident) = type_path.path.segments.last().map(|segment| &segment.ident)
            else {
                continue;
            };
            if self_ident != name {
                continue;
            }

            let mut path = String::from("crate");
            for segment in module_path_for_source(&src_dir, &file)? {
                path.push_str("::");
                path.push_str(&segment.replace('-', "_"));
            }
            path.push_str("::");
            path.push_str(&hidden_shared_wrapper_ident(name).to_string());

            let ty = syn::parse_str::<Type>(&path)
                .map_err(|e| format!("Failed to build shared wrapper type `{path}`: {e}"))?;
            return Ok(Some(ty));
        }
    }

    Ok(None)
}

fn find_interface_for_resource(resource_name: &str) -> std::result::Result<Option<String>, String> {
    let mut matches = parse_world_exports()?
        .into_iter()
        .filter_map(|interface| match parse_interface_resources(&interface) {
            Ok(resources) if resources.iter().any(|resource| resource == resource_name) => {
                Some(Ok(interface))
            }
            Ok(_) => None,
            Err(error) => Some(Err(error)),
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;

    if matches.len() > 1 {
        return Err(format!(
            "resource `{resource_name}` is exported from multiple interfaces; specify `interface = ...` explicitly"
        ));
    }

    Ok(matches.pop())
}

fn resolve_rc_resource_binding(
    interface: Option<&str>,
    explicit_resource: Option<&str>,
    self_ident: &Ident,
) -> std::result::Result<(String, String), String> {
    if let Some(interface) = interface {
        let resources = parse_interface_resources(interface)?;
        if let Some(resource) = explicit_resource {
            if !resources.iter().any(|candidate| candidate == resource) {
                return Err(format!(
                    "resource `{resource}` was not found in exported interface `{interface}`"
                ));
            }
            return Ok((interface.to_string(), resource.to_string()));
        }

        let target = resource_name_from_ident(self_ident);
        let mut matches = resources
            .into_iter()
            .filter(|resource| to_upper_camel(resource) == target)
            .collect::<Vec<_>>();
        if matches.len() > 1 {
            return Err(format!(
                "resource type `{target}` matches multiple resources in interface `{interface}`; specify `resource = ...` explicitly"
            ));
        }
        if let Some(resource) = matches.pop() {
            return Ok((interface.to_string(), resource));
        }
        return Err(format!(
            "could not infer resource name for type `{target}` in interface `{interface}`"
        ));
    }

    if let Some(resource) = explicit_resource {
        let interface = find_interface_for_resource(resource)?
            .ok_or_else(|| format!("resource `{resource}` was not found in exported WIT"))?;
        return Ok((interface, resource.to_string()));
    }

    let target = resource_name_from_ident(self_ident);
    let mut matches = Vec::new();
    for interface in parse_world_exports()? {
        for resource in parse_interface_resources(&interface)? {
            if to_upper_camel(&resource) == target {
                matches.push((interface.clone(), resource));
            }
        }
    }

    if matches.len() > 1 {
        return Err(format!(
            "resource type `{target}` matches multiple exported resources; specify `interface = ...` or `resource = ...` explicitly"
        ));
    }

    matches.pop().ok_or_else(|| {
        format!("could not infer WIT resource binding for type `{target}`; specify `resource = ...` explicitly")
    })
}

fn infer_component_bindings_from_wit() -> std::result::Result<Vec<InterfaceBindings>, String> {
    let exports = parse_world_exports()?;
    let single_export = exports.len() == 1;
    let mut interfaces = Vec::new();
    for interface in exports {
        let resources = parse_interface_resources(&interface)?;
        if resources.is_empty() {
            continue;
        }

        let interface_module = interface.replace('-', "_");
        let type_prefix = if interface_has_dedicated_module(&interface)? {
            format!("crate::{interface_module}")
        } else if single_export {
            "crate".to_string()
        } else {
            format!("crate::{interface_module}")
        };
        let bindings = resources
            .into_iter()
            .map(|resource| {
                let assoc_name = to_upper_camel(&resource);
                let impl_name = format!("{type_prefix}::{assoc_name}");
                let name = Ident::new(&assoc_name, Span::call_site());
                let ty = syn::parse_str::<Type>(&impl_name)
                    .map_err(|e| format!("Failed to build inferred type `{impl_name}`: {e}"))?;
                Ok(AssociatedTypeBinding { name, ty })
            })
            .collect::<std::result::Result<Vec<_>, String>>()?;

        interfaces.push(InterfaceBindings {
            interface: to_rust_ident(&interface),
            bindings,
        });
    }

    Ok(interfaces)
}

struct ComponentInput {
    component: Ident,
    overrides: Vec<AssociatedTypeBinding>,
}

impl Parse for ComponentInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        if input.is_empty() {
            return Ok(Self {
                component: default_component_ident(),
                overrides: Vec::new(),
            });
        }

        let component = input.parse()?;
        let mut overrides = Vec::new();
        if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            while !input.is_empty() {
                overrides.push(parse_associated_type_binding(input)?);
                if input.peek(Token![,]) {
                    input.parse::<Token![,]>()?;
                }
            }
        }

        Ok(Self {
            component,
            overrides,
        })
    }
}

fn module_function_path_tokens(
    module_path: &[String],
    function_name: &Ident,
) -> proc_macro2::TokenStream {
    let segments = module_path
        .iter()
        .map(|segment| Ident::new(&segment.replace('-', "_"), Span::call_site()))
        .collect::<Vec<_>>();
    if segments.is_empty() {
        quote!(crate::#function_name)
    } else {
        quote!(crate::#(#segments::)*#function_name)
    }
}

fn generate_guest_function_method_tokens(
    item_fn: &ItemFn,
    module_path: &[String],
) -> syn::Result<proc_macro2::TokenStream> {
    let mut trait_sig = item_fn.sig.clone();
    let mut rewritten_inputs = syn::punctuated::Punctuated::new();
    let mut local_bindings = Vec::new();
    let mut call_args = Vec::new();

    for arg in &item_fn.sig.inputs {
        let FnArg::Typed(pat_ty) = arg else {
            return Err(Error::new_spanned(
                arg,
                "interface-level free functions must not have a receiver",
            ));
        };
        let Pat::Ident(pat_ident) = &*pat_ty.pat else {
            return Err(Error::new_spanned(
                &pat_ty.pat,
                "guest binding functions require simple identifier arguments",
            ));
        };
        let ident = pat_ident.ident.clone();
        let local_ty = (*pat_ty.ty).clone();
        let wit_ty = rewrite_wit_type(&local_ty)?;
        let mut rewritten = pat_ty.clone();
        rewritten.ty = Box::new(wit_ty);
        rewritten_inputs.push(FnArg::Typed(rewritten));

        let local_value = convert_expr(&local_ty, quote! { #ident })?;
        local_bindings.push(quote! {
            let #ident: #local_ty = #local_value;
        });
        call_args.push(quote! { #ident });
    }

    trait_sig.inputs = rewritten_inputs;
    trait_sig.output = match &item_fn.sig.output {
        ReturnType::Default => ReturnType::Default,
        ReturnType::Type(token, ty) => ReturnType::Type(*token, Box::new(rewrite_wit_type(ty)?)),
    };

    let function_name = &item_fn.sig.ident;
    let attrs = &item_fn.attrs;
    let call_path = module_function_path_tokens(module_path, function_name);
    let body = match &item_fn.sig.output {
        ReturnType::Default => quote! {{
            #(#local_bindings)*
            #call_path(#(#call_args),*);
        }},
        ReturnType::Type(_, ty) => {
            let converted = convert_expr(ty, quote! { result })?;
            quote! {{
                #(#local_bindings)*
                let result = #call_path(#(#call_args),*);
                #converted
            }}
        }
    };

    Ok(quote! {
        #(#attrs)*
        #trait_sig #body
    })
}

struct WitInterfaceInput {
    interface: Ident,
}

impl Parse for WitInterfaceInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        Ok(Self {
            interface: input.parse()?,
        })
    }
}

struct WitEnumInput {
    interface: Option<String>,
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

        Ok(Self { interface, name })
    }
}

struct GuestBindingInput {
    interface: Option<String>,
    resource: Option<String>,
}

impl Parse for GuestBindingInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let mut interface = None;
        let mut resource = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            let value: syn::LitStr = input.parse()?;
            match key.to_string().as_str() {
                "interface" => interface = Some(value.value()),
                "resource" => resource = Some(value.value()),
                other => {
                    return Err(Error::new(
                        key.span(),
                        format!("unsupported guest binding argument `{other}`"),
                    ));
                }
            }

            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(Self {
            interface,
            resource,
        })
    }
}

fn wit_type_path(
    interface: Option<&str>,
    explicit_name: Option<&str>,
    fallback_ident: &Ident,
) -> std::result::Result<Path, Error> {
    let resolved_interface = if let Some(interface) = interface {
        Some(interface.to_string())
    } else {
        let symbol_name = explicit_name
            .map(to_upper_camel)
            .unwrap_or_else(|| fallback_ident.to_string());
        find_interface_for_symbol(&symbol_name)
            .map_err(|message| Error::new(Span::call_site(), message))?
    };

    if let Some(interface) = resolved_interface.as_deref() {
        let exports =
            read_wit_exports_path().map_err(|message| Error::new(Span::call_site(), message))?;
        let interface_ident = to_rust_ident(interface);
        let wit_type_ident = Ident::new(
            &explicit_name
                .map(to_upper_camel)
                .unwrap_or_else(|| fallback_ident.to_string()),
            Span::call_site(),
        );
        return Ok(syn::parse_quote!(#exports::#interface_ident::#wit_type_ident));
    }
    let wit_type_ident = Ident::new(
        &explicit_name
            .map(to_upper_camel)
            .map(|name| format!("Wit{name}"))
            .unwrap_or_else(|| format!("Wit{}", fallback_ident)),
        Span::call_site(),
    );
    Ok(syn::parse_quote!(#wit_type_ident))
}

fn generic_inner_type<'a>(ty: &'a Type, outer: &str) -> Option<&'a Type> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    let segment = type_path.path.segments.last()?;
    if segment.ident != outer {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    args.args.iter().find_map(|arg| match arg {
        GenericArgument::Type(inner) => Some(inner),
        _ => None,
    })
}

fn is_identity_type(ty: &Type) -> bool {
    match ty {
        Type::Path(type_path) if type_path.qself.is_none() => {
            let Some(segment) = type_path.path.segments.last() else {
                return false;
            };
            matches!(
                segment.ident.to_string().as_str(),
                "String"
                    | "bool"
                    | "u8"
                    | "u16"
                    | "u32"
                    | "u64"
                    | "usize"
                    | "i8"
                    | "i16"
                    | "i32"
                    | "i64"
                    | "isize"
                    | "f32"
                    | "f64"
            )
        }
        _ => false,
    }
}

fn convert_expr(
    ty: &Type,
    value: proc_macro2::TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    if is_identity_type(ty) {
        return Ok(quote! { #value });
    }

    if let Some(inner) = generic_inner_type(ty, "Option") {
        let inner_expr = convert_expr(inner, quote! { inner })?;
        return Ok(quote! {
            (#value).map(|inner| #inner_expr)
        });
    }

    if let Some(inner) = generic_inner_type(ty, "Vec") {
        let inner_expr = convert_expr(inner, quote! { inner })?;
        return Ok(quote! {
            (#value).into_iter().map(|inner| #inner_expr).collect()
        });
    }

    if let Type::Tuple(tuple) = ty {
        let bindings = tuple
            .elems
            .iter()
            .enumerate()
            .map(|(idx, _)| quote::format_ident!("field_{idx}"))
            .collect::<Vec<_>>();
        let converted = tuple
            .elems
            .iter()
            .zip(bindings.iter())
            .map(|(elem_ty, binding)| convert_expr(elem_ty, quote! { #binding }))
            .collect::<syn::Result<Vec<_>>>()?;
        return Ok(quote! {{
            let (#(#bindings),*) = #value;
            (#(#converted),*)
        }});
    }

    Ok(quote! { (#value).into() })
}

fn exported_type_path_for_ident(
    ident: &Ident,
    arguments: &PathArguments,
) -> syn::Result<Option<Path>> {
    let ident_str = ident.to_string();
    let lookup_name = if let Some(stripped) = ident_str.strip_prefix("Wit") {
        stripped
    } else {
        &ident_str
    };
    let symbol_name = lookup_name
        .strip_suffix("Borrow")
        .unwrap_or(lookup_name)
        .to_string();

    let Some(interface) = find_interface_for_symbol(&symbol_name)
        .map_err(|message| Error::new(Span::call_site(), message))?
    else {
        return Ok(None);
    };

    let exports =
        read_wit_exports_path().map_err(|message| Error::new(Span::call_site(), message))?;
    let interface_ident = to_rust_ident(&interface);
    let target_ident = Ident::new(lookup_name, ident.span());
    let mut path: Path = syn::parse_quote!(#exports::#interface_ident::#target_ident);
    if let Some(last) = path.segments.last_mut() {
        last.arguments = arguments.clone();
    }
    Ok(Some(path))
}

fn rewrite_wit_type(ty: &Type) -> syn::Result<Type> {
    match ty {
        Type::Path(type_path) if type_path.qself.is_none() => {
            let mut rewritten = type_path.clone();
            for segment in &mut rewritten.path.segments {
                if let PathArguments::AngleBracketed(args) = &mut segment.arguments {
                    for arg in &mut args.args {
                        if let GenericArgument::Type(inner_ty) = arg {
                            *inner_ty = rewrite_wit_type(inner_ty)?;
                        }
                    }
                }
            }

            if rewritten.path.segments.len() == 1 {
                let segment = rewritten.path.segments.last_mut().expect("segment");
                let ident = segment.ident.to_string();
                if ident != "Self"
                    && !matches!(ident.as_str(), "Option" | "Vec" | "Result")
                    && !is_identity_type(ty)
                {
                    if let Some(path) =
                        exported_type_path_for_ident(&segment.ident, &segment.arguments)?
                    {
                        return Ok(Type::Path(TypePath { qself: None, path }));
                    }

                    if !ident.starts_with("Wit") && !ident.ends_with("Borrow") {
                        segment.ident = Ident::new(&format!("Wit{ident}"), segment.ident.span());
                    }
                }
            }

            Ok(Type::Path(TypePath {
                qself: None,
                path: rewritten.path,
            }))
        }
        Type::Reference(reference) => {
            let mut rewritten = reference.clone();
            rewritten.elem = Box::new(rewrite_wit_type(&reference.elem)?);
            Ok(Type::Reference(rewritten))
        }
        Type::Slice(slice) => {
            let mut rewritten = slice.clone();
            rewritten.elem = Box::new(rewrite_wit_type(&slice.elem)?);
            Ok(Type::Slice(rewritten))
        }
        Type::Tuple(tuple) => {
            let mut rewritten = tuple.clone();
            for elem in &mut rewritten.elems {
                *elem = rewrite_wit_type(elem)?;
            }
            Ok(Type::Tuple(rewritten))
        }
        _ => Ok(ty.clone()),
    }
}

fn derive_guest_trait_path(
    interface: Option<&str>,
    self_ty: &Type,
    resource_impl: bool,
    resource_name_override: Option<&str>,
) -> syn::Result<Path> {
    let trait_ident = if resource_impl {
        let resource_name = if let Some(resource_name) = resource_name_override {
            to_upper_camel(resource_name)
        } else {
            let Type::Path(type_path) = self_ty else {
                return Err(Error::new_spanned(
                    self_ty,
                    "#[inferlib_macros::guest_resource] requires a concrete resource type",
                ));
            };
            let self_ident = &type_path.path.segments.last().expect("segment").ident;
            resource_name_from_ident(self_ident)
        };
        let span = match self_ty {
            Type::Path(type_path) => type_path
                .path
                .segments
                .last()
                .expect("segment")
                .ident
                .span(),
            _ => Span::call_site(),
        };
        Ident::new(&format!("Guest{resource_name}"), span)
    } else {
        Ident::new("Guest", Span::call_site())
    };

    if let Some(interface) = interface {
        let exports =
            read_wit_exports_path().map_err(|message| Error::new(Span::call_site(), message))?;
        let interface_ident = to_rust_ident(interface);
        return Ok(syn::parse_quote!(#exports::#interface_ident::#trait_ident));
    }
    Ok(syn::parse_quote!(#trait_ident))
}

fn generate_guest_binding_tokens(
    args: &GuestBindingInput,
    item_impl: &ItemImpl,
    resource_impl: bool,
    allowed_method_names: Option<&BTreeSet<String>>,
) -> syn::Result<proc_macro2::TokenStream> {
    let self_ty = item_impl.self_ty.clone();
    let inferred_interface = if resource_impl {
        match &*item_impl.self_ty {
            Type::Path(type_path) => {
                let self_ident = type_path
                    .path
                    .segments
                    .last()
                    .expect("segment")
                    .ident
                    .clone();
                match resolve_rc_resource_binding(
                    args.interface.as_deref(),
                    args.resource.as_deref(),
                    &self_ident,
                ) {
                    Ok((interface, _)) => Some(interface),
                    Err(_) => None,
                }
            }
            _ => None,
        }
    } else if args.interface.is_none() {
        let method_names = item_impl
            .items
            .iter()
            .filter_map(|item| match item {
                ImplItem::Fn(method) => Some(method.sig.ident.to_string()),
                _ => None,
            })
            .collect::<Vec<_>>();
        find_interface_for_functions(method_names.iter().map(String::as_str))
            .map_err(|message| Error::new(Span::call_site(), message))?
    } else {
        None
    };
    let resolved_interface = args.interface.clone().or(inferred_interface);
    let trait_path = derive_guest_trait_path(
        resolved_interface.as_deref(),
        &self_ty,
        resource_impl,
        args.resource.as_deref(),
    )?;
    let resource_conversion = if resource_impl {
        let resource_name = args
            .resource
            .clone()
            .unwrap_or(resource_name_from_type(&self_ty)?);
        let wit_resource_path = wit_type_path(
            resolved_interface.as_deref(),
            Some(&resource_name),
            &Ident::new(&to_upper_camel(&resource_name), Span::call_site()),
        )?;
        Some(quote! {
            impl ::core::convert::From<#self_ty> for #wit_resource_path {
                fn from(value: #self_ty) -> Self {
                    Self::new(value)
                }
            }
        })
    } else {
        None
    };

    let trait_methods = item_impl
        .items
        .iter()
        .filter_map(|item| match item {
            ImplItem::Fn(method) => Some(method),
            _ => None,
        })
        .filter(|method| {
            allowed_method_names
                .map(|names| names.contains(&method.sig.ident.to_string()))
                .unwrap_or(true)
        })
        .map(|method| {
            let mut trait_sig = method.sig.clone();
            let mut rewritten_inputs = syn::punctuated::Punctuated::new();
            let mut local_bindings = Vec::new();
            let mut call_args = Vec::new();
            let has_receiver = method
                .sig
                .inputs
                .iter()
                .any(|arg| matches!(arg, FnArg::Receiver(_)));

            for arg in &method.sig.inputs {
                match arg {
                    FnArg::Receiver(receiver) => {
                        rewritten_inputs.push(FnArg::Receiver(receiver.clone()))
                    }
                    FnArg::Typed(pat_ty) => {
                        let Pat::Ident(pat_ident) = &*pat_ty.pat else {
                            return Err(Error::new_spanned(
                                &pat_ty.pat,
                                "guest binding methods require simple identifier arguments",
                            ));
                        };
                        let ident = pat_ident.ident.clone();
                        let local_ty = (*pat_ty.ty).clone();
                        let wit_ty = rewrite_wit_type(&local_ty)?;
                        let mut rewritten = pat_ty.clone();
                        rewritten.ty = Box::new(wit_ty);
                        rewritten_inputs.push(FnArg::Typed(rewritten));

                        let local_value = convert_expr(&local_ty, quote! { #ident })?;
                        local_bindings.push(quote! {
                            let #ident: #local_ty = #local_value;
                        });
                        call_args.push(quote! { #ident });
                    }
                }
            }

            trait_sig.inputs = rewritten_inputs;
            trait_sig.output = match &method.sig.output {
                ReturnType::Default => ReturnType::Default,
                ReturnType::Type(token, ty) => {
                    ReturnType::Type(*token, Box::new(rewrite_wit_type(ty)?))
                }
            };

            let method_name = &method.sig.ident;
            let attrs = &method.attrs;
            let call = if has_receiver {
                quote! { Self::#method_name(self, #(#call_args),*) }
            } else {
                quote! { Self::#method_name(#(#call_args),*) }
            };

            let body = match &method.sig.output {
                ReturnType::Default => quote! {{
                    #(#local_bindings)*
                    #call;
                }},
                ReturnType::Type(_, ty) => {
                    let converted = convert_expr(ty, quote! { result })?;
                    quote! {{
                        #(#local_bindings)*
                        let result = #call;
                        #converted
                    }}
                }
            };

            Ok(quote! {
                #(#attrs)*
                #trait_sig #body
            })
        })
        .collect::<syn::Result<Vec<_>>>()?;

    let (impl_generics, _, where_clause) = item_impl.generics.split_for_impl();

    Ok(quote! {
        #resource_conversion

        impl #impl_generics #trait_path for #self_ty #where_clause {
            #(#trait_methods)*
        }
    })
}

fn expand_guest_impl(
    args: GuestBindingInput,
    item_impl: ItemImpl,
    resource_impl: bool,
) -> syn::Result<proc_macro2::TokenStream> {
    if item_impl.trait_.is_some() {
        return Err(Error::new_spanned(
            &item_impl.self_ty,
            "guest binding attributes must be attached to inherent impl blocks",
        ));
    }

    let binding_tokens = generate_guest_binding_tokens(&args, &item_impl, resource_impl, None)?;

    Ok(quote! {
        #item_impl

        #binding_tokens
    })
}

fn is_self_keyword_type(ty: &Type) -> bool {
    let Type::Path(type_path) = ty else {
        return false;
    };
    type_path.qself.is_none()
        && type_path.path.segments.len() == 1
        && type_path
            .path
            .segments
            .last()
            .map(|segment| segment.ident == "Self")
            .unwrap_or(false)
}

fn is_same_named_type(ty: &Type, self_ty: &Type) -> bool {
    let (Type::Path(ty_path), Type::Path(self_path)) = (ty, self_ty) else {
        return false;
    };
    ty_path.qself.is_none()
        && self_path.qself.is_none()
        && ty_path.path.segments.len() == 1
        && self_path.path.segments.len() == 1
        && ty_path.path.segments.last().map(|segment| &segment.ident)
            == self_path.path.segments.last().map(|segment| &segment.ident)
}

fn returns_self_like(ty: &Type, self_ty: &Type) -> bool {
    is_self_keyword_type(ty) || is_same_named_type(ty, self_ty)
}

fn returns_other_resource_handle(
    ty: &Type,
    self_ident: &Ident,
    interface_resources: &[String],
) -> bool {
    if let Some(inner) = generic_inner_type(ty, "Option") {
        return returns_other_resource_handle(inner, self_ident, interface_resources);
    }

    let Type::Path(type_path) = ty else {
        return false;
    };
    let Some(segment) = type_path.path.segments.last() else {
        return false;
    };
    let returned = segment.ident.to_string();
    returned != self_ident.to_string()
        && interface_resources
            .iter()
            .any(|resource| to_upper_camel(resource) == returned)
}

enum SharedReceiverKind {
    Static,
    SharedRef,
    SharedMut,
}

fn expand_rc_resource_struct(item_struct: ItemStruct) -> syn::Result<proc_macro2::TokenStream> {
    if !item_struct.generics.params.is_empty() {
        return Err(Error::new_spanned(
            &item_struct.generics,
            "#[inferlib_macros::rc_resource] does not yet support generic resources",
        ));
    }

    let wrapper_ident = item_struct.ident.clone();
    let hidden_ident = hidden_shared_state_ident(&wrapper_ident);
    let visibility = item_struct.vis.clone();

    let mut hidden_struct = item_struct;
    hidden_struct.ident = hidden_ident.clone();

    Ok(quote! {
        #hidden_struct

        #visibility struct #wrapper_ident {
            inner: ::std::rc::Rc<::std::cell::RefCell<#hidden_ident>>,
        }

        impl ::core::clone::Clone for #wrapper_ident {
            fn clone(&self) -> Self {
                Self {
                    inner: ::std::rc::Rc::clone(&self.inner),
                }
            }
        }

        impl ::core::convert::From<#hidden_ident> for #wrapper_ident {
            fn from(value: #hidden_ident) -> Self {
                Self {
                    inner: ::std::rc::Rc::new(::std::cell::RefCell::new(value)),
                }
            }
        }
    })
}

fn expand_rc_resource_impl(
    args: GuestBindingInput,
    item_impl: ItemImpl,
) -> syn::Result<proc_macro2::TokenStream> {
    if item_impl.trait_.is_some() {
        return Err(Error::new_spanned(
            &item_impl.self_ty,
            "rc_resource must be attached to an inherent impl block",
        ));
    }

    let self_ty = item_impl.self_ty.clone();
    let Type::Path(type_path) = &*self_ty else {
        return Err(Error::new_spanned(
            &item_impl.self_ty,
            "#[inferlib_macros::rc_resource] requires a concrete resource type",
        ));
    };
    let self_ident = type_path
        .path
        .segments
        .last()
        .ok_or_else(|| Error::new_spanned(&item_impl.self_ty, "expected a resource type"))?
        .ident
        .clone();
    let state_wrapper_mode = has_rc_resource_struct(&self_ident)
        .map_err(|message| Error::new(Span::call_site(), message))?;

    let (resolved_interface, resolved_resource_name) = resolve_rc_resource_binding(
        args.interface.as_deref(),
        args.resource.as_deref(),
        &self_ident,
    )
    .map_err(|message| Error::new(Span::call_site(), message))?;

    let wit_resource_path = wit_type_path(
        Some(&resolved_interface),
        Some(&resolved_resource_name),
        &self_ident,
    )?;

    let interface_lit = syn::LitStr::new(&resolved_interface, Span::call_site());

    let build_wrapper_method = |method: &syn::ImplItemFn,
                                state_ty: &Type,
                                public_ty: &Type|
     -> syn::Result<proc_macro2::TokenStream> {
        let mut wrapper_sig = method.sig.clone();
        let receiver_kind = match method.sig.inputs.first() {
            Some(FnArg::Receiver(receiver)) => {
                if receiver.reference.is_none() {
                    return Err(Error::new_spanned(
                        receiver,
                        "#[inferlib_macros::rc_resource] only supports &self and &mut self methods",
                    ));
                }
                *wrapper_sig.inputs.iter_mut().next().expect("receiver") = syn::parse_quote!(&self);
                if receiver.mutability.is_some() {
                    SharedReceiverKind::SharedMut
                } else {
                    SharedReceiverKind::SharedRef
                }
            }
            Some(FnArg::Typed(_)) | None => SharedReceiverKind::Static,
        };

        let mut call_args = Vec::new();
        for arg in &method.sig.inputs {
            if let FnArg::Typed(pat_ty) = arg {
                let Pat::Ident(pat_ident) = &*pat_ty.pat else {
                    return Err(Error::new_spanned(
                        &pat_ty.pat,
                        "rc_resource methods require simple identifier arguments",
                    ));
                };
                call_args.push(pat_ident.ident.clone());
            }
        }

        let is_constructor =
            matches!(receiver_kind, SharedReceiverKind::Static) && method.sig.ident == "new";
        let method_name = &method.sig.ident;

        if let ReturnType::Type(_, ty) = &method.sig.output {
            if returns_self_like(ty, public_ty) {
                if is_constructor {
                    wrapper_sig.output = ReturnType::Type(
                        syn::token::RArrow::default(),
                        Box::new(syn::parse_quote!(Self)),
                    );
                } else {
                    wrapper_sig.output = ReturnType::Type(
                        syn::token::RArrow::default(),
                        Box::new(public_ty.clone()),
                    );
                }
            } else if matches!(**ty, Type::Reference(_)) {
                return Err(Error::new_spanned(
                    ty,
                    "#[inferlib_macros::rc_resource] wrapper methods cannot return references; return an owned value instead",
                ));
            }
        }

        let call = match receiver_kind {
            SharedReceiverKind::Static => {
                quote! { #state_ty::#method_name(#(#call_args),*) }
            }
            SharedReceiverKind::SharedRef => {
                quote! { #state_ty::#method_name(&*self.inner.borrow(), #(#call_args),*) }
            }
            SharedReceiverKind::SharedMut => {
                quote! { #state_ty::#method_name(&mut *self.inner.borrow_mut(), #(#call_args),*) }
            }
        };

        let body = match &method.sig.output {
            ReturnType::Default => quote! {{ #call; }},
            ReturnType::Type(_, ty) if returns_self_like(ty, public_ty) => {
                quote! {{
                    let result = #call;
                    Self::from(result)
                }}
            }
            ReturnType::Type(_, _) => quote! {{
                let result = #call;
                result
            }},
        };

        let attrs = &method.attrs;
        let vis = &method.vis;
        Ok(quote! {
            #(#attrs)*
            #vis #wrapper_sig #body
        })
    };

    if state_wrapper_mode {
        let hidden_state_ident = hidden_shared_state_ident(&self_ident);
        let hidden_state_ty: Type = syn::parse_quote!(#hidden_state_ident);
        let public_ty: Type = syn::parse_quote!(#self_ident);
        let interface_resources = parse_interface_resources(&resolved_interface)
            .map_err(|message| Error::new(Span::call_site(), message))?;

        let mut state_items = Vec::new();
        let mut wrapper_only_items = Vec::new();
        for item in &item_impl.items {
            match item {
                ImplItem::Fn(method)
                    if matches!(&method.sig.output, ReturnType::Type(_, ty)
                        if returns_other_resource_handle(ty, &self_ident, &interface_resources)) =>
                {
                    wrapper_only_items.push(ImplItem::Fn(method.clone()));
                }
                ImplItem::Fn(method) => {
                    let mut state_method = method.clone();
                    if let ReturnType::Type(_, ty) = &method.sig.output {
                        if returns_self_like(ty, &self_ty) {
                            state_method.sig.output = syn::parse_quote!(-> #hidden_state_ty);
                        }
                    }
                    state_items.push(ImplItem::Fn(state_method));
                }
                _ => state_items.push(item.clone()),
            }
        }

        let mut state_impl = item_impl.clone();
        state_impl.self_ty = Box::new(hidden_state_ty.clone());
        state_impl.items = state_items;

        let mut wrapper_only_impl = item_impl.clone();
        wrapper_only_impl.items = wrapper_only_items;

        let wrapper_methods = item_impl
            .items
            .iter()
            .filter_map(|item| match item {
                ImplItem::Fn(method) => Some(method),
                _ => None,
            })
            .filter(|method| {
                !matches!(
                    &method.sig.output,
                    ReturnType::Type(_, ty)
                        if returns_other_resource_handle(ty, &self_ident, &interface_resources)
                )
            })
            .map(|method| build_wrapper_method(method, &hidden_state_ty, &public_ty))
            .collect::<syn::Result<Vec<_>>>()?;

        let exported_method_names =
            parse_resource_method_names(&resolved_interface, &resolved_resource_name)
                .map_err(|message| Error::new(Span::call_site(), message))?;

        let trait_path = derive_guest_trait_path(
            Some(&resolved_interface),
            &public_ty,
            true,
            Some(&resolved_resource_name),
        )?;

        let guest_methods = item_impl
            .items
            .iter()
            .filter_map(|item| match item {
                ImplItem::Fn(method)
                    if exported_method_names.iter().any(|name| name == &method.sig.ident.to_string()) =>
                {
                    Some(method)
                }
                _ => None,
            })
            .map(|method| {
                let mut trait_sig = method.sig.clone();
                let mut rewritten_inputs = syn::punctuated::Punctuated::new();
                let mut local_bindings = Vec::new();
                let mut call_args = Vec::new();
                let has_receiver = method
                    .sig
                    .inputs
                    .iter()
                    .any(|arg| matches!(arg, FnArg::Receiver(_)));

                for arg in &method.sig.inputs {
                    match arg {
                        FnArg::Receiver(receiver) => {
                            let rewritten: FnArg = syn::parse_quote!(&self);
                            if receiver.reference.is_some() {
                                rewritten_inputs.push(rewritten);
                            } else {
                                return Err(Error::new_spanned(
                                    receiver,
                                    "#[inferlib_macros::rc_resource] only supports &self and &mut self methods",
                                ));
                            }
                        }
                        FnArg::Typed(pat_ty) => {
                            let Pat::Ident(pat_ident) = &*pat_ty.pat else {
                                return Err(Error::new_spanned(
                                    &pat_ty.pat,
                                    "guest binding methods require simple identifier arguments",
                                ));
                            };
                            let ident = pat_ident.ident.clone();
                            let local_ty = (*pat_ty.ty).clone();
                            let wit_ty = rewrite_wit_type(&local_ty)?;
                            let mut rewritten = pat_ty.clone();
                            rewritten.ty = Box::new(wit_ty);
                            rewritten_inputs.push(FnArg::Typed(rewritten));

                            let local_value = convert_expr(&local_ty, quote! { #ident })?;
                            local_bindings.push(quote! {
                                let #ident: #local_ty = #local_value;
                            });
                            call_args.push(quote! { #ident });
                        }
                    }
                }

                trait_sig.inputs = rewritten_inputs;
                trait_sig.output = match &method.sig.output {
                    ReturnType::Default => ReturnType::Default,
                    ReturnType::Type(token, ty) => {
                        ReturnType::Type(*token, Box::new(rewrite_wit_type(ty)?))
                    }
                };

                let method_name = &method.sig.ident;
                let attrs = &method.attrs;
                let call = if has_receiver {
                    quote! { #self_ident::#method_name(self, #(#call_args),*) }
                } else {
                    quote! { #self_ident::#method_name(#(#call_args),*) }
                };

                let body = match &method.sig.output {
                    ReturnType::Default => quote! {{
                        #(#local_bindings)*
                        #call;
                    }},
                    ReturnType::Type(_, ty) => {
                        let converted = convert_expr(ty, quote! { result })?;
                        quote! {{
                            #(#local_bindings)*
                            let result = #call;
                            #converted
                        }}
                    }
                };

                Ok(quote! {
                    #(#attrs)*
                    #trait_sig #body
                })
            })
            .collect::<syn::Result<Vec<_>>>()?;

        Ok(quote! {
            #state_impl

            impl #self_ident {
                #(#wrapper_methods)*
            }

            #wrapper_only_impl

            impl ::core::convert::From<#self_ident> for #wit_resource_path {
                fn from(value: #self_ident) -> Self {
                    Self::new(value)
                }
            }

            impl #trait_path for #self_ident {
                #(#guest_methods)*
            }
        })
    } else {
        let hidden_ident = hidden_shared_wrapper_ident(&self_ident);
        let simple_resource_name = args
            .resource
            .clone()
            .unwrap_or(resource_name_from_type(&self_ty)?);
        let simple_resource_lit = syn::LitStr::new(&simple_resource_name, Span::call_site());
        let simple_wit_resource_path = wit_type_path(
            Some(&resolved_interface),
            Some(&simple_resource_name),
            &self_ident,
        )?;

        let exported_methods = item_impl
            .items
            .iter()
            .filter_map(|item| match item {
                ImplItem::Fn(method) if !matches!(method.vis, syn::Visibility::Inherited) => {
                    Some(method)
                }
                _ => None,
            })
            .map(|method| build_wrapper_method(method, &self_ty, &self_ty))
            .collect::<syn::Result<Vec<_>>>()?;

        Ok(quote! {
            #item_impl

            pub(crate) struct #hidden_ident {
                inner: ::std::rc::Rc<::std::cell::RefCell<#self_ty>>,
            }

            impl ::core::convert::From<#self_ty> for #hidden_ident {
                fn from(value: #self_ty) -> Self {
                    Self {
                        inner: ::std::rc::Rc::new(::std::cell::RefCell::new(value)),
                    }
                }
            }

            impl ::core::convert::From<#self_ty> for #simple_wit_resource_path {
                fn from(value: #self_ty) -> Self {
                    #hidden_ident::from(value).into()
                }
            }

            #[inferlib_macros::guest_resource(interface = #interface_lit, resource = #simple_resource_lit)]
            impl #hidden_ident {
                #(#exported_methods)*
            }
        })
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
pub fn old_main(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut input_fn = parse_macro_input!(item as ItemFn);
    let original_fn_name = input_fn.sig.ident.clone();
    let inner_fn_name = syn::Ident::new("__pie_main_inner", original_fn_name.span());

    if input_fn.sig.asyncness.is_none() {
        return syn::Error::new_spanned(
            input_fn.sig.ident,
            "The #[inferlib_macros::old_main] attribute can only be used on async functions",
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
                    ::inferlib_inference_bindings_old::get_arguments()
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

                        ::inferlib_inference_bindings_old::set_return(&output);
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

fn generate_wit_enum_impls(
    item_enum: &ItemEnum,
    interface: Option<&str>,
    name: Option<&str>,
) -> syn::Result<proc_macro2::TokenStream> {
    if item_enum
        .variants
        .iter()
        .any(|variant| !matches!(variant.fields, syn::Fields::Unit))
    {
        return Err(Error::new_spanned(
            &item_enum.ident,
            "#[inferlib_macros::wit_enum] currently supports only unit enums",
        ));
    }

    let enum_ident = item_enum.ident.clone();
    let wit_type_path = wit_type_path(interface, name, &enum_ident)?;
    let variants = item_enum
        .variants
        .iter()
        .map(|variant| variant.ident.clone())
        .collect::<Vec<_>>();

    Ok(quote! {
        impl ::core::convert::From<#enum_ident> for #wit_type_path {
            fn from(value: #enum_ident) -> Self {
                match value {
                    #( #enum_ident::#variants => Self::#variants, )*
                }
            }
        }

        impl ::core::convert::From<#wit_type_path> for #enum_ident {
            fn from(value: #wit_type_path) -> Self {
                match value {
                    #( #wit_type_path::#variants => Self::#variants, )*
                }
            }
        }
    })
}

fn generate_wit_record_impls(
    item_struct: &ItemStruct,
    interface: Option<&str>,
    name: Option<&str>,
) -> syn::Result<proc_macro2::TokenStream> {
    let Fields::Named(fields) = &item_struct.fields else {
        return Err(Error::new_spanned(
            &item_struct.ident,
            "#[inferlib_macros::wit_record] requires a struct with named fields",
        ));
    };

    let struct_ident = item_struct.ident.clone();
    let wit_type_path = wit_type_path(interface, name, &struct_ident)?;

    let field_idents = fields
        .named
        .iter()
        .map(|field| field.ident.clone().expect("named field"))
        .collect::<Vec<_>>();
    let to_wit_fields = fields
        .named
        .iter()
        .zip(field_idents.iter())
        .map(|(field, ident)| convert_expr(&field.ty, quote! { value.#ident }))
        .collect::<syn::Result<Vec<_>>>()?;
    let from_wit_fields = fields
        .named
        .iter()
        .zip(field_idents.iter())
        .map(|(field, ident)| convert_expr(&field.ty, quote! { value.#ident }))
        .collect::<syn::Result<Vec<_>>>()?;

    Ok(quote! {
        impl ::core::convert::From<#struct_ident> for #wit_type_path {
            fn from(value: #struct_ident) -> Self {
                Self {
                    #( #field_idents: #to_wit_fields, )*
                }
            }
        }

        impl ::core::convert::From<#wit_type_path> for #struct_ident {
            fn from(value: #wit_type_path) -> Self {
                Self {
                    #( #field_idents: #from_wit_fields, )*
                }
            }
        }
    })
}

fn generate_wit_variant_impls(
    item_enum: &ItemEnum,
    interface: Option<&str>,
    name: Option<&str>,
) -> syn::Result<proc_macro2::TokenStream> {
    let enum_ident = item_enum.ident.clone();
    let wit_type_path = wit_type_path(interface, name, &enum_ident)?;

    let mut to_wit_arms = Vec::new();
    let mut from_wit_arms = Vec::new();

    for variant in &item_enum.variants {
        let variant_ident = &variant.ident;
        match &variant.fields {
            Fields::Unit => {
                to_wit_arms.push(quote! { #enum_ident::#variant_ident => Self::#variant_ident, });
                from_wit_arms
                    .push(quote! { #wit_type_path::#variant_ident => Self::#variant_ident, });
            }
            Fields::Unnamed(fields) if fields.unnamed.len() == 1 => {
                let binding = quote::format_ident!("value");
                let field_ty = &fields.unnamed.first().expect("field").ty;
                let to_wit = convert_expr(field_ty, quote! { #binding })?;
                let from_wit = convert_expr(field_ty, quote! { value })?;
                to_wit_arms.push(
                    quote! { #enum_ident::#variant_ident(#binding) => Self::#variant_ident(#to_wit), },
                );
                from_wit_arms.push(quote! {
                    #wit_type_path::#variant_ident(value) => Self::#variant_ident(#from_wit),
                });
            }
            Fields::Unnamed(fields) => {
                let bindings = fields
                    .unnamed
                    .iter()
                    .enumerate()
                    .map(|(idx, _)| quote::format_ident!("value_{idx}"))
                    .collect::<Vec<_>>();
                let field_types = fields
                    .unnamed
                    .iter()
                    .map(|field| &field.ty)
                    .collect::<Vec<_>>();
                let tuple_ty: Type = syn::parse_quote!((#(#field_types),*));
                let tuple_value = quote!((#(#bindings),*));
                let to_wit = convert_expr(&tuple_ty, tuple_value)?;
                let from_wit = convert_expr(&tuple_ty, quote! { value })?;
                to_wit_arms.push(quote! {
                    #enum_ident::#variant_ident(#(#bindings),*) => Self::#variant_ident(#to_wit),
                });
                from_wit_arms.push(quote! {
                    #wit_type_path::#variant_ident(value) => {
                        let value = #from_wit;
                        let (#(#bindings),*) = value;
                        Self::#variant_ident(#(#bindings),*)
                    }
                });
            }
            Fields::Named(_) => {
                return Err(Error::new_spanned(
                    variant,
                    "#[inferlib_macros::wit_variant] does not support named-field variants",
                ));
            }
        }
    }

    Ok(quote! {
        impl ::core::convert::From<#enum_ident> for #wit_type_path {
            fn from(value: #enum_ident) -> Self {
                match value {
                    #(#to_wit_arms)*
                }
            }
        }

        impl ::core::convert::From<#wit_type_path> for #enum_ident {
            fn from(value: #wit_type_path) -> Self {
                match value {
                    #(#from_wit_arms)*
                }
            }
        }
    })
}

#[proc_macro_attribute]
pub fn wit_enum(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as WitEnumInput);
    let item_enum = parse_macro_input!(item as ItemEnum);
    let impls =
        match generate_wit_enum_impls(&item_enum, args.interface.as_deref(), args.name.as_deref())
        {
            Ok(tokens) => tokens,
            Err(error) => return error.to_compile_error().into(),
        };

    quote! {
        #item_enum
        #impls
    }
    .into()
}

#[proc_macro_attribute]
pub fn wit_record(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as WitEnumInput);
    let item_struct = parse_macro_input!(item as ItemStruct);
    let impls = match generate_wit_record_impls(
        &item_struct,
        args.interface.as_deref(),
        args.name.as_deref(),
    ) {
        Ok(tokens) => tokens,
        Err(error) => return error.to_compile_error().into(),
    };

    quote! {
        #item_struct
        #impls
    }
    .into()
}

#[proc_macro_attribute]
pub fn wit_variant(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as WitEnumInput);
    let item_enum = parse_macro_input!(item as ItemEnum);
    let impls = match generate_wit_variant_impls(
        &item_enum,
        args.interface.as_deref(),
        args.name.as_deref(),
    ) {
        Ok(tokens) => tokens,
        Err(error) => return error.to_compile_error().into(),
    };

    quote! {
        #item_enum
        #impls
    }
    .into()
}

#[proc_macro_attribute]
pub fn guest_interface(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as GuestBindingInput);
    let item_impl = parse_macro_input!(item as ItemImpl);

    match expand_guest_impl(args, item_impl, false) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

#[proc_macro_attribute]
pub fn guest_resource(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as GuestBindingInput);
    let item_impl = parse_macro_input!(item as ItemImpl);

    match expand_guest_impl(args, item_impl, true) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

#[proc_macro_attribute]
pub fn rc_resource(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as GuestBindingInput);
    let item = parse_macro_input!(item as Item);

    match item {
        Item::Struct(item_struct) => match expand_rc_resource_struct(item_struct) {
            Ok(tokens) => tokens.into(),
            Err(error) => error.to_compile_error().into(),
        },
        Item::Impl(item_impl) => match expand_rc_resource_impl(args, item_impl) {
            Ok(tokens) => tokens.into(),
            Err(error) => error.to_compile_error().into(),
        },
        other => Error::new_spanned(
            other,
            "#[inferlib_macros::rc_resource] must be attached to a struct or inherent impl block",
        )
        .to_compile_error()
        .into(),
    }
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
    let assoc_name = last.strip_suffix("Impl").unwrap_or(&last);

    Ok(AssociatedTypeBinding {
        name: Ident::new(assoc_name, Span::call_site()),
        ty,
    })
}

fn infer_auto_component_tokens(
    component: &Ident,
) -> std::result::Result<(Vec<proc_macro2::TokenStream>, Vec<proc_macro2::TokenStream>), String>
{
    let src_dir = manifest_dir()?.join("src");
    let mut files = Vec::new();
    collect_rust_files(&src_dir, &mut files);

    let mut type_impls = Vec::new();
    let mut guest_impls = Vec::new();
    let mut free_function_methods_by_interface: std::collections::BTreeMap<
        String,
        Vec<proc_macro2::TokenStream>,
    > = std::collections::BTreeMap::new();
    let exports = read_wit_exports_path()?;

    for file in files {
        let Some(interface) = source_interface_for_file(&file)? else {
            continue;
        };
        let module_path = module_path_for_source(&src_dir, &file)?;

        let source = std::fs::read_to_string(&file)
            .map_err(|e| format!("Failed to read `{}`: {e}", file.display()))?;
        let syntax = syn::parse_file(&source)
            .map_err(|e| format!("Failed to parse `{}`: {e}", file.display()))?;

        let interface_function_names = parse_interface_function_names(&interface)?
            .into_iter()
            .collect::<BTreeSet<_>>();

        for item in syntax.items {
            match item {
                Item::Fn(item_fn) => {
                    if matches!(item_fn.vis, syn::Visibility::Inherited)
                        || !interface_function_names.contains(&item_fn.sig.ident.to_string())
                    {
                        continue;
                    }
                    let tokens = generate_guest_function_method_tokens(&item_fn, &module_path)
                        .map_err(|e| e.to_string())?;
                    free_function_methods_by_interface
                        .entry(interface.clone())
                        .or_default()
                        .push(tokens);
                }
                Item::Struct(item_struct) => {
                    if has_attr_named(&item_struct.attrs, "wit_record")
                        || has_rc_resource_attr(&item_struct.attrs)
                    {
                        continue;
                    }
                    if current_wit_resource_name_for_ident(&interface, &item_struct.ident)?.is_some()
                    {
                        continue;
                    }
                    if current_wit_named_type_for_ident(&interface, &item_struct.ident)?.is_none() {
                        continue;
                    }
                    if !matches!(item_struct.fields, Fields::Named(_)) {
                        continue;
                    }
                    if let Ok(tokens) =
                        generate_wit_record_impls(&item_struct, Some(&interface), None)
                    {
                        type_impls.push(tokens);
                    }
                }
                Item::Enum(item_enum) => {
                    if has_attr_named(&item_enum.attrs, "wit_enum")
                        || has_attr_named(&item_enum.attrs, "wit_variant")
                    {
                        continue;
                    }
                    if current_wit_named_type_for_ident(&interface, &item_enum.ident)?.is_none() {
                        continue;
                    }
                    let tokens = if item_enum
                        .variants
                        .iter()
                        .all(|variant| matches!(variant.fields, Fields::Unit))
                    {
                        generate_wit_enum_impls(&item_enum, Some(&interface), None)
                    } else {
                        generate_wit_variant_impls(&item_enum, Some(&interface), None)
                    };
                    if let Ok(tokens) = tokens {
                        type_impls.push(tokens);
                    }
                }
                Item::Impl(item_impl) => {
                    if item_impl.trait_.is_some()
                        || has_attr_named(&item_impl.attrs, "guest_interface")
                        || has_attr_named(&item_impl.attrs, "guest_resource")
                        || has_rc_resource_attr(&item_impl.attrs)
                    {
                        continue;
                    }

                    let Type::Path(type_path) = &*item_impl.self_ty else {
                        continue;
                    };
                    let Some(self_ident) = type_path.path.segments.last().map(|segment| &segment.ident)
                    else {
                        continue;
                    };

                    if self_ident == component {
                        if interface_function_names.is_empty() {
                            continue;
                        }
                        let has_matching_methods = item_impl.items.iter().any(|item| {
                            matches!(
                                item,
                                ImplItem::Fn(method)
                                    if interface_function_names.contains(&method.sig.ident.to_string())
                            )
                        });
                        if !has_matching_methods {
                            continue;
                        }
                        let args = GuestBindingInput {
                            interface: Some(interface.clone()),
                            resource: None,
                        };
                        let tokens = generate_guest_binding_tokens(
                            &args,
                            &item_impl,
                            false,
                            Some(&interface_function_names),
                        )
                        .map_err(|e| e.to_string())?;
                        guest_impls.push(tokens);
                        continue;
                    }

                    let Some(resource_name) =
                        current_wit_resource_name_for_ident(&interface, self_ident)?
                    else {
                        continue;
                    };
                    let exported_method_names = parse_resource_method_names(&interface, &resource_name)?
                        .into_iter()
                        .collect::<BTreeSet<_>>();
                    let has_matching_methods = item_impl.items.iter().any(|item| {
                        matches!(
                            item,
                            ImplItem::Fn(method)
                                if exported_method_names.contains(&method.sig.ident.to_string())
                        )
                    });
                    if !has_matching_methods {
                        continue;
                    }
                    let args = GuestBindingInput {
                        interface: Some(interface.clone()),
                        resource: Some(resource_name),
                    };
                    let tokens = generate_guest_binding_tokens(
                        &args,
                        &item_impl,
                        true,
                        Some(&exported_method_names),
                    )
                    .map_err(|e| e.to_string())?;
                    guest_impls.push(tokens);
                }
                _ => {}
            }
        }
    }

    for (interface, methods) in free_function_methods_by_interface {
        if methods.is_empty() {
            continue;
        }
        let interface_ident = to_rust_ident(&interface);
        guest_impls.push(quote! {
            impl #exports::#interface_ident::Guest for #component {
                #(#methods)*
            }
        });
    }

    Ok((type_impls, guest_impls))
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

#[proc_macro]
pub fn component(item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ComponentInput);
    let component = input.component;
    let overrides = input.overrides;

    let world = match read_wit_world_name() {
        Ok(world) => world,
        Err(message) => {
            return Error::new(Span::call_site(), message)
                .to_compile_error()
                .into();
        }
    };

    let with_entries = match read_wit_world_with_entries() {
        Ok(entries) => entries,
        Err(message) => {
            return Error::new(Span::call_site(), message)
                .to_compile_error()
                .into();
        }
    };

    let exports = match read_wit_exports_path() {
        Ok(path) => path,
        Err(message) => {
            return Error::new(Span::call_site(), message)
                .to_compile_error()
                .into();
        }
    };

    let interfaces = match infer_component_bindings_from_wit() {
        Ok(interfaces) => interfaces,
        Err(message) => {
            return Error::new(Span::call_site(), message)
                .to_compile_error()
                .into();
        }
    };

    let (auto_type_impls, auto_guest_impls) = match infer_auto_component_tokens(&component) {
        Ok(tokens) => tokens,
        Err(message) => {
            return Error::new(Span::call_site(), message)
                .to_compile_error()
                .into();
        }
    };
    let auto_imports = match parse_world_exports() {
        Ok(exports) => exports
            .into_iter()
            .filter_map(|interface| {
                match interface_has_dedicated_module(&interface) {
                    Ok(true) => {
                        let module_ident = to_rust_ident(&interface);
                        Some(quote! { use crate::#module_ident::*; })
                    }
                    Ok(false) => None,
                    Err(message) => Some(
                        Error::new(Span::call_site(), message).to_compile_error(),
                    ),
                }
            })
            .collect::<Vec<_>>(),
        Err(message) => {
            return Error::new(Span::call_site(), message)
                .to_compile_error()
                .into();
        }
    };

    let interface_impls = interfaces.into_iter().map(|interface| {
        let interface_name = interface.interface;
        let bindings = interface.bindings.into_iter().map(|binding| {
            let name = binding.name;
            let inferred_ty = overrides
                .iter()
                .find(|override_binding| override_binding.name == name)
                .map(|override_binding| override_binding.ty.clone())
                .unwrap_or(binding.ty);
            let ty = if overrides
                .iter()
                .any(|override_binding| override_binding.name == name)
            {
                inferred_ty
            } else if let Type::Path(type_path) = &inferred_ty {
                let ty_ident = type_path
                    .path
                    .segments
                    .last()
                    .expect("segment")
                    .ident
                    .clone();
                rc_resource_wrapper_type(&ty_ident)
                    .ok()
                    .flatten()
                    .unwrap_or(inferred_ty)
            } else {
                inferred_ty
            };
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

    let with_tokens = if with_entries.is_empty() {
        quote! {}
    } else {
        let entries = with_entries.iter().map(|(import, path)| {
            quote! {
                #import: #path,
            }
        });
        quote! {
            with: {
                #(#entries)*
            },
        }
    };

    quote! {
        mod __wit {
            wit_bindgen::generate!({
                path: "wit",
                world: #world,
                generate_all,
                pub_export_macro: true,
                #with_tokens
            });
        }

        pub(crate) use __wit::exports;
        #(#auto_imports)*

        #(#auto_type_impls)*

        struct #component;

        #(#interface_impls)*
        #(#auto_guest_impls)*

        __wit::export!(#component with_types_in __wit);
    }
    .into()
}

#[proc_macro]
pub fn wit_interface(item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as WitInterfaceInput);
    let interface_ident = input.interface;
    let interface_name = interface_ident.to_string();

    let exports = match read_wit_exports_path() {
        Ok(path) => path,
        Err(message) => {
            return Error::new(Span::call_site(), message)
                .to_compile_error()
                .into();
        }
    };

    let symbols = match parse_wit_interface_symbols(&interface_name) {
        Ok(symbols) => symbols,
        Err(message) => {
            return Error::new(Span::call_site(), message)
                .to_compile_error()
                .into();
        }
    };

    let guest_traits = std::iter::once(Ident::new("Guest", Span::call_site()))
        .chain(symbols.resources.iter().map(|resource| {
            Ident::new(
                &format!("Guest{}", to_upper_camel(resource)),
                Span::call_site(),
            )
        }))
        .collect::<Vec<_>>();

    let plain_resource_borrow_types = symbols
        .resources
        .iter()
        .map(|resource| {
            let resource_name = to_upper_camel(resource);
            Ident::new(&format!("{resource_name}Borrow"), Span::call_site())
        })
        .collect::<Vec<_>>();

    let aliased_types = symbols
        .resources
        .iter()
        .flat_map(|resource| {
            let resource_name = to_upper_camel(resource);
            [
                (
                    Ident::new(&resource_name, Span::call_site()),
                    Ident::new(&format!("Wit{resource_name}"), Span::call_site()),
                ),
                (
                    Ident::new(&format!("{resource_name}Borrow"), Span::call_site()),
                    Ident::new(&format!("Wit{resource_name}Borrow"), Span::call_site()),
                ),
            ]
        })
        .chain(symbols.named_types.iter().map(|name| {
            let rust_name = to_upper_camel(name);
            (
                Ident::new(&rust_name, Span::call_site()),
                Ident::new(&format!("Wit{rust_name}"), Span::call_site()),
            )
        }))
        .collect::<Vec<_>>();

    let guest_use = if guest_traits.is_empty() {
        quote! {}
    } else {
        quote! {
            pub(crate) use #exports::#interface_ident::{ #(#guest_traits),* };
        }
    };

    let plain_resource_use = if plain_resource_borrow_types.is_empty() {
        quote! {}
    } else {
        quote! {
            pub(crate) use #exports::#interface_ident::{ #(#plain_resource_borrow_types),* };
        }
    };

    let type_use = if aliased_types.is_empty() {
        quote! {}
    } else {
        let renamed = aliased_types
            .iter()
            .map(|(orig, alias)| quote! { #orig as #alias });
        quote! {
            pub(crate) use #exports::#interface_ident::{ #(#renamed),* };
        }
    };

    quote! {
        #guest_use
        #plain_resource_use
        #type_use
    }
    .into()
}
