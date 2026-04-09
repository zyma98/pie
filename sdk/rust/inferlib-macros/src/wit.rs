use std::collections::BTreeSet;

use syn::Path;

use super::{manifest_dir, to_upper_camel};

pub(crate) fn read_wit_exports_path() -> std::result::Result<Path, String> {
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

fn read_world_wit() -> std::result::Result<String, String> {
    let manifest_dir = manifest_dir()?;
    let world_wit_path = manifest_dir.join("wit/world.wit");
    std::fs::read_to_string(&world_wit_path).map_err(|_| {
        "Failed to read wit/world.wit - make sure it exists next to Cargo.toml".to_string()
    })
}

pub(crate) fn read_wit_world_name() -> std::result::Result<String, String> {
    let world_wit = read_world_wit()?;
    world_wit
        .lines()
        .map(str::trim)
        .find_map(|line| {
            line.strip_prefix("world ")
                .and_then(|rest| rest.split_once('{').map(|(name, _)| name.trim()))
                .filter(|name| !name.is_empty())
                .map(ToOwned::to_owned)
        })
        .ok_or_else(|| "Failed to find `world ... {` in wit/world.wit".to_string())
}

pub(crate) fn read_wit_world_imports() -> std::result::Result<Vec<String>, String> {
    let world_wit = read_world_wit()?;
    Ok(world_wit
        .lines()
        .map(str::trim)
        .filter_map(|line| {
            line.strip_prefix("import ")
                .and_then(|rest| rest.strip_suffix(';'))
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .map(ToOwned::to_owned)
        })
        .collect())
}

pub(crate) fn read_interface_wit(interface: &str) -> std::result::Result<String, String> {
    let manifest_dir = manifest_dir()?;
    let file_name = format!("{}.wit", interface.replace('_', "-"));
    let interface_wit_path = manifest_dir.join("wit").join(file_name);
    std::fs::read_to_string(&interface_wit_path).map_err(|_| {
        format!("Failed to read wit/{interface}.wit - make sure it exists next to Cargo.toml")
    })
}

#[derive(Default)]
pub(crate) struct WitInterfaceSymbols {
    pub(crate) resources: Vec<String>,
    pub(crate) named_types: Vec<String>,
}

pub(crate) fn parse_wit_interface_symbols(
    interface: &str,
) -> std::result::Result<WitInterfaceSymbols, String> {
    let wit = read_interface_wit(interface)?;
    let mut symbols = WitInterfaceSymbols::default();

    for line in wit.lines().map(str::trim) {
        if let Some(name) = parse_wit_decl_name(line, "resource") {
            symbols.resources.push(name);
            continue;
        }
        for keyword in ["record", "enum", "variant", "flags", "type"] {
            if let Some(name) = parse_wit_decl_name(line, keyword) {
                symbols.named_types.push(name);
                break;
            }
        }
    }

    Ok(symbols)
}

fn parse_wit_decl_name(line: &str, keyword: &str) -> Option<String> {
    let rest = line.strip_prefix(keyword)?.trim_start();
    let name = rest
        .split([' ', '{', '=', ';'])
        .next()
        .map(str::trim)
        .filter(|name| !name.is_empty())?;
    Some(name.to_string())
}

pub(crate) fn known_import_remap(import: &str) -> Option<Path> {
    if import.starts_with("wasi:io/poll@") {
        return syn::parse_str("wasip2::io::poll").ok();
    }
    None
}

pub(crate) fn parse_world_exports() -> std::result::Result<Vec<String>, String> {
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

pub(crate) fn parse_interface_resources(
    interface: &str,
) -> std::result::Result<Vec<String>, String> {
    let manifest_dir = manifest_dir()?;
    let interface_wit_path = manifest_dir.join("wit").join(format!("{interface}.wit"));
    let interface_wit = std::fs::read_to_string(&interface_wit_path).map_err(|_| {
        format!("Failed to read wit/{interface}.wit - make sure it exists next to Cargo.toml")
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

pub(crate) fn parse_resource_method_names(
    interface: &str,
    resource_name: &str,
) -> std::result::Result<Vec<String>, String> {
    let wit = read_interface_wit(interface)?;
    let mut in_resource = false;
    let mut methods = Vec::new();

    for line in wit.lines().map(str::trim) {
        if !in_resource {
            let Some(rest) = line.strip_prefix("resource ") else {
                continue;
            };
            let Some((name, _)) = rest.split_once('{') else {
                continue;
            };
            if name.trim() == resource_name {
                in_resource = true;
            }
            continue;
        }

        if line.starts_with('}') {
            break;
        }
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        if line.starts_with("constructor(") {
            methods.push("new".to_string());
            continue;
        }
        if let Some((name, _)) = line.split_once(':') {
            methods.push(name.trim().replace('-', "_"));
            continue;
        }
        if let Some((name, _)) = line.split_once('(') {
            methods.push(name.trim().replace('-', "_"));
        }
    }

    Ok(methods)
}

pub(crate) fn find_interface_for_symbol(
    symbol_name: &str,
) -> std::result::Result<Option<String>, String> {
    let mut matches = Vec::new();

    for interface in parse_world_exports()? {
        let symbols = parse_wit_interface_symbols(&interface)?;
        let resource_match = symbols
            .resources
            .iter()
            .any(|resource| to_upper_camel(resource) == symbol_name);
        let type_match = symbols
            .named_types
            .iter()
            .any(|name| to_upper_camel(name) == symbol_name);

        if resource_match || type_match {
            matches.push(interface);
        }
    }

    if matches.len() > 1 {
        return Err(format!(
            "symbol `{symbol_name}` matches multiple exported interfaces; specify `interface = ...` explicitly"
        ));
    }

    Ok(matches.pop())
}

pub(crate) fn parse_interface_function_names(
    interface: &str,
) -> std::result::Result<BTreeSet<String>, String> {
    let wit = read_interface_wit(interface)?;
    let mut functions = BTreeSet::new();
    let mut in_interface = false;
    let mut depth = 0usize;

    for raw_line in wit.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }

        if !in_interface {
            if line.starts_with("interface ") && line.ends_with('{') {
                in_interface = true;
                depth = 1;
            }
            continue;
        }

        if depth == 1 {
            if let Some(rest) = line
                .strip_prefix("resource ")
                .or_else(|| line.strip_prefix("record "))
                .or_else(|| line.strip_prefix("enum "))
                .or_else(|| line.strip_prefix("variant "))
                .or_else(|| line.strip_prefix("flags "))
                .or_else(|| line.strip_prefix("type "))
            {
                let open = rest.matches('{').count();
                let close = rest.matches('}').count();
                depth = depth + open - close;
                continue;
            }

            if let Some((name, _)) = line.split_once(':') {
                let name = name.trim();
                if !name.is_empty() {
                    functions.insert(name.replace('-', "_"));
                }
            }
        }

        depth += line.matches('{').count();
        depth = depth.saturating_sub(line.matches('}').count());
        if depth == 0 {
            break;
        }
    }

    Ok(functions)
}

pub(crate) fn find_interface_for_functions<'a>(
    function_names: impl IntoIterator<Item = &'a str>,
) -> std::result::Result<Option<String>, String> {
    let names = function_names
        .into_iter()
        .filter(|name| !name.is_empty())
        .collect::<BTreeSet<_>>();
    if names.is_empty() {
        return Ok(None);
    }

    let mut matches = Vec::new();
    for interface in parse_world_exports()? {
        let functions = parse_interface_function_names(&interface)?;
        if names.iter().all(|name| functions.contains(*name)) {
            matches.push(interface);
        }
    }

    if matches.len() > 1 {
        return Err(format!(
            "function set `{}` matches multiple exported interfaces; specify `interface = ...` explicitly",
            names.into_iter().collect::<Vec<_>>().join(", ")
        ));
    }

    Ok(matches.pop())
}
