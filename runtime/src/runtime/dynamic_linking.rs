//! Dynamic linking support for Wasm components.
//!
//! This module provides functionality for dynamically linking Wasm component libraries
//! at runtime. It handles:
//! - Library validation and stub definitions for `instantiate_pre` checks
//! - Forwarding function calls from consumer components to provider libraries
//! - Resource handle translation between consumer and provider views
//! - Borrow tracking for cross-provider resource passing

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use wasmtime::component::types::{ComponentItem, Type};
use wasmtime::component::{Component, Func, Linker, Resource, ResourceAny, ResourceType, Val};
use wasmtime::{Engine, Store};

use super::{InstanceState, LoadedLibrary, RuntimeError, api};

/// Dynamic marker type for host-defined resources used in dynamic linking.
/// This is a phantom type used to create host resource handles.
struct DynamicResource;

/// Categories of functions in the component model
enum FuncCategory {
    Constructor { resource: String },
    Method { resource: String },
    StaticMethod { resource: String },
    FreeFunction,
}

/// Categorize a function based on its name and the known resources
fn categorize_function(func_name: &str, resources: &[String]) -> FuncCategory {
    // Check for constructor: [constructor]resource-name
    if let Some(resource_name) = func_name.strip_prefix("[constructor]") {
        let resource = resources
            .iter()
            .find(|r| r.as_str() == resource_name)
            .cloned()
            .unwrap_or_else(|| resource_name.into());
        return FuncCategory::Constructor { resource };
    }

    // Check for method: [method]resource-name.method-name
    if let Some(rest) = func_name.strip_prefix("[method]") {
        if let Some(dot_pos) = rest.find('.') {
            let resource_name = &rest[..dot_pos];
            let resource = resources
                .iter()
                .find(|r| r.as_str() == resource_name)
                .cloned()
                .unwrap_or_else(|| resource_name.into());
            return FuncCategory::Method { resource };
        }
    }

    // Check for static method: [static]resource-name.method-name
    if let Some(rest) = func_name.strip_prefix("[static]") {
        if let Some(dot_pos) = rest.find('.') {
            let resource_name = &rest[..dot_pos];
            let resource = resources
                .iter()
                .find(|r| r.as_str() == resource_name)
                .cloned()
                .unwrap_or_else(|| resource_name.into());
            return FuncCategory::StaticMethod { resource };
        }
    }

    // Otherwise it's a free function
    FuncCategory::FreeFunction
}

/// Result of transforming arguments, including borrowed resources to drop after call.
struct TransformedArgs {
    /// The transformed argument values
    args: Vec<Val>,
    /// Borrowed ResourceAny handles that need to be dropped after the call completes
    /// to signal that the borrows have ended (cross-provider borrows only)
    borrows_to_end: Vec<ResourceAny>,
}

/// Transform arguments from consumer view to provider view.
/// Only resources owned by the provider interface are unwrapped.
/// Cross-provider borrows are tracked in borrows_to_end for cleanup after the call.
fn transform_args_for_provider(
    store: &mut wasmtime::StoreContextMut<'_, InstanceState>,
    args: &[Val],
    param_types: &[Type],
    owned_resource_types: &[ResourceType],
) -> Result<TransformedArgs, wasmtime::Error> {
    if args.len() != param_types.len() {
        return Err(wasmtime::Error::msg(format!(
            "argument count mismatch: got {}, expected {}",
            args.len(),
            param_types.len()
        )));
    }

    let mut borrows_to_end = Vec::new();
    let mut transformed_args = Vec::with_capacity(args.len());

    for (val, ty) in args.iter().zip(param_types.iter()) {
        let transformed = transform_incoming_val_with_borrow_tracking(
            store,
            val.clone(),
            ty,
            owned_resource_types,
            &mut borrows_to_end,
        )?;
        transformed_args.push(transformed);
    }

    Ok(TransformedArgs {
        args: transformed_args,
        borrows_to_end,
    })
}

/// Transform results from provider view to consumer view.
/// Only provider-owned resources are wrapped into host handles.
fn transform_results_from_provider(
    store: &mut wasmtime::StoreContextMut<'_, InstanceState>,
    results: Vec<Val>,
    result_types: &[Type],
    owned_resource_types: &[ResourceType],
) -> Result<Vec<Val>, wasmtime::Error> {
    if results.len() != result_types.len() {
        return Err(wasmtime::Error::msg(format!(
            "result count mismatch: got {}, expected {}",
            results.len(),
            result_types.len()
        )));
    }

    results
        .into_iter()
        .zip(result_types.iter())
        .map(|(val, ty)| transform_outgoing_val(store, val, ty, owned_resource_types))
        .collect()
}

/// Transform an incoming value, collecting any cross-provider borrows that need to be
/// ended after the call completes. This function recursively processes composite types
/// to find all nested borrowed resources.
///
/// Borrow tracking is essential for cross-provider calls:
/// - Same-provider borrow: Provider owns the resource type, unwrap host handle to ResourceAny
/// - Cross-provider borrow: Provider imports the resource type, track for cleanup after call
fn transform_incoming_val_with_borrow_tracking(
    store: &mut wasmtime::StoreContextMut<'_, InstanceState>,
    val: Val,
    ty: &Type,
    owned_resource_types: &[ResourceType],
    borrows_to_end: &mut Vec<ResourceAny>,
) -> Result<Val, wasmtime::Error> {
    match ty {
        Type::Borrow(resource_type) => match val {
            Val::Resource(resource_any) => {
                // For borrowed resources, we need to track cross-provider borrows
                // so we can end them when the call completes.
                if owned_resource_types.contains(resource_type) {
                    // Provider owns this resource type - unwrap the host handle
                    let host_resource: Resource<DynamicResource> =
                        Resource::try_from_resource_any(resource_any, &mut *store)?;
                    let rep = host_resource.rep();
                    let provider_resource = store
                        .data()
                        .dynamic_resource_map
                        .get(&rep)
                        .copied()
                        .ok_or_else(|| {
                            wasmtime::Error::msg(format!("unknown resource rep={}", rep))
                        })?;
                    Ok(Val::Resource(provider_resource))
                } else {
                    // Cross-provider borrow: pass through, but track for cleanup.
                    // We need to drop this ResourceAny after the call to end the borrow.
                    borrows_to_end.push(resource_any);
                    Ok(Val::Resource(resource_any))
                }
            }
            other => Err(wasmtime::Error::msg(format!(
                "expected resource for borrow {:?}, got {:?}",
                ty, other
            ))),
        },
        Type::Own(resource_type) => match val {
            Val::Resource(resource_any) => {
                // For owned resources, ownership transfers - no borrow tracking needed
                if owned_resource_types.contains(resource_type) {
                    let host_resource: Resource<DynamicResource> =
                        Resource::try_from_resource_any(resource_any, &mut *store)?;
                    let rep = host_resource.rep();
                    let provider_resource = store
                        .data()
                        .dynamic_resource_map
                        .get(&rep)
                        .copied()
                        .ok_or_else(|| {
                            wasmtime::Error::msg(format!("unknown resource rep={}", rep))
                        })?;
                    Ok(Val::Resource(provider_resource))
                } else {
                    // Cross-provider owned resource - ownership transfers
                    Ok(Val::Resource(resource_any))
                }
            }
            other => Err(wasmtime::Error::msg(format!(
                "expected resource for own {:?}, got {:?}",
                ty, other
            ))),
        },
        // For composite types, recursively transform and collect any nested borrows
        Type::List(list_type) => match val {
            Val::List(values) => {
                let element_type = list_type.ty();
                let transformed = values
                    .into_iter()
                    .map(|v| {
                        transform_incoming_val_with_borrow_tracking(
                            store,
                            v,
                            &element_type,
                            owned_resource_types,
                            borrows_to_end,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Val::List(transformed))
            }
            other => Err(wasmtime::Error::msg(format!(
                "expected list, got {:?}",
                other
            ))),
        },
        Type::Record(record_type) => match val {
            Val::Record(fields) => {
                let field_types: Vec<_> = record_type.fields().collect();
                if field_types.len() != fields.len() {
                    return Err(wasmtime::Error::msg(format!(
                        "record field count mismatch: got {}, expected {}",
                        fields.len(),
                        field_types.len()
                    )));
                }
                let mut transformed = Vec::with_capacity(fields.len());
                for ((name, value), field) in fields.into_iter().zip(field_types.into_iter()) {
                    if name != field.name {
                        return Err(wasmtime::Error::msg(format!(
                            "record field name mismatch: got {}, expected {}",
                            name, field.name
                        )));
                    }
                    let value = transform_incoming_val_with_borrow_tracking(
                        store,
                        value,
                        &field.ty,
                        owned_resource_types,
                        borrows_to_end,
                    )?;
                    transformed.push((name, value));
                }
                Ok(Val::Record(transformed))
            }
            other => Err(wasmtime::Error::msg(format!(
                "expected record, got {:?}",
                other
            ))),
        },
        Type::Tuple(tuple_type) => match val {
            Val::Tuple(values) => {
                let types: Vec<_> = tuple_type.types().collect();
                if types.len() != values.len() {
                    return Err(wasmtime::Error::msg(format!(
                        "tuple size mismatch: got {}, expected {}",
                        values.len(),
                        types.len()
                    )));
                }
                let transformed = values
                    .into_iter()
                    .zip(types.iter())
                    .map(|(v, t)| {
                        transform_incoming_val_with_borrow_tracking(
                            store,
                            v,
                            t,
                            owned_resource_types,
                            borrows_to_end,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Val::Tuple(transformed))
            }
            other => Err(wasmtime::Error::msg(format!(
                "expected tuple, got {:?}",
                other
            ))),
        },
        Type::Variant(variant_type) => match val {
            Val::Variant(case_name, payload) => {
                let mut case_type = None;
                for case in variant_type.cases() {
                    if case.name == case_name {
                        case_type = case.ty;
                        break;
                    }
                }
                match (case_type, payload) {
                    (None, None) => Ok(Val::Variant(case_name, None)),
                    (Some(ty), Some(value)) => {
                        let inner = transform_incoming_val_with_borrow_tracking(
                            store,
                            *value,
                            &ty,
                            owned_resource_types,
                            borrows_to_end,
                        )?;
                        Ok(Val::Variant(case_name, Some(Box::new(inner))))
                    }
                    (None, Some(_)) => Err(wasmtime::Error::msg(format!(
                        "variant {} has no payload but value provided",
                        case_name
                    ))),
                    (Some(_), None) => Err(wasmtime::Error::msg(format!(
                        "variant {} expects payload but none provided",
                        case_name
                    ))),
                }
            }
            other => Err(wasmtime::Error::msg(format!(
                "expected variant, got {:?}",
                other
            ))),
        },
        Type::Option(option_type) => match val {
            Val::Option(Some(value)) => {
                let inner = transform_incoming_val_with_borrow_tracking(
                    store,
                    *value,
                    &option_type.ty(),
                    owned_resource_types,
                    borrows_to_end,
                )?;
                Ok(Val::Option(Some(Box::new(inner))))
            }
            Val::Option(None) => Ok(Val::Option(None)),
            other => Err(wasmtime::Error::msg(format!(
                "expected option, got {:?}",
                other
            ))),
        },
        Type::Result(result_type) => match val {
            Val::Result(Ok(value)) => match (result_type.ok(), value) {
                (Some(ty), Some(inner)) => {
                    let inner = transform_incoming_val_with_borrow_tracking(
                        store,
                        *inner,
                        &ty,
                        owned_resource_types,
                        borrows_to_end,
                    )?;
                    Ok(Val::Result(Ok(Some(Box::new(inner)))))
                }
                (None, None) => Ok(Val::Result(Ok(None))),
                (None, Some(_)) => Err(wasmtime::Error::msg(
                    "result ok has no payload but value provided",
                )),
                (Some(_), None) => Err(wasmtime::Error::msg(
                    "result ok expects payload but none provided",
                )),
            },
            Val::Result(Err(value)) => match (result_type.err(), value) {
                (Some(ty), Some(inner)) => {
                    let inner = transform_incoming_val_with_borrow_tracking(
                        store,
                        *inner,
                        &ty,
                        owned_resource_types,
                        borrows_to_end,
                    )?;
                    Ok(Val::Result(Err(Some(Box::new(inner)))))
                }
                (None, None) => Ok(Val::Result(Err(None))),
                (None, Some(_)) => Err(wasmtime::Error::msg(
                    "result err has no payload but value provided",
                )),
                (Some(_), None) => Err(wasmtime::Error::msg(
                    "result err expects payload but none provided",
                )),
            },
            other => Err(wasmtime::Error::msg(format!(
                "expected result, got {:?}",
                other
            ))),
        },
        // For primitive types, no transformation or borrow tracking needed
        _ => Ok(val),
    }
}

/// Transform a single value from provider view to consumer view (outgoing).
/// Recursively handles container types (option, result, list, record, tuple, variant).
fn transform_outgoing_val(
    store: &mut wasmtime::StoreContextMut<'_, InstanceState>,
    val: Val,
    ty: &Type,
    owned_resource_types: &[ResourceType],
) -> Result<Val, wasmtime::Error> {
    match ty {
        Type::Own(resource_type) | Type::Borrow(resource_type) => match val {
            Val::Resource(provider_resource) => {
                // Provider-owned resources are wrapped into host handles.
                // Imported resources are passed through unchanged.
                if owned_resource_types.contains(resource_type) {
                    // Reuse existing host rep if provider returns an already-known resource.
                    // This preserves identity and avoids double-dropping.
                    let rep = if let Some(existing) =
                        store.data().rep_for_provider_resource(provider_resource)
                    {
                        existing
                    } else {
                        let rep = store.data_mut().alloc_dynamic_rep();
                        store
                            .data_mut()
                            .insert_dynamic_resource_mapping(rep, provider_resource);
                        rep
                    };
                    let host_resource = match ty {
                        Type::Borrow(_) => Resource::<DynamicResource>::new_borrow(rep),
                        _ => Resource::<DynamicResource>::new_own(rep),
                    };
                    let host_resource_any =
                        ResourceAny::try_from_resource(host_resource, &mut *store)?;
                    Ok(Val::Resource(host_resource_any))
                } else {
                    // Returning an imported resource: keep as-is.
                    Ok(Val::Resource(provider_resource))
                }
            }
            other => Err(wasmtime::Error::msg(format!(
                "expected resource for {:?}, got {:?}",
                ty, other
            ))),
        },
        Type::List(list_type) => match val {
            Val::List(values) => {
                let element_type = list_type.ty();
                let transformed = values
                    .into_iter()
                    .map(|v| transform_outgoing_val(store, v, &element_type, owned_resource_types))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Val::List(transformed))
            }
            other => Err(wasmtime::Error::msg(format!(
                "expected list, got {:?}",
                other
            ))),
        },
        Type::Record(record_type) => match val {
            Val::Record(fields) => {
                let field_types: Vec<_> = record_type.fields().collect();
                if field_types.len() != fields.len() {
                    return Err(wasmtime::Error::msg(format!(
                        "record field count mismatch: got {}, expected {}",
                        fields.len(),
                        field_types.len()
                    )));
                }
                let mut transformed = Vec::with_capacity(fields.len());
                for ((name, value), field) in fields.into_iter().zip(field_types.into_iter()) {
                    if name != field.name {
                        return Err(wasmtime::Error::msg(format!(
                            "record field name mismatch: got {}, expected {}",
                            name, field.name
                        )));
                    }
                    let value =
                        transform_outgoing_val(store, value, &field.ty, owned_resource_types)?;
                    transformed.push((name, value));
                }
                Ok(Val::Record(transformed))
            }
            other => Err(wasmtime::Error::msg(format!(
                "expected record, got {:?}",
                other
            ))),
        },
        Type::Tuple(tuple_type) => match val {
            Val::Tuple(values) => {
                let types: Vec<_> = tuple_type.types().collect();
                if types.len() != values.len() {
                    return Err(wasmtime::Error::msg(format!(
                        "tuple size mismatch: got {}, expected {}",
                        values.len(),
                        types.len()
                    )));
                }
                let transformed = values
                    .into_iter()
                    .zip(types.iter())
                    .map(|(v, t)| transform_outgoing_val(store, v, t, owned_resource_types))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Val::Tuple(transformed))
            }
            other => Err(wasmtime::Error::msg(format!(
                "expected tuple, got {:?}",
                other
            ))),
        },
        Type::Variant(variant_type) => match val {
            Val::Variant(case_name, payload) => {
                let mut case_type = None;
                for case in variant_type.cases() {
                    if case.name == case_name {
                        case_type = case.ty;
                        break;
                    }
                }
                match (case_type, payload) {
                    (None, None) => Ok(Val::Variant(case_name, None)),
                    (Some(ty), Some(value)) => {
                        let inner =
                            transform_outgoing_val(store, *value, &ty, owned_resource_types)?;
                        Ok(Val::Variant(case_name, Some(Box::new(inner))))
                    }
                    (None, Some(_)) => Err(wasmtime::Error::msg(format!(
                        "variant {} has no payload but value provided",
                        case_name
                    ))),
                    (Some(_), None) => Err(wasmtime::Error::msg(format!(
                        "variant {} expects payload but none provided",
                        case_name
                    ))),
                }
            }
            other => Err(wasmtime::Error::msg(format!(
                "expected variant, got {:?}",
                other
            ))),
        },
        Type::Option(option_type) => match val {
            Val::Option(Some(value)) => {
                let inner =
                    transform_outgoing_val(store, *value, &option_type.ty(), owned_resource_types)?;
                Ok(Val::Option(Some(Box::new(inner))))
            }
            Val::Option(None) => Ok(Val::Option(None)),
            other => Err(wasmtime::Error::msg(format!(
                "expected option, got {:?}",
                other
            ))),
        },
        Type::Result(result_type) => match val {
            Val::Result(Ok(value)) => match (result_type.ok(), value) {
                (Some(ty), Some(inner)) => {
                    let inner = transform_outgoing_val(store, *inner, &ty, owned_resource_types)?;
                    Ok(Val::Result(Ok(Some(Box::new(inner)))))
                }
                (None, None) => Ok(Val::Result(Ok(None))),
                (None, Some(_)) => Err(wasmtime::Error::msg(
                    "result ok has no payload but value provided",
                )),
                (Some(_), None) => Err(wasmtime::Error::msg(
                    "result ok expects payload but none provided",
                )),
            },
            Val::Result(Err(value)) => match (result_type.err(), value) {
                (Some(ty), Some(inner)) => {
                    let inner = transform_outgoing_val(store, *inner, &ty, owned_resource_types)?;
                    Ok(Val::Result(Err(Some(Box::new(inner)))))
                }
                (None, None) => Ok(Val::Result(Err(None))),
                (None, Some(_)) => Err(wasmtime::Error::msg(
                    "result err has no payload but value provided",
                )),
                (Some(_), None) => Err(wasmtime::Error::msg(
                    "result err expects payload but none provided",
                )),
            },
            other => Err(wasmtime::Error::msg(format!(
                "expected result, got {:?}",
                other
            ))),
        },
        // Primitive types pass through unchanged
        _ => Ok(val),
    }
}

/// Centralized call forwarding: transform args, call provider, transform results, end borrows.
///
/// This function handles borrow tracking for cross-provider resource passing:
/// 1. Transform args and collect any cross-provider borrows
/// 2. Call the provider function
/// 3. End any cross-provider borrows by dropping their ResourceAny handles
/// 4. Transform results back to consumer view
async fn forward_call(
    store: &mut wasmtime::StoreContextMut<'_, InstanceState>,
    provider_func: &Func,
    args: &[Val],
    results: &mut [Val],
    param_types: &[Type],
    result_types: &[Type],
    owned_resource_types: &[ResourceType],
) -> Result<(), wasmtime::Error> {
    if results.len() != result_types.len() {
        return Err(wasmtime::Error::msg(format!(
            "result slot mismatch: got {}, expected {}",
            results.len(),
            result_types.len()
        )));
    }

    // Transform arguments and collect any cross-provider borrows that need cleanup
    let TransformedArgs {
        args: provider_args,
        borrows_to_end,
    } = transform_args_for_provider(store, args, param_types, owned_resource_types)?;

    let mut provider_results = vec![Val::Bool(false); result_types.len()];

    provider_func
        .call_async(&mut *store, &provider_args, &mut provider_results)
        .await?;
    provider_func.post_return_async(&mut *store).await?;

    // End any cross-provider borrows by dropping their ResourceAny handles.
    // This signals to the Component Model runtime that the borrows have completed.
    // This must happen after the call returns but before we return to the caller.
    for borrow in borrows_to_end {
        borrow
            .resource_drop_async::<InstanceState>(&mut *store)
            .await?;
    }

    let transformed = transform_results_from_provider(
        store,
        provider_results,
        result_types,
        owned_resource_types,
    )?;
    for (index, value) in transformed.into_iter().enumerate() {
        results[index] = value;
    }

    Ok(())
}

/// Add stub definitions for all exports of a component to the linker.
///
/// This scans the component's exports and creates stub functions with matching
/// signatures. The stubs are never actually called - they're only used for
/// validation via `instantiate_pre`.
pub(super) fn add_stub_definitions_for_component(
    engine: &Engine,
    linker: &mut Linker<InstanceState>,
    component: &Component,
) -> Result<(), wasmtime::Error> {
    // Get the component's type to iterate exports
    let component_type = linker.substituted_component_type(component)?;

    // Iterate directly over interface exports without collecting into a vector
    for (export_name, export_item) in component_type.exports(engine) {
        if let ComponentItem::ComponentInstance(instance_type) = export_item {
            add_stub_definitions_for_interface(engine, linker, &export_name, &instance_type)?;
        }
    }

    Ok(())
}

/// Add stub definitions for an interface to the linker.
fn add_stub_definitions_for_interface(
    engine: &Engine,
    linker: &mut Linker<InstanceState>,
    interface_name: &str,
    instance_type: &wasmtime::component::types::ComponentInstance,
) -> Result<(), wasmtime::Error> {
    // Try to get or create the instance; if it fails, the interface is already defined
    let mut root = linker.root();
    let mut inst = match root.instance(interface_name) {
        Ok(inst) => inst,
        Err(_) => return Ok(()), // Interface already fully defined, skip
    };

    // Register both resources and functions
    for (export_name, export_item) in instance_type.exports(engine) {
        match export_item {
            ComponentItem::Resource(_) => {
                // Register a stub resource with a no-op destructor
                // Silently skip if already defined
                let _ = inst.resource_async(
                    export_name,
                    ResourceType::host::<DynamicResource>(),
                    |_store, _rep| Box::new(async { Ok(()) }),
                );
            }
            ComponentItem::ComponentFunc(_) => {
                // Create a stub function that matches the signature.
                // Since we're only validating via instantiate_pre, these stubs will never
                // actually be called - they just need to exist with matching signatures.
                let func_name = export_name.to_string();

                // Silently skip if already defined
                let _ = inst.func_new_async(export_name, move |_store, _args, _results| {
                    let func_name = func_name.clone();
                    Box::new(async move {
                        // This stub should never be called during validation.
                        // instantiate_pre only checks that imports are defined,
                        // it doesn't actually call them.
                        unreachable!(
                            "Stub function '{}' was unexpectedly called during validation",
                            func_name
                        );
                    })
                });
            }
            _ => {}
        }
    }

    Ok(())
}

/// Validate that a library component's imports can be satisfied by the host
/// interfaces and the declared dependencies.
///
/// This creates a validation linker with:
/// 1. Host-defined interfaces (WASI, HTTP, Pie API)
/// 2. Stub definitions for all exports from dependencies (recursively)
///
/// Then uses `instantiate_pre` to verify that all imports are satisfiable.
pub(super) fn validate_library_imports(
    engine: &Engine,
    loaded_libraries: &HashMap<String, LoadedLibrary>,
    component: &Component,
    dependencies: &[String],
) -> Result<(), RuntimeError> {
    // Create a fresh linker for validation
    let mut validation_linker = Linker::<InstanceState>::new(engine);

    // Add host-defined interfaces
    wasmtime_wasi::p2::add_to_linker_async(&mut validation_linker)
        .map_err(|e| RuntimeError::Other(format!("Failed to link WASI: {e}")))?;
    wasmtime_wasi_http::add_only_http_to_linker_async(&mut validation_linker)
        .map_err(|e| RuntimeError::Other(format!("Failed to link WASI HTTP: {e}")))?;
    api::add_to_linker(&mut validation_linker)
        .map_err(|e| RuntimeError::Other(format!("Failed to link Pie API: {e}")))?;

    // Collect all dependencies (recursively) in topological order
    let all_deps = collect_recursive_dependencies(loaded_libraries, dependencies)?;

    // Add stub definitions for all dependency exports
    for dep_name in &all_deps {
        if let Some(dep_lib) = loaded_libraries.get(dep_name) {
            add_stub_definitions_for_component(engine, &mut validation_linker, &dep_lib.component)
                .map_err(|e| {
                    RuntimeError::Other(format!(
                        "Failed to add stub definitions for '{}': {e}",
                        dep_name
                    ))
                })?;
        }
    }

    // Try to create an InstancePre to verify all imports are satisfiable
    validation_linker.instantiate_pre(component).map_err(|e| {
        RuntimeError::UnsatisfiableImports(format!(
            "Library imports not satisfiable. \
             Declared dependencies: [{}]. \
             Available dependencies: [{}]. \
             Error: {e}",
            dependencies.join(", "),
            all_deps.join(", ")
        ))
    })?;

    Ok(())
}

/// Collect all dependencies recursively in topological order (dependencies before dependents).
pub(super) fn collect_recursive_dependencies(
    loaded_libraries: &HashMap<String, LoadedLibrary>,
    direct_deps: &[String],
) -> Result<Vec<String>, RuntimeError> {
    let mut result = Vec::new();
    let mut visited = HashSet::new();

    fn visit(
        dep_name: &str,
        loaded_libraries: &HashMap<String, LoadedLibrary>,
        result: &mut Vec<String>,
        visited: &mut HashSet<String>,
    ) -> Result<(), RuntimeError> {
        if visited.contains(dep_name) {
            return Ok(());
        }
        visited.insert(dep_name.to_string());

        if let Some(lib) = loaded_libraries.get(dep_name) {
            // Visit transitive dependencies first
            for transitive_dep in &lib.dependencies {
                visit(transitive_dep, loaded_libraries, result, visited)?;
            }
        }

        result.push(dep_name.to_string());
        Ok(())
    }

    for dep in direct_deps {
        visit(dep, loaded_libraries, &mut result, &mut visited)?;
    }

    Ok(result)
}

/// Register forwarding implementations for a library's exports.
/// This scans the library component's exports and registers functions that forward calls
/// to the library instance.
fn register_library_exports(
    engine: &Engine,
    linker: &mut Linker<InstanceState>,
    store: &mut Store<InstanceState>,
    library_component: &Component,
    library_instance: wasmtime::component::Instance,
) -> Result<(), wasmtime::Error> {
    // Get the component's type to iterate exports
    let component_type = linker.substituted_component_type(library_component)?;

    // Iterate over interface exports
    for (export_name, export_item) in component_type.exports(engine) {
        if let ComponentItem::ComponentInstance(instance_type) = export_item {
            register_interface_exports(
                engine,
                linker,
                store,
                &export_name,
                &instance_type,
                library_instance,
            )?;
        }
    }

    Ok(())
}

/// Register forwarding implementations for an interface.
fn register_interface_exports(
    engine: &Engine,
    linker: &mut Linker<InstanceState>,
    store: &mut Store<InstanceState>,
    interface_name: &str,
    instance_type: &wasmtime::component::types::ComponentInstance,
    library_instance: wasmtime::component::Instance,
) -> Result<(), wasmtime::Error> {
    // Get the interface export index from the library
    let (_, interface_idx) = match library_instance.get_export(&mut *store, None, interface_name) {
        Some(idx) => idx,
        None => return Ok(()), // Interface not exported, skip
    };

    let mut root = linker.root();
    let mut inst = match root.instance(interface_name) {
        Ok(inst) => inst,
        Err(_) => return Ok(()), // Interface already fully defined, skip
    };

    // Collect resources and functions in a single pass
    let mut resources: Vec<String> = Vec::new();
    let mut resource_type_by_name: HashMap<String, ResourceType> = HashMap::new();
    let mut functions = Vec::new();

    for (export_name, export_item) in instance_type.exports(engine) {
        match export_item {
            ComponentItem::Resource(resource_type) => {
                let resource_name: String = export_name.into();
                resources.push(resource_name.clone());
                resource_type_by_name.insert(resource_name, resource_type);

                // Register a stub resource with a destructor that forwards to the library
                let _ = inst.resource_async(
                    export_name,
                    ResourceType::host::<DynamicResource>(),
                    move |mut store, rep| {
                        Box::new(async move {
                            // Look up and remove the provider resource from the resource map
                            let provider_resource =
                                store.data_mut().remove_dynamic_resource_mapping(rep);

                            if let Some(resource_any) = provider_resource {
                                resource_any
                                    .resource_drop_async::<InstanceState>(&mut store)
                                    .await?;
                            }

                            Ok(())
                        })
                    },
                );
            }
            ComponentItem::ComponentFunc(func_type) => {
                functions.push((export_name.to_string(), func_type));
            }
            _ => {}
        }
    }

    // Identify resources that this interface *defines* (owned) by scanning function patterns.
    // Imported resources will not appear in constructor/method/static patterns.
    let mut owned_resource_names: HashSet<String> = HashSet::new();
    for (export_name, _func_type) in functions.iter() {
        match categorize_function(export_name, &resources) {
            FuncCategory::Constructor { resource }
            | FuncCategory::Method { resource }
            | FuncCategory::StaticMethod { resource } => {
                owned_resource_names.insert(resource.to_string());
            }
            FuncCategory::FreeFunction => {}
        }
    }

    // Convert owned resource names to concrete ResourceType handles for runtime comparisons
    let mut owned_resource_types: Vec<ResourceType> = Vec::new();
    for resource_name in &owned_resource_names {
        if let Some(resource_type) = resource_type_by_name.get(resource_name) {
            owned_resource_types.push(*resource_type);
        }
    }
    let owned_resource_types = Arc::new(owned_resource_types);

    // Process functions
    for (export_name, func_type) in functions {
        // Look up the function export index
        let (_, func_idx) =
            match library_instance.get_export(&mut *store, Some(&interface_idx), &export_name) {
                Some(idx) => idx,
                None => continue, // Function not found, skip
            };

        // Resolve the Func handle
        let provider_func = match library_instance.get_func(&mut *store, &func_idx) {
            Some(f) => f,
            None => continue, // Not a function, skip
        };

        // Collect param and result types
        let param_types: Arc<Vec<Type>> = Arc::new(func_type.params().map(|(_, ty)| ty).collect());
        let result_types: Arc<Vec<Type>> = Arc::new(func_type.results().collect());

        // Categorize the function
        let func_category = categorize_function(&export_name, &resources);

        match func_category {
            FuncCategory::Constructor { resource: _ } => {
                register_constructor_forwarding(
                    &mut inst,
                    &export_name,
                    provider_func,
                    param_types,
                    result_types,
                    owned_resource_types.clone(),
                )?;
            }
            FuncCategory::Method { resource: _ } => {
                register_method_forwarding(
                    &mut inst,
                    &export_name,
                    provider_func,
                    param_types,
                    result_types,
                    owned_resource_types.clone(),
                )?;
            }
            FuncCategory::StaticMethod { .. } | FuncCategory::FreeFunction => {
                register_static_function_forwarding(
                    &mut inst,
                    &export_name,
                    provider_func,
                    param_types,
                    result_types,
                    owned_resource_types.clone(),
                )?;
            }
        }
    }

    Ok(())
}

/// Register a constructor function that forwards to the library.
/// This now uses forward_call to handle Result<Self, T> and Option<Self> return types.
fn register_constructor_forwarding(
    inst: &mut wasmtime::component::LinkerInstance<'_, InstanceState>,
    func_name: &str,
    provider_func: Func,
    param_types: Arc<Vec<Type>>,
    result_types: Arc<Vec<Type>>,
    owned_resource_types: Arc<Vec<ResourceType>>,
) -> Result<(), wasmtime::Error> {
    inst.func_new_async(func_name, move |mut store, args, results| {
        let param_types = Arc::clone(&param_types);
        let result_types = Arc::clone(&result_types);
        let owned_resource_types = Arc::clone(&owned_resource_types);

        Box::new(async move {
            forward_call(
                &mut store,
                &provider_func,
                args,
                results,
                &param_types,
                &result_types,
                &owned_resource_types,
            )
            .await
        })
    })?;

    Ok(())
}

/// Register a method function that forwards to the library.
/// This uses forward_call to properly transform all resource arguments and return values.
fn register_method_forwarding(
    inst: &mut wasmtime::component::LinkerInstance<'_, InstanceState>,
    func_name: &str,
    provider_func: Func,
    param_types: Arc<Vec<Type>>,
    result_types: Arc<Vec<Type>>,
    owned_resource_types: Arc<Vec<ResourceType>>,
) -> Result<(), wasmtime::Error> {
    inst.func_new_async(func_name, move |mut store, args, results| {
        let param_types = Arc::clone(&param_types);
        let result_types = Arc::clone(&result_types);
        let owned_resource_types = Arc::clone(&owned_resource_types);

        Box::new(async move {
            forward_call(
                &mut store,
                &provider_func,
                args,
                results,
                &param_types,
                &result_types,
                &owned_resource_types,
            )
            .await
        })
    })?;

    Ok(())
}

/// Register a static function that forwards to the library.
/// This uses forward_call to properly transform all resource arguments and return values.
fn register_static_function_forwarding(
    inst: &mut wasmtime::component::LinkerInstance<'_, InstanceState>,
    func_name: &str,
    provider_func: Func,
    param_types: Arc<Vec<Type>>,
    result_types: Arc<Vec<Type>>,
    owned_resource_types: Arc<Vec<ResourceType>>,
) -> Result<(), wasmtime::Error> {
    inst.func_new_async(func_name, move |mut store, args, results| {
        let param_types = Arc::clone(&param_types);
        let result_types = Arc::clone(&result_types);
        let owned_resource_types = Arc::clone(&owned_resource_types);

        Box::new(async move {
            forward_call(
                &mut store,
                &provider_func,
                args,
                results,
                &param_types,
                &result_types,
                &owned_resource_types,
            )
            .await
        })
    })?;

    Ok(())
}

/// Instantiate library components and register their exports in the linker.
///
/// This iterates through library components in dependency order, instantiates each one,
/// and registers forwarding implementations for their exports so that subsequent
/// components (including the main program) can import them.
pub(super) async fn instantiate_libraries(
    engine: &Engine,
    linker: &mut Linker<InstanceState>,
    store: &mut Store<InstanceState>,
    library_components: Vec<(String, Component)>,
) -> Result<(), RuntimeError> {
    for (lib_name, lib_component) in library_components {
        let lib_instance = linker
            .instantiate_async(&mut *store, &lib_component)
            .await
            .map_err(|e| {
                RuntimeError::Other(format!("Failed to instantiate library '{}': {e}", lib_name))
            })?;

        // Register forwarding implementations for this library's exports
        register_library_exports(engine, linker, store, &lib_component, lib_instance).map_err(
            |e| {
                RuntimeError::Other(format!(
                    "Failed to register exports for library '{}': {e}",
                    lib_name
                ))
            },
        )?;
    }

    Ok(())
}
