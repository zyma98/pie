//! ComponentHub: A monotonic dynamic-linking host for WebAssembly Component Model
//!
//! This host loads WebAssembly components and dynamically links them together,
//! forwarding imports from consumer components to provider components.
//!
//! This version uses async mode for all WebAssembly operations.
//!
//! Commands (via stdin):
//!   load <path>  - Load a library WASM and register its exports
//!   run <path>   - Run an application WASM (must export `run` function)
//!   help         - Show available commands
//!   quit/exit    - Exit the host
//!
//! Libraries are persistent once loaded. Later libraries can depend on earlier ones.

use anyhow::{anyhow, Context, Result};
use std::collections::{HashMap, HashSet};
use std::io::{self, BufRead, Write};
use std::sync::Arc;
use wasmtime::component::{
    types::{ComponentItem, Type},
    Component, Func, Linker, Resource, ResourceAny, ResourceType, Val,
};
use wasmtime::{Config, Engine, Store, StoreContextMut};
use wasmtime_wasi::{ResourceTable, WasiCtx, WasiCtxBuilder, WasiCtxView, WasiView};

/// Host state stored in the Store
pub struct HostState {
    /// WASI context
    wasi_ctx: WasiCtx,
    /// Resource table for WASI
    table: ResourceTable,
    /// Maps host_rep -> provider's ResourceAny (rep is globally unique, so no need for interface/resource name in key)
    resource_map: HashMap<u32, ResourceAny>,
    /// Reverse map to preserve identity when provider returns an existing resource.
    /// ResourceAny doesn't implement Hash, so we use a linear scan here.
    provider_resource_map: Vec<(ResourceAny, u32)>,
    /// Counter for generating unique resource reps
    next_rep: u32,
}

impl HostState {
    fn new() -> Self {
        let wasi_ctx = WasiCtxBuilder::new().inherit_stdio().inherit_env().build();
        Self {
            wasi_ctx,
            table: ResourceTable::new(),
            resource_map: HashMap::new(),
            provider_resource_map: Vec::new(),
            next_rep: 1,
        }
    }

    fn alloc_rep(&mut self) -> u32 {
        let rep = self.next_rep;
        self.next_rep += 1;
        rep
    }

    fn rep_for_provider_resource(&self, resource: ResourceAny) -> Option<u32> {
        self.provider_resource_map
            .iter()
            .find(|(r, _)| *r == resource)
            .map(|(_, rep)| *rep)
    }

    fn insert_resource_mapping(&mut self, rep: u32, resource: ResourceAny) {
        self.resource_map.insert(rep, resource);
        if self.rep_for_provider_resource(resource).is_none() {
            self.provider_resource_map.push((resource, rep));
        }
    }

    fn remove_resource_mapping(&mut self, rep: u32) -> Option<ResourceAny> {
        let resource = self.resource_map.remove(&rep);
        if let Some(resource) = resource {
            self.provider_resource_map
                .retain(|(r, _)| *r != resource);
            Some(resource)
        } else {
            None
        }
    }
}

impl WasiView for HostState {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.wasi_ctx,
            table: &mut self.table,
        }
    }
}

/// Dynamic marker type for host-defined resources
/// This is a phantom type used to create host resource handles
struct DynamicResource;

/// The ComponentHub manages loading and linking components
struct ComponentHub {
    engine: Engine,
    store: Store<HostState>,
    linker: Linker<HostState>,
    loaded_libraries: Vec<String>,
}

impl ComponentHub {
    fn new() -> Result<Self> {
        let mut config = Config::new();
        config.wasm_component_model(true);
        config.async_support(true);
        let engine = Engine::new(&config)?;

        let store = Store::new(&engine, HostState::new());
        let mut linker: Linker<HostState> = Linker::new(&engine);

        // Add WASI imports (async version)
        wasmtime_wasi::p2::add_to_linker_async(&mut linker)?;

        Ok(Self {
            engine,
            store,
            linker,
            loaded_libraries: Vec::new(),
        })
    }

    /// Load a library component and register its exports
    async fn load_library(&mut self, library_path: &str) -> Result<()> {
        let lib_num = self.loaded_libraries.len() + 1;
        println!("=== Loading Library Component {}: {} ===", lib_num, library_path);

        let library_component = Component::from_file(&self.engine, library_path)
            .with_context(|| format!("failed to load library component: {}", library_path))?;

        println!("=== Instantiating Library {} ===", lib_num);
        let library_instance = self
            .linker
            .instantiate_async(&mut self.store, &library_component)
            .await
            .with_context(|| format!("failed to instantiate library: {}", library_path))?;

        println!("=== Library {} instantiated, scanning exports... ===\n", lib_num);

        // Scan library's exports and register forwarding imports
        // This makes the library's exports available to subsequent libraries and the app
        self.register_provider_exports(&library_component, library_instance)?;

        self.loaded_libraries.push(library_path.to_string());
        println!(
            "\n=== Library {} loaded successfully! ===\n",
            library_path
        );

        Ok(())
    }

    /// Run an application component
    async fn run_app(&mut self, app_path: &str) -> Result<()> {
        println!("\n=== Loading App Component: {} ===", app_path);
        let app_component = Component::from_file(&self.engine, app_path)
            .with_context(|| format!("failed to load app component: {}", app_path))?;

        // Check if app imports are satisfiable
        println!("=== Checking if app imports are satisfiable ===");
        let app_pre = self
            .linker
            .instantiate_pre(&app_component)
            .context("app imports not satisfiable")?;
        println!("App imports are satisfiable!");

        // Instantiate the app (async)
        println!("\n=== Instantiating App ===");
        let app_instance = app_pre.instantiate_async(&mut self.store).await?;

        // Call the app's run export (async)
        println!("\n=== Calling app.run() ===\n");
        let run_func = app_instance
            .get_func(&mut self.store, "run")
            .context("missing run export")?;
        run_func
            .call_async(&mut self.store, &[], &mut [])
            .await?;
        run_func.post_return_async(&mut self.store).await?;

        println!("\n=== App execution completed successfully! ===\n");
        Ok(())
    }

    /// Scan provider's exports and register forwarding implementations in the linker
    fn register_provider_exports(
        &mut self,
        provider_component: &Component,
        provider_instance: wasmtime::component::Instance,
    ) -> Result<()> {
        // Get the component's type to iterate exports
        let component_type = self.linker.substituted_component_type(provider_component)?;

        // Collect interface exports first to avoid borrow conflicts
        let interface_exports: Vec<_> = component_type
            .exports(&self.engine)
            .filter_map(|(export_name, export_item)| {
                println!(
                    "[HOST] Found export: {} ({:?})",
                    export_name,
                    std::mem::discriminant(&export_item)
                );

                match export_item {
                    ComponentItem::ComponentInstance(instance_type) => {
                        Some((export_name.to_string(), instance_type))
                    }
                    ComponentItem::ComponentFunc(_) => {
                        println!("[HOST]   -> Top-level function, skipping for now");
                        None
                    }
                    _ => {
                        println!("[HOST]   -> Other export type, skipping");
                        None
                    }
                }
            })
            .collect();

        // Now register each interface
        for (export_name, instance_type) in interface_exports {
            self.register_interface(&export_name, &instance_type, provider_instance)?;
        }

        Ok(())
    }

    /// Register an interface as host imports that forward to the provider
    fn register_interface(
        &mut self,
        interface_name: &str,
        instance_type: &wasmtime::component::types::ComponentInstance,
        provider_instance: wasmtime::component::Instance,
    ) -> Result<()> {
        println!("[HOST] Registering interface: {}", interface_name);

        // Get the interface export index from the provider
        let (_, interface_idx) = provider_instance
            .get_export(&mut self.store, None, interface_name)
            .ok_or_else(|| anyhow!("missing {} export in provider", interface_name))?;

        let mut root = self.linker.root();
        let mut inst = root.instance(interface_name)?;

        // Use Arc<str> for names to avoid cloning strings on every call
        let interface_name_arc: Arc<str> = interface_name.into();

        // Single pass: collect resources and functions, then process them.
        // We keep a map of resource name -> ResourceType to reason about ownership
        // when deciding whether to unwrap host resources for provider calls.
        let mut resources: Vec<Arc<str>> = Vec::new();
        let mut resource_type_by_name: HashMap<String, ResourceType> = HashMap::new();
        let mut functions = Vec::new();

        for (export_name, export_item) in instance_type.exports(&self.engine) {
            match export_item {
                ComponentItem::Resource(resource_type) => {
                    println!("[HOST]   Found resource: {}", export_name);
                    let resource_name_arc: Arc<str> = export_name.into();
                    resources.push(resource_name_arc.clone());
                    resource_type_by_name
                        .insert(resource_name_arc.to_string(), resource_type);

                    // Register the resource with a destructor
                    let iface = interface_name_arc.clone();
                    let res = resource_name_arc;

                    inst.resource_async(
                        export_name,
                        ResourceType::host::<DynamicResource>(),
                        move |mut store, rep| {
                            // Arc::clone is just a reference count increment - very cheap
                            let iface = iface.clone();
                            let res = res.clone();

                            Box::new(async move {
                                println!(
                                    "[HOST] Resource destructor called: {}::{} rep={}",
                                    iface, res, rep
                                );

                                // Use just rep as key - it's globally unique
                                let provider_resource =
                                    store.data_mut().remove_resource_mapping(rep);

                                if let Some(resource_any) = provider_resource {
                                    println!("[HOST] Forwarding drop to provider...");
                                    resource_any.resource_drop_async::<HostState>(&mut store).await?;
                                    println!("[HOST] Provider resource dropped successfully");
                                } else {
                                    eprintln!("[HOST] Warning: No provider resource found for rep={}", rep);
                                }

                                Ok(())
                            })
                        },
                    )?;
                }
                ComponentItem::ComponentFunc(func_type) => {
                    functions.push((export_name.to_string(), func_type));
                }
                _ => {}
            }
        }

        // Identify resources that this interface *defines* (owned). We do this by
        // scanning functions for constructor/method/static patterns that reference
        // a resource name. Imported resources will not appear in these patterns.
        let mut owned_resource_names: HashSet<String> = HashSet::new();
        for (export_name, _func_type) in functions.iter() {
            match categorize_function(export_name, &resources) {
                FuncCategory::Constructor { resource }
                | FuncCategory::Method { resource }
                | FuncCategory::StaticMethod { _resource: resource } => {
                    owned_resource_names.insert(resource.to_string());
                }
                FuncCategory::FreeFunction => {}
            }
        }

        // Convert owned resource names to the concrete ResourceType handles for
        // runtime comparisons when transforming values.
        let mut owned_resource_types: Vec<ResourceType> = Vec::new();
        for resource_name in owned_resource_names {
            if let Some(resource_type) = resource_type_by_name.get(&resource_name) {
                owned_resource_types.push(*resource_type);
            } else {
                eprintln!(
                    "[HOST] Warning: resource type not found for {}",
                    resource_name
                );
            }
        }
        let owned_resource_types = Arc::new(owned_resource_types);

        // Process functions (need to do this after resources are collected for categorization)
        for (export_name, func_type) in functions {
            println!("[HOST]   Found function: {}", export_name);

            // Look up the function export index
            let (_, func_idx) = provider_instance
                .get_export(&mut self.store, Some(&interface_idx), &export_name)
                .ok_or_else(|| anyhow!("missing {} in provider interface", export_name))?;

            // Resolve the Func handle NOW, during registration - this is the key optimization!
            let provider_func = provider_instance
                .get_func(&mut self.store, func_idx)
                .ok_or_else(|| anyhow!("{} is not a function", export_name))?;

            // Determine the function category based on naming convention
            let func_category = categorize_function(&export_name, &resources);

            let param_types: Arc<Vec<Type>> =
                Arc::new(func_type.params().map(|(_, ty)| ty).collect());
            let result_types: Arc<Vec<Type>> = Arc::new(func_type.results().collect());

            match func_category {
                FuncCategory::Constructor { resource } => {
                    Self::register_constructor(
                        &mut inst,
                        interface_name_arc.clone(),
                        &export_name,
                        resource,
                        provider_func,
                        param_types,
                        result_types,
                        owned_resource_types.clone(),
                    )?;
                }
                FuncCategory::Method { resource } => {
                    Self::register_method(
                        &mut inst,
                        interface_name_arc.clone(),
                        &export_name,
                        resource,
                        provider_func,
                        param_types,
                        result_types,
                        owned_resource_types.clone(),
                    )?;
                }
                FuncCategory::StaticMethod { .. } | FuncCategory::FreeFunction => {
                    Self::register_static_function(
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

        println!(
            "[HOST] Interface {} registered successfully",
            interface_name
        );
        Ok(())
    }

    /// Register a constructor function (async)
    fn register_constructor(
        inst: &mut wasmtime::component::LinkerInstance<'_, HostState>,
        _interface_name: Arc<str>,
        func_name: &str,
        _resource_name: Arc<str>,
        provider_func: Func,
        param_types: Arc<Vec<Type>>,
        result_types: Arc<Vec<Type>>,
        owned_resource_types: Arc<Vec<ResourceType>>,
    ) -> Result<()> {
        println!("[HOST]     -> Constructor for resource: {}", _resource_name);

        let func_name_for_log: Arc<str> = func_name.into();

        // provider_func is captured directly - no lookup needed at call time!
        // Use func_new_async for async support
        inst.func_new_async(func_name, move |mut store, args, results| {
            // Arc::clone is just a reference count increment - very cheap
            let func_name_for_log = func_name_for_log.clone();
            let param_types = Arc::clone(&param_types);
            let result_types = Arc::clone(&result_types);
            let owned_resource_types = Arc::clone(&owned_resource_types);

            Box::new(async move {
                println!(
                    "[HOST] Constructor {} called with {} args",
                    func_name_for_log,
                    args.len()
                );

                forward_call(
                    &mut store,
                    &provider_func,
                    &args,
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

    /// Register a method function (takes self as first argument) (async)
    fn register_method(
        inst: &mut wasmtime::component::LinkerInstance<'_, HostState>,
        _interface_name: Arc<str>,
        func_name: &str,
        resource_name: Arc<str>,
        provider_func: Func,
        param_types: Arc<Vec<Type>>,
        result_types: Arc<Vec<Type>>,
        owned_resource_types: Arc<Vec<ResourceType>>,
    ) -> Result<()> {
        let has_results = !result_types.is_empty();
        println!(
            "[HOST]     -> Method on {}: {} (returns: {})",
            resource_name, func_name, has_results
        );

        let func_name_for_log: Arc<str> = func_name.into();

        // provider_func is captured directly - no lookup needed at call time!
        // Use func_new_async for async support
        inst.func_new_async(func_name, move |mut store, args, results| {
            // Arc::clone is just a reference count increment - very cheap
            let func_name_for_log = func_name_for_log.clone();
            let num_results = results.len();
            let param_types = Arc::clone(&param_types);
            let result_types = Arc::clone(&result_types);
            let owned_resource_types = Arc::clone(&owned_resource_types);

            Box::new(async move {
                println!(
                    "[HOST] Method {} called with {} args",
                    func_name_for_log,
                    args.len()
                );

                forward_call(
                    &mut store,
                    &provider_func,
                    &args,
                    results,
                    &param_types,
                    &result_types,
                    &owned_resource_types,
                )
                .await?;

                if num_results == 0 {
                    println!("[HOST] Method {} completed (no results)", func_name_for_log);
                }

                Ok(())
            })
        })?;

        Ok(())
    }

    /// Register a static function (no self argument) (async)
    fn register_static_function(
        inst: &mut wasmtime::component::LinkerInstance<'_, HostState>,
        func_name: &str,
        provider_func: Func,
        param_types: Arc<Vec<Type>>,
        result_types: Arc<Vec<Type>>,
        owned_resource_types: Arc<Vec<ResourceType>>,
    ) -> Result<()> {
        let has_results = !result_types.is_empty();
        println!(
            "[HOST]     -> Static function: {} (returns: {})",
            func_name, has_results
        );

        let func_name_for_log: Arc<str> = func_name.into();

        // provider_func is captured directly - no lookup needed at call time!
        // Use func_new_async for async support
        inst.func_new_async(func_name, move |mut store, args, results| {
            // Arc::clone is just a reference count increment - very cheap
            let func_name_for_log = func_name_for_log.clone();
            let num_results = results.len();
            let param_types = Arc::clone(&param_types);
            let result_types = Arc::clone(&result_types);
            let owned_resource_types = Arc::clone(&owned_resource_types);

            Box::new(async move {
                println!(
                    "[HOST] Static function {} called with {} args",
                    func_name_for_log,
                    args.len()
                );

                forward_call(
                    &mut store,
                    &provider_func,
                    &args,
                    results,
                    &param_types,
                    &result_types,
                    &owned_resource_types,
                )
                .await?;

                if num_results == 0 {
                    println!(
                        "[HOST] Static function {} completed (no results)",
                        func_name_for_log
                    );
                }

                Ok(())
            })
        })?;

        Ok(())
    }

    /// Show the list of loaded libraries
    fn show_status(&self) {
        if self.loaded_libraries.is_empty() {
            println!("No libraries loaded.");
        } else {
            println!("Loaded libraries ({}):", self.loaded_libraries.len());
            for (i, path) in self.loaded_libraries.iter().enumerate() {
                println!("  {}. {}", i + 1, path);
            }
        }
    }

    /// Purge all loaded libraries by creating fresh store and linker
    fn purge(&mut self) -> Result<()> {
        let old_count = self.loaded_libraries.len();

        // Create fresh store with new HostState
        self.store = Store::new(&self.engine, HostState::new());

        // Create fresh linker
        self.linker = Linker::new(&self.engine);

        // Re-add WASI imports
        wasmtime_wasi::p2::add_to_linker_async(&mut self.linker)?;

        // Clear the library list
        self.loaded_libraries.clear();

        println!(
            "Purged {} libraries. All state has been reset.",
            old_count
        );
        Ok(())
    }
}

/// Categories of functions in the component model
enum FuncCategory {
    Constructor { resource: Arc<str> },
    Method { resource: Arc<str> },
    StaticMethod { _resource: Arc<str> },
    FreeFunction,
}

/// Categorize a function based on its name and the known resources
fn categorize_function(func_name: &str, resources: &[Arc<str>]) -> FuncCategory {
    // Check for constructor: [constructor]resource-name
    if let Some(resource_name) = func_name.strip_prefix("[constructor]") {
        // Try to find matching resource Arc to reuse it
        let resource = resources
            .iter()
            .find(|r| r.as_ref() == resource_name)
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
                .find(|r| r.as_ref() == resource_name)
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
                .find(|r| r.as_ref() == resource_name)
                .cloned()
                .unwrap_or_else(|| resource_name.into());
            return FuncCategory::StaticMethod { _resource: resource };
        }
    }

    // Otherwise it's a free function
    FuncCategory::FreeFunction
}

fn transform_args_for_provider(
    store: &mut StoreContextMut<'_, HostState>,
    args: &[Val],
    param_types: &[Type],
    owned_resource_types: &[ResourceType],
) -> Result<Vec<Val>> {
    // Transform arguments from host view -> provider view. Only resources that
    // belong to the provider's interface are unwrapped.
    if args.len() != param_types.len() {
        return Err(anyhow!(
            "argument count mismatch: got {}, expected {}",
            args.len(),
            param_types.len()
        ));
    }

    args.iter()
        .zip(param_types.iter())
        .map(|(val, ty)| transform_incoming_val(store, val.clone(), ty, owned_resource_types))
        .collect()
}

fn transform_results_from_provider(
    store: &mut StoreContextMut<'_, HostState>,
    results: Vec<Val>,
    result_types: &[Type],
    owned_resource_types: &[ResourceType],
) -> Result<Vec<Val>> {
    // Transform results from provider view -> host view. Only provider-owned
    // resources are wrapped into host resource handles.
    if results.len() != result_types.len() {
        return Err(anyhow!(
            "result count mismatch: got {}, expected {}",
            results.len(),
            result_types.len()
        ));
    }

    results
        .into_iter()
        .zip(result_types.iter())
        .map(|(val, ty)| transform_outgoing_val(store, val, ty, owned_resource_types))
        .collect()
}

fn transform_incoming_val(
    store: &mut StoreContextMut<'_, HostState>,
    val: Val,
    ty: &Type,
    owned_resource_types: &[ResourceType],
) -> Result<Val> {
    match ty {
        Type::Own(resource_type) | Type::Borrow(resource_type) => match val {
            Val::Resource(resource_any) => {
                // If the provider owns this resource type, unwrap the host
                // handle into the provider's ResourceAny. Otherwise keep the
                // host handle so cross-provider usage works.
                if owned_resource_types.contains(resource_type) {
                    let host_resource: Resource<DynamicResource> =
                        Resource::try_from_resource_any(resource_any, &mut *store)?;
                    let rep = host_resource.rep();
                    let provider_resource = store
                        .data()
                        .resource_map
                        .get(&rep)
                        .copied()
                        .ok_or_else(|| anyhow!("unknown resource rep={}", rep))?;
                    Ok(Val::Resource(provider_resource))
                } else {
                    // Imported resources should stay as host resources.
                    Ok(Val::Resource(resource_any))
                }
            }
            other => Err(anyhow!("expected resource for {:?}, got {:?}", ty, other)),
        },
        Type::List(list_type) => match val {
            Val::List(values) => {
                let element_type = list_type.ty();
                let transformed = values
                    .into_iter()
                    .map(|v| transform_incoming_val(store, v, &element_type, owned_resource_types))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Val::List(transformed))
            }
            other => Err(anyhow!("expected list, got {:?}", other)),
        },
        Type::Record(record_type) => match val {
            Val::Record(fields) => {
                let field_types: Vec<_> = record_type.fields().collect();
                if field_types.len() != fields.len() {
                    return Err(anyhow!(
                        "record field count mismatch: got {}, expected {}",
                        fields.len(),
                        field_types.len()
                    ));
                }
                let mut transformed = Vec::with_capacity(fields.len());
                for ((name, value), field) in fields.into_iter().zip(field_types.into_iter()) {
                    if name != field.name {
                        return Err(anyhow!(
                            "record field name mismatch: got {}, expected {}",
                            name,
                            field.name
                        ));
                    }
                    let value =
                        transform_incoming_val(store, value, &field.ty, owned_resource_types)?;
                    transformed.push((name, value));
                }
                Ok(Val::Record(transformed))
            }
            other => Err(anyhow!("expected record, got {:?}", other)),
        },
        Type::Tuple(tuple_type) => match val {
            Val::Tuple(values) => {
                let types: Vec<_> = tuple_type.types().collect();
                if types.len() != values.len() {
                    return Err(anyhow!(
                        "tuple size mismatch: got {}, expected {}",
                        values.len(),
                        types.len()
                    ));
                }
                let transformed = values
                    .into_iter()
                    .zip(types.iter())
                    .map(|(v, t)| transform_incoming_val(store, v, t, owned_resource_types))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Val::Tuple(transformed))
            }
            other => Err(anyhow!("expected tuple, got {:?}", other)),
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
                            transform_incoming_val(store, *value, &ty, owned_resource_types)?;
                        Ok(Val::Variant(case_name, Some(Box::new(inner))))
                    }
                    (None, Some(_)) => Err(anyhow!(
                        "variant {} has no payload but value provided",
                        case_name
                    )),
                    (Some(_), None) => Err(anyhow!(
                        "variant {} expects payload but none provided",
                        case_name
                    )),
                }
            }
            other => Err(anyhow!("expected variant, got {:?}", other)),
        },
        Type::Option(option_type) => match val {
            Val::Option(Some(value)) => {
                let inner =
                    transform_incoming_val(store, *value, &option_type.ty(), owned_resource_types)?;
                Ok(Val::Option(Some(Box::new(inner))))
            }
            Val::Option(None) => Ok(Val::Option(None)),
            other => Err(anyhow!("expected option, got {:?}", other)),
        },
        Type::Result(result_type) => match val {
            Val::Result(Ok(value)) => match (result_type.ok(), value) {
                (Some(ty), Some(inner)) => {
                    let inner =
                        transform_incoming_val(store, *inner, &ty, owned_resource_types)?;
                    Ok(Val::Result(Ok(Some(Box::new(inner)))))
                }
                (None, None) => Ok(Val::Result(Ok(None))),
                (None, Some(_)) => Err(anyhow!("result ok has no payload but value provided")),
                (Some(_), None) => Err(anyhow!("result ok expects payload but none provided")),
            },
            Val::Result(Err(value)) => match (result_type.err(), value) {
                (Some(ty), Some(inner)) => {
                    let inner =
                        transform_incoming_val(store, *inner, &ty, owned_resource_types)?;
                    Ok(Val::Result(Err(Some(Box::new(inner)))))
                }
                (None, None) => Ok(Val::Result(Err(None))),
                (None, Some(_)) => Err(anyhow!("result err has no payload but value provided")),
                (Some(_), None) => Err(anyhow!("result err expects payload but none provided")),
            },
            other => Err(anyhow!("expected result, got {:?}", other)),
        },
        _ => Ok(val),
    }
}

fn transform_outgoing_val(
    store: &mut StoreContextMut<'_, HostState>,
    val: Val,
    ty: &Type,
    owned_resource_types: &[ResourceType],
) -> Result<Val> {
    match ty {
        Type::Own(resource_type) | Type::Borrow(resource_type) => match val {
            Val::Resource(provider_resource) => {
                // Provider-owned resources are wrapped into host handles.
                // Imported resources are passed through unchanged.
                if owned_resource_types.contains(resource_type) {
                    // Reuse existing host rep if provider returns an already-known resource.
                    // This preserves identity and avoids double-dropping.
                    let rep = if let Some(existing) = store.data().rep_for_provider_resource(provider_resource) {
                        existing
                    } else {
                        let rep = store.data_mut().alloc_rep();
                        store
                            .data_mut()
                            .insert_resource_mapping(rep, provider_resource);
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
                    // Returning an imported resource: keep host resource as-is.
                    Ok(Val::Resource(provider_resource))
                }
            }
            other => Err(anyhow!("expected resource for {:?}, got {:?}", ty, other)),
        },
        Type::List(list_type) => match val {
            Val::List(values) => {
                let element_type = list_type.ty();
                let transformed = values
                    .into_iter()
                    .map(|v| transform_outgoing_val(store, v, &element_type, owned_resource_types))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Val::List(transformed))
            }
            other => Err(anyhow!("expected list, got {:?}", other)),
        },
        Type::Record(record_type) => match val {
            Val::Record(fields) => {
                let field_types: Vec<_> = record_type.fields().collect();
                if field_types.len() != fields.len() {
                    return Err(anyhow!(
                        "record field count mismatch: got {}, expected {}",
                        fields.len(),
                        field_types.len()
                    ));
                }
                let mut transformed = Vec::with_capacity(fields.len());
                for ((name, value), field) in fields.into_iter().zip(field_types.into_iter()) {
                    if name != field.name {
                        return Err(anyhow!(
                            "record field name mismatch: got {}, expected {}",
                            name,
                            field.name
                        ));
                    }
                    let value =
                        transform_outgoing_val(store, value, &field.ty, owned_resource_types)?;
                    transformed.push((name, value));
                }
                Ok(Val::Record(transformed))
            }
            other => Err(anyhow!("expected record, got {:?}", other)),
        },
        Type::Tuple(tuple_type) => match val {
            Val::Tuple(values) => {
                let types: Vec<_> = tuple_type.types().collect();
                if types.len() != values.len() {
                    return Err(anyhow!(
                        "tuple size mismatch: got {}, expected {}",
                        values.len(),
                        types.len()
                    ));
                }
                let transformed = values
                    .into_iter()
                    .zip(types.iter())
                    .map(|(v, t)| transform_outgoing_val(store, v, t, owned_resource_types))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Val::Tuple(transformed))
            }
            other => Err(anyhow!("expected tuple, got {:?}", other)),
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
                    (None, Some(_)) => Err(anyhow!(
                        "variant {} has no payload but value provided",
                        case_name
                    )),
                    (Some(_), None) => Err(anyhow!(
                        "variant {} expects payload but none provided",
                        case_name
                    )),
                }
            }
            other => Err(anyhow!("expected variant, got {:?}", other)),
        },
        Type::Option(option_type) => match val {
            Val::Option(Some(value)) => {
                let inner =
                    transform_outgoing_val(store, *value, &option_type.ty(), owned_resource_types)?;
                Ok(Val::Option(Some(Box::new(inner))))
            }
            Val::Option(None) => Ok(Val::Option(None)),
            other => Err(anyhow!("expected option, got {:?}", other)),
        },
        Type::Result(result_type) => match val {
            Val::Result(Ok(value)) => match (result_type.ok(), value) {
                (Some(ty), Some(inner)) => {
                    let inner =
                        transform_outgoing_val(store, *inner, &ty, owned_resource_types)?;
                    Ok(Val::Result(Ok(Some(Box::new(inner)))))
                }
                (None, None) => Ok(Val::Result(Ok(None))),
                (None, Some(_)) => Err(anyhow!("result ok has no payload but value provided")),
                (Some(_), None) => Err(anyhow!("result ok expects payload but none provided")),
            },
            Val::Result(Err(value)) => match (result_type.err(), value) {
                (Some(ty), Some(inner)) => {
                    let inner =
                        transform_outgoing_val(store, *inner, &ty, owned_resource_types)?;
                    Ok(Val::Result(Err(Some(Box::new(inner)))))
                }
                (None, None) => Ok(Val::Result(Err(None))),
                (None, Some(_)) => Err(anyhow!("result err has no payload but value provided")),
                (Some(_), None) => Err(anyhow!("result err expects payload but none provided")),
            },
            other => Err(anyhow!("expected result, got {:?}", other)),
        },
        _ => Ok(val),
    }
}

async fn forward_call(
    store: &mut StoreContextMut<'_, HostState>,
    provider_func: &Func,
    args: &[Val],
    results: &mut [Val],
    param_types: &[Type],
    result_types: &[Type],
    owned_resource_types: &[ResourceType],
) -> Result<()> {
    // Centralized call path: transform args, call provider, then transform results.
    if results.len() != result_types.len() {
        return Err(anyhow!(
            "result slot mismatch: got {}, expected {}",
            results.len(),
            result_types.len()
        ));
    }
    let provider_args = transform_args_for_provider(store, args, param_types, owned_resource_types)?;
    let mut provider_results = vec![Val::Bool(false); result_types.len()];

    provider_func
        .call_async(&mut *store, &provider_args, &mut provider_results)
        .await?;
    provider_func.post_return_async(&mut *store).await?;

    let transformed =
        transform_results_from_provider(store, provider_results, result_types, owned_resource_types)?;
    for (index, value) in transformed.into_iter().enumerate() {
        results[index] = value;
    }

    Ok(())
}

fn print_help() {
    println!("Available commands:");
    println!("  load <path>  - Load a library WASM and register its exports");
    println!("  run <path>   - Run an application WASM (must export `run` function)");
    println!("  status       - Show loaded libraries");
    println!("  purge        - Remove all loaded libraries and reset state");
    println!("  help         - Show this help message");
    println!("  quit/exit    - Exit the host");
    println!();
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  ComponentHub: Monotonic Dynamic Linking Host (Async)        ║");
    println!("║  WebAssembly Component Model with Resource Forwarding        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    print_help();

    let mut hub = ComponentHub::new()?;

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        // Print prompt
        print!("> ");
        stdout.flush()?;

        // Read line
        let mut line = String::new();
        let bytes_read = stdin.lock().read_line(&mut line)?;

        // Check for EOF
        if bytes_read == 0 {
            println!("\nGoodbye!");
            break;
        }

        let line = line.trim();

        // Skip empty lines
        if line.is_empty() {
            continue;
        }

        // Parse command
        let parts: Vec<&str> = line.splitn(2, ' ').collect();
        let command = parts[0].to_lowercase();

        match command.as_str() {
            "load" => {
                if parts.len() < 2 {
                    eprintln!("Error: load command requires a path argument");
                    eprintln!("Usage: load <path>");
                    continue;
                }
                let path = parts[1].trim();
                if let Err(e) = hub.load_library(path).await {
                    eprintln!("Error loading library: {:#}", e);
                }
            }
            "run" => {
                if parts.len() < 2 {
                    eprintln!("Error: run command requires a path argument");
                    eprintln!("Usage: run <path>");
                    continue;
                }
                let path = parts[1].trim();
                if let Err(e) = hub.run_app(path).await {
                    eprintln!("Error running app: {:#}", e);
                }
            }
            "status" => {
                hub.show_status();
            }
            "purge" => {
                if let Err(e) = hub.purge() {
                    eprintln!("Error purging: {:#}", e);
                }
            }
            "help" | "?" => {
                print_help();
            }
            "quit" | "exit" => {
                println!("Goodbye!");
                break;
            }
            _ => {
                eprintln!("Unknown command: {}", command);
                eprintln!("Type 'help' for available commands.");
            }
        }
    }

    Ok(())
}
