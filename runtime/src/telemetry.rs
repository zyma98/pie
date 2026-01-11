//! OpenTelemetry telemetry module for Pie runtime.
//!
//! This module provides OTLP tracing export to SigNoz.
//! Telemetry is controlled via the `[telemetry]` section in pie config.

use opentelemetry::trace::TracerProvider as _;
use opentelemetry::KeyValue;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::trace::TracerProvider;
use opentelemetry_sdk::{runtime, Resource};
use std::sync::atomic::{AtomicBool, Ordering};

/// Whether telemetry is enabled (runtime toggle)
static TELEMETRY_ENABLED: AtomicBool = AtomicBool::new(false);

/// Configuration for telemetry
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    pub enabled: bool,
    pub endpoint: String,
    pub service_name: String,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: "http://localhost:4317".to_string(),
            service_name: "pie-runtime".to_string(),
        }
    }
}

/// Initialize OpenTelemetry OTLP tracing.
///
/// Returns an optional tracing-opentelemetry layer that can be added to the subscriber.
/// If telemetry is disabled, returns None.
pub fn init_otel_layer<S>(
    config: &TelemetryConfig,
) -> Option<tracing_opentelemetry::OpenTelemetryLayer<S, opentelemetry_sdk::trace::Tracer>>
where
    S: tracing::Subscriber + for<'span> tracing_subscriber::registry::LookupSpan<'span>,
{
    if !config.enabled {
        TELEMETRY_ENABLED.store(false, Ordering::SeqCst);
        return None;
    }

    TELEMETRY_ENABLED.store(true, Ordering::SeqCst);

    // Build OTLP exporter
    let exporter = match opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(&config.endpoint)
        .build()
    {
        Ok(e) => e,
        Err(err) => {
            eprintln!("Failed to create OTLP exporter: {}, telemetry disabled", err);
            TELEMETRY_ENABLED.store(false, Ordering::SeqCst);
            return None;
        }
    };

    // Build resource with service name
    let resource = Resource::new(vec![
        KeyValue::new("service.name", config.service_name.clone()),
    ]);

    // Build tracer provider
    let provider = TracerProvider::builder()
        .with_resource(resource)
        .with_batch_exporter(exporter, runtime::Tokio)
        .build();

    // Get a tracer
    let tracer = provider.tracer("pie-runtime");

    // Store the provider globally so it stays alive
    opentelemetry::global::set_tracer_provider(provider);

    Some(tracing_opentelemetry::layer().with_tracer(tracer))
}

/// Check if telemetry is enabled at runtime.
pub fn is_enabled() -> bool {
    TELEMETRY_ENABLED.load(Ordering::SeqCst)
}

/// Shutdown OpenTelemetry, flushing any pending spans.
pub fn shutdown() {
    if TELEMETRY_ENABLED.load(Ordering::SeqCst) {
        opentelemetry::global::shutdown_tracer_provider();
    }
}
