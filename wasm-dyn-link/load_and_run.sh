pie-client load target/wasm32-wasip2/release/provider_logging.wasm
pie-client load target/wasm32-wasip2/release/provider_calculator.wasm -d provider_logging
pie-client load ~/work/pie/sdk/rust/inferlib/target/wasm32-wasip2/release/inferlib_environment.wasm
pie-client submit --path target/wasm32-wasip2/release/app_consumer.wasm -d provider_logging -d provider_calculator -d inferlib_environment