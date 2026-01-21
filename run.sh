pie-client submit --path std/text-completion-inferlib/target/wasm32-wasip2/release/text_completion_inferlib.wasm \
    -d inferlib_environment -d inferlib_dummy -d inferlib_context \
    -- -p Hello
