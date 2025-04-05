use boa_engine::{Context, JsResult, Source};
use symphony::wstd::time::Duration;

#[symphony::main]
async fn main() -> Result<(), String> {
    let max_num_outputs = 32;

    let available_models = symphony::available_models();

    // Simulate agentic behavior

    let model = symphony::Model::new(available_models.first().unwrap()).unwrap();
    let tokenizer = model.get_tokenizer();

    let mut ctx = model.create_context();

    ctx.fill("<|begin_of_text|>");
    ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");
    ctx.fill("<|start_header_id|>user<|end_header_id|>\n\nExplain the LLM decoding process ELI5.<|eot_id|>");
    ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");

    let text = ctx.generate_until("<|eot_id|>", max_num_outputs).await;

    let js_code = r#"
          let two = 1 + 1;
          let definitely_not_four = two + "2";
    
          definitely_not_four
      "#;

    // Instantiate the execution context
    let mut context = Context::default();

    // Parse the source code
    let result = context
        .eval(Source::from_bytes(js_code))
        .map_err(|e| e.to_string())?;

    let res = format!("result from the function call: {}", result.display());

    ctx.fill(&res);

    let text = ctx.generate_until("<|eot_id|>", max_num_outputs).await;

    Ok(())
}
