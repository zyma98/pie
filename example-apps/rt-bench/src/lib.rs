#[inferlet::main]
async fn main() -> Result<(), String> {
    let msg = inferlet::receive().await;

    inferlet::send(&msg);

    Ok(())
}
