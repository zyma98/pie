use inferlet::{Args, Result};
use std::time::Duration;

#[inferlet::main]
async fn main(mut args: Args) -> Result<()> {
    let text: Option<String> = args.opt_value_from_str(["-t", "--text"])?;
    let delay: Option<u64> = args.opt_value_from_str(["-d", "--delay"])?;
    let text_before_delay: Option<String> =
        args.opt_value_from_str(["-b", "--text-before-delay"])?;

    if let Some(text_before_delay) = text_before_delay {
        println!("{}", text_before_delay);
    }
    if let Some(delay) = delay {
        std::thread::sleep(Duration::from_millis(delay));
    }
    if let Some(text) = text {
        println!("{}", text);
    }

    Ok(())
}
