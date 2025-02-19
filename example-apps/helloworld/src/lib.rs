use crate::spi::app::system;
use anyhow::Result;

#[symphony::main]
async fn main() -> Result<()> {
    let inst_id = system::get_instance_id();

    println!(
        "[{}] I am a WASM module running in the Symphony runtime!",
        { inst_id }
    );

    system::send_to_origin("What is your name?");
    system::send_to_origin("Have a great day!");
    Ok(())
}


// 
// wit_bindgen::generate!({
//     path: "../../api/app/wit",
//     world: "app",
//     generate_all,
// });
// 
// use crate::spi::app::system;
// //use tokio::runtime::Builder;
// 
// struct App;
// 
// impl exports::spi::app::run::Guest for App {
//     fn run() -> core::result::Result<(), String> {
//         let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
// 
//         // Run the async main function inside the runtime
//         let result = runtime.block_on(async_main());
//         
//         if let Err(e) = result {
//             return Err(format!("{:?}", e));
//         }
//         
//         Ok(())
//     }
// }
// 
// async fn async_main() -> anyhow::Result<()> {
//     let inst_id = system::get_instance_id();
// 
//     println!(
//         "[{}] I am a WASM module running in the Symphony runtime!",
//         { inst_id }
//     );
// 
//     system::send_to_origin("What is your name?");
//     system::send_to_origin("Have a great day!");
// }
// 
// export!(App);
