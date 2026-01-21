// App consumer component: imports the calculator and logging interfaces and exports run

wit_bindgen::generate!({
    world: "app",
    path: "../wit",
});

use inferlib_run_bindings::{Args, Result, anyhow};

use demo::logging::calculator::{self, Calc};
use demo::logging::logging::{self, Logger, Level};

struct MyApp;

impl Guest for MyApp {
    fn run() {
        println!("[APP] run() starting...");

        // ===== Test standalone functions from logging interface =====
        println!("\n[APP] === Testing logging standalone functions ===");

        // Test get_default_level standalone function
        let default_level = logging::get_default_level();
        println!("[APP] Default log level: {:?}", default_level);

        // Test level_to_string standalone function
        let level_str = logging::level_to_string(Level::Error);
        println!("[APP] Level::Error as string: \"{}\"", level_str);

        // ===== Test static function from logging interface =====
        println!("\n[APP] === Testing logging static functions ===");
        
        // Test get_logger_count static function (before creating any loggers)
        let count_before = Logger::get_logger_count();
        println!("[APP] Logger count before creating loggers: {}", count_before);

        // Create a logger to increment the count
        {
            let _logger = Logger::new(Level::Debug);
            let count_after = Logger::get_logger_count();
            println!("[APP] Logger count after creating one logger: {}", count_after);
        }

        // ===== Test factory and fallible logger creation =====
        println!("\n[APP] === Testing logging factories ===");

        let factory_logger = logging::create_logger(Level::Info);
        factory_logger.log(Level::Info, "[APP] log from factory logger");

        let maybe_logger = logging::maybe_create_logger(Level::Warn, true);
        println!(
            "[APP] maybe_create_logger(enabled=true) -> {}",
            if maybe_logger.is_some() { "Some(logger)" } else { "None" }
        );

        let maybe_none = logging::maybe_create_logger(Level::Warn, false);
        println!(
            "[APP] maybe_create_logger(enabled=false) -> {}",
            if maybe_none.is_some() { "Some(logger)" } else { "None" }
        );

        let try_logger = Logger::try_new(Level::Debug, true).expect("try_new enabled");
        try_logger.log(Level::Debug, "[APP] log from try_new logger");

        let try_err = Logger::try_new(Level::Debug, false);
        println!(
            "[APP] Logger::try_new(enabled=false) -> {}",
            if try_err.is_err() { "Err" } else { "Ok" }
        );

        println!("\n[APP] === Testing logger identity echo ===");
        let echo_logger = logging::create_logger(Level::Info);
        let echoed = logging::echo_logger(echo_logger);
        // Use the echoed logger to ensure it remains valid.
        echoed.log(Level::Info, "[APP] log from echoed logger");

        // ===== Test standalone functions from calculator interface =====
        println!("\n[APP] === Testing calculator standalone functions ===");

        // Test pi standalone function
        let pi = calculator::pi();
        println!("[APP] Value of pi: {}", pi);

        // Test quick_add standalone function
        let quick_result = calculator::quick_add(100.0, 23.0);
        println!("[APP] quick_add(100, 23) = {}", quick_result);

        // Test passing a logging resource into calculator (cross-provider resource)
        println!("\n[APP] === Testing cross-provider resource passing ===");
        let shared_logger = logging::create_logger(Level::Info);
        calculator::log_with_logger(shared_logger, "hello from app");

        // ===== Test static functions from calculator interface =====
        println!("\n[APP] === Testing calculator static functions ===");

        // Test version static function
        let version = Calc::version();
        println!("[APP] Calculator version: \"{}\"", version);

        // Test total_operations static function (before any operations)
        let ops_before = Calc::total_operations();
        println!("[APP] Total operations before calculations: {}", ops_before);

        // ===== Test resource methods (instance functions) =====
        println!("\n[APP] === Testing calculator resource methods ===");

        // Create a calculator
        println!("[APP] Creating calculator");
        let calc = Calc::new();

        // Perform some calculations
        println!("[APP] Performing calculations...");

        let a = 10.0;
        let b = 3.0;

        let sum = calc.add(a, b);
        println!("[APP] {} + {} = {}", a, b, sum);

        let diff = calc.subtract(a, b);
        println!("[APP] {} - {} = {}", a, b, diff);

        let product = calc.multiply(a, b);
        println!("[APP] {} * {} = {}", a, b, product);

        let quotient = calc.divide(a, b);
        println!("[APP] {} / {} = {}", a, b, quotient);

        // Test division by zero
        println!("[APP] Testing division by zero...");
        let zero_div = calc.divide(5.0, 0.0);
        println!("[APP] 5.0 / 0.0 = {} (should be 0)", zero_div);

        // Check total operations after calculations
        let ops_after = Calc::total_operations();
        println!("[APP] Total operations after calculations: {}", ops_after);

        println!("\n[APP] About to drop calculator...");
        // Calculator will be dropped here when it goes out of scope
        // This should trigger calc's destructor, which logs and then drops the logger
        drop(calc);

        // Final logger count
        let final_count = Logger::get_logger_count();
        println!("[APP] Final logger count: {}", final_count);

        println!("[APP] run() completed!");
    }
}

export!(MyApp);

#[inferlib_macros::main]
async fn main(_: Args) -> Result<()> {
    MyApp::run();
    Ok(())
}
