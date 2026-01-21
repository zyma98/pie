// Calculator provider component: imports logging, exports calculator interface

wit_bindgen::generate!({
    world: "calculator-provider",
    path: "../wit",
});

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use demo::logging::logging::{level_to_string, get_default_level, Level, Logger};
use exports::demo::logging::calculator::{Guest, GuestCalc};

/// The actual Calc resource implementation
pub struct Calc {
    logger: Logger,
    id: u32,
}

// Global counter for calc IDs (thread-safe for static function)
static CALC_COUNTER: AtomicU32 = AtomicU32::new(0);

// Global counter for total operations performed
static TOTAL_OPERATIONS: AtomicU64 = AtomicU64::new(0);

fn next_calc_id() -> u32 {
    CALC_COUNTER.fetch_add(1, Ordering::SeqCst) + 1
}

fn increment_operations() {
    TOTAL_OPERATIONS.fetch_add(1, Ordering::SeqCst);
}

impl GuestCalc for Calc {
    fn new() -> Self {
        let id = next_calc_id();
        // Use the standalone function to get the default level
        let default_level = get_default_level();
        // Use the standalone function to convert level to string for logging
        let level_str = level_to_string(default_level);
        
        // Create a logger for this calculator with debug level to see all operations
        let logger = Logger::new(Level::Debug);
        logger.log(
            Level::Info,
            &format!(
                "[CALCULATOR] Calc::new(id={}) - constructor called (default level was: {})",
                id, level_str
            ),
        );
        Calc { logger, id }
    }

    /// Static function: get the calculator version
    fn version() -> String {
        let version = "1.0.0".to_string();
        println!("[CALCULATOR] Calc::version() -> \"{}\"", version);
        version
    }

    /// Static function: get total number of operations performed
    fn total_operations() -> u64 {
        let count = TOTAL_OPERATIONS.load(Ordering::SeqCst);
        println!("[CALCULATOR] Calc::total_operations() -> {}", count);
        count
    }

    fn add(&self, a: f64, b: f64) -> f64 {
        increment_operations();
        let result = a + b;
        self.logger.log(
            Level::Debug,
            &format!(
                "[CALCULATOR] Calc::add(id={}) {} + {} = {}",
                self.id, a, b, result
            ),
        );
        result
    }

    fn subtract(&self, a: f64, b: f64) -> f64 {
        increment_operations();
        let result = a - b;
        self.logger.log(
            Level::Debug,
            &format!(
                "[CALCULATOR] Calc::subtract(id={}) {} - {} = {}",
                self.id, a, b, result
            ),
        );
        result
    }

    fn multiply(&self, a: f64, b: f64) -> f64 {
        increment_operations();
        let result = a * b;
        self.logger.log(
            Level::Debug,
            &format!(
                "[CALCULATOR] Calc::multiply(id={}) {} * {} = {}",
                self.id, a, b, result
            ),
        );
        result
    }

    fn divide(&self, a: f64, b: f64) -> f64 {
        increment_operations();
        if b == 0.0 {
            self.logger.log(
                Level::Warn,
                &format!(
                    "[CALCULATOR] Calc::divide(id={}) {} / {} - division by zero!",
                    self.id, a, b
                ),
            );
            return 0.0;
        }
        let result = a / b;
        self.logger.log(
            Level::Debug,
            &format!(
                "[CALCULATOR] Calc::divide(id={}) {} / {} = {}",
                self.id, a, b, result
            ),
        );
        result
    }
}

impl Drop for Calc {
    fn drop(&mut self) {
        self.logger.log(
            Level::Info,
            &format!(
                "[CALCULATOR] Calc::drop(id={}) - destructor called!",
                self.id
            ),
        );
        // Note: the logger will also be dropped here, triggering logging's destructor
    }
}

struct MyCalculatorProvider;

impl Guest for MyCalculatorProvider {
    type Calc = Calc;

    /// Standalone function: get the value of pi
    fn pi() -> f64 {
        let pi = std::f64::consts::PI;
        println!("[CALCULATOR] pi() -> {}", pi);
        pi
    }

    /// Standalone function: quick add without creating a calculator instance
    fn quick_add(a: f64, b: f64) -> f64 {
        let result = a + b;
        println!("[CALCULATOR] quick_add({}, {}) -> {}", a, b, result);
        result
    }

    fn log_with_logger(logger: Logger, msg: String) {
        logger.log(
            Level::Info,
            &format!("[CALCULATOR] log_with_logger -> {}", msg),
        );
    }
}

export!(MyCalculatorProvider);
