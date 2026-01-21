// Provider component: exports the logging interface with a logger resource

wit_bindgen::generate!({
    world: "logging-provider",
    path: "../wit",
});

use std::cell::Cell;
use std::sync::atomic::{AtomicU32, Ordering};
use exports::demo::logging::logging::{Guest, GuestLogger, Level, Logger as LoggerHandle};

/// The actual Logger resource implementation
pub struct LoggerImpl {
    max_level: Cell<Level>,
    id: u32,
}

// Global counter for logger IDs (thread-safe for static function)
static LOGGER_COUNTER: AtomicU32 = AtomicU32::new(0);

fn next_logger_id() -> u32 {
    LOGGER_COUNTER.fetch_add(1, Ordering::SeqCst) + 1
}

fn level_to_str(level: Level) -> &'static str {
    match level {
        Level::Debug => "DEBUG",
        Level::Info => "INFO",
        Level::Warn => "WARN",
        Level::Error => "ERROR",
    }
}

fn level_to_num(level: Level) -> u8 {
    match level {
        Level::Debug => 0,
        Level::Info => 1,
        Level::Warn => 2,
        Level::Error => 3,
    }
}

impl GuestLogger for LoggerImpl {
    fn new(max_level: Level) -> Self {
        let id = next_logger_id();
        println!(
            "[PROVIDER] Logger::new(id={}, max_level={}) - constructor called",
            id,
            level_to_str(max_level)
        );
        LoggerImpl {
            max_level: Cell::new(max_level),
            id,
        }
    }

    fn try_new(max_level: Level, enabled: bool) -> Result<LoggerHandle, String> {
        if !enabled {
            return Err("logger creation disabled".to_string());
        }
        Ok(LoggerHandle::new(Self::new(max_level)))
    }

    fn maybe_new(max_level: Level, enabled: bool) -> Option<LoggerHandle> {
        if !enabled {
            return None;
        }
        Some(LoggerHandle::new(Self::new(max_level)))
    }

    /// Static function: get the total count of loggers created
    fn get_logger_count() -> u32 {
        let count = LOGGER_COUNTER.load(Ordering::SeqCst);
        println!("[PROVIDER] Logger::get_logger_count() -> {}", count);
        count
    }

    fn get_max_level(&self) -> Level {
        let level = self.max_level.get();
        println!(
            "[PROVIDER] Logger::get_max_level(id={}) -> {}",
            self.id,
            level_to_str(level)
        );
        level
    }

    fn set_max_level(&self, level: Level) {
        println!(
            "[PROVIDER] Logger::set_max_level(id={}, level={})",
            self.id,
            level_to_str(level)
        );
        self.max_level.set(level);
    }

    fn log(&self, level: Level, msg: String) {
        let max = self.max_level.get();
        let should_log = level_to_num(level) >= level_to_num(max);
        if should_log {
            println!(
                "[PROVIDER] Logger::log(id={}, level={}) -> LOGGED: {}",
                self.id,
                level_to_str(level),
                msg
            );
        } else {
            println!(
                "[PROVIDER] Logger::log(id={}, level={}) -> FILTERED (below max_level={}): {}",
                self.id,
                level_to_str(level),
                level_to_str(max),
                msg
            );
        }
    }
}

impl Drop for LoggerImpl {
    fn drop(&mut self) {
        println!(
            "[PROVIDER] Logger::drop(id={}) - destructor called!",
            self.id
        );
    }
}

struct MyProvider;

impl Guest for MyProvider {
    type Logger = LoggerImpl;

    /// Standalone function: get the default log level
    fn get_default_level() -> Level {
        println!("[PROVIDER] get_default_level() -> INFO");
        Level::Info
    }

    /// Standalone function: convert a level to its string representation
    fn level_to_string(lvl: Level) -> String {
        let s = level_to_str(lvl);
        println!("[PROVIDER] level_to_string({}) -> \"{}\"", level_to_str(lvl), s);
        s.to_string()
    }

    /// Factory function: create a logger
    fn create_logger(max_level: Level) -> LoggerHandle {
        println!(
            "[PROVIDER] create_logger(max_level={})",
            level_to_str(max_level)
        );
        LoggerHandle::new(LoggerImpl::new(max_level))
    }

    /// Factory function: create a logger optionally
    fn maybe_create_logger(max_level: Level, enabled: bool) -> Option<LoggerHandle> {
        println!(
            "[PROVIDER] maybe_create_logger(max_level={}, enabled={})",
            level_to_str(max_level),
            enabled
        );
        if enabled {
            Some(LoggerHandle::new(LoggerImpl::new(max_level)))
        } else {
            None
        }
    }

    fn echo_logger(logger: LoggerHandle) -> LoggerHandle {
        println!("[PROVIDER] echo_logger(logger)");
        logger
    }
}

export!(MyProvider);
