use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;
use tempfile::TempDir;

/// Test utilities
struct TestContext {
    temp_dir: TempDir,
}

impl TestContext {
    fn new() -> Self {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");

        Self {
            temp_dir,
        }
    }

    fn get_cli_binary() -> String {
        std::env::var("CARGO_BIN_EXE_pie-cli")
            .unwrap_or_else(|_| "target/debug/pie-cli".to_string())
    }
}

// Phase 1 Tests: Controller Functionality

#[tokio::test]
async fn test_cli_shows_help_by_default() {
    let mut cmd = Command::new(TestContext::get_cli_binary());
    cmd.assert()
        .failure() // clap returns error code when no subcommand provided
        .stderr(predicate::str::contains("Usage:"));
}

#[tokio::test]
async fn test_controller_help_command() {
    let mut cmd = Command::new(TestContext::get_cli_binary());
    cmd.arg("controller").arg("--help");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("controller"))
        .stdout(predicate::str::contains("start"))
        .stdout(predicate::str::contains("status"))
        .stdout(predicate::str::contains("stop"));
}

#[tokio::test]
async fn test_controller_status_when_not_running() {
    let mut cmd = Command::new(TestContext::get_cli_binary());
    cmd.arg("controller").arg("status");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Engine Manager: Not running or not accessible"));
}

#[tokio::test]
async fn test_controller_start_without_engine_manager_binary() {
    // This test verifies error handling when engine-manager binary is not available
    // We need to test from a directory where relative paths won't work
    let ctx = TestContext::new();

    // Change to temp directory where relative paths to engine-manager won't work
    let temp_path = ctx.temp_dir.path().to_string_lossy();
    let cli_binary = std::env::current_dir()
        .unwrap()
        .join(TestContext::get_cli_binary());

    let mut cmd = Command::new(&cli_binary);
    cmd.arg("controller")
        .arg("start")
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg("18080") // Use a different port to avoid conflicts
        .env("PATH", temp_path.as_ref())
        .current_dir(ctx.temp_dir.path()); // Run from temp dir where relative paths fail

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Could not find pie_engine_manager binary"));
}

// Backend command tests (Phase 2/3 - stubs for now)

#[tokio::test]
async fn test_backend_help_command() {
    let mut cmd = Command::new(TestContext::get_cli_binary());
    cmd.arg("backend").arg("--help");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("backend"))
        .stdout(predicate::str::contains("list"))
        .stdout(predicate::str::contains("start"))
        .stdout(predicate::str::contains("status"))
        .stdout(predicate::str::contains("terminate"));
}

#[tokio::test]
async fn test_backend_list_without_service() {
    // Test that backend list fails gracefully when no management service is running
    let mut cmd = Command::new(TestContext::get_cli_binary());
    cmd.arg("backend").arg("list");
    cmd.assert()
        .failure() // Expected to fail when no management service is running
        .stderr(predicate::str::contains("Failed to connect to management service"));
}

#[tokio::test]
async fn test_backend_start_without_config() {
    // Test that backend start fails gracefully when configuration is missing
    let mut cmd = Command::new(TestContext::get_cli_binary());
    cmd.arg("backend")
        .arg("start")
        .arg("test-backend-type");
    cmd.assert()
        .failure() // Expected to fail because test-backend-type doesn't exist
        .stderr(predicate::str::contains("Unknown model or backend type"));
}

// Model command tests (Phase 4 - stubs for now)

#[tokio::test]
async fn test_model_help_command() {
    let mut cmd = Command::new(TestContext::get_cli_binary());
    cmd.arg("model").arg("--help");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("model"))
        .stdout(predicate::str::contains("load"))
        .stdout(predicate::str::contains("unload"))
        .stdout(predicate::str::contains("download"));
}

#[tokio::test]
async fn test_model_load_command_stub() {
    let mut cmd = Command::new(TestContext::get_cli_binary());
    cmd.arg("model")
        .arg("load")
        .arg("backend-123")
        .arg("--model-name")
        .arg("test-model");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("will be implemented in Phase"));
}

// Application command tests (Phase 5 - stubs for now)

#[tokio::test]
async fn test_application_help_command() {
    let mut cmd = Command::new(TestContext::get_cli_binary());
    cmd.arg("application").arg("--help");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("application"))
        .stdout(predicate::str::contains("deploy"))
        .stdout(predicate::str::contains("run"))
        .stdout(predicate::str::contains("serve"));
}

#[tokio::test]
async fn test_application_deploy_command_stub() {
    let mut cmd = Command::new(TestContext::get_cli_binary());
    cmd.arg("application")
        .arg("deploy")
        .arg("/path/to/test.wasm");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("will be implemented in Phase"));
}

// Integration tests combining multiple components

#[tokio::test]
async fn test_full_help_structure() {
    // Test that all main commands are available
    let mut cmd = Command::new(TestContext::get_cli_binary());
    cmd.arg("--help");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("controller"))
        .stdout(predicate::str::contains("backend"))
        .stdout(predicate::str::contains("model"))
        .stdout(predicate::str::contains("application"));
}

#[tokio::test]
async fn test_invalid_command() {
    let mut cmd = Command::new(TestContext::get_cli_binary());
    cmd.arg("invalid-command");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("error:"));
}

#[tokio::test]
async fn test_controller_start_with_custom_port() {
    let ctx = TestContext::new();
    let temp_path = ctx.temp_dir.path().to_string_lossy();
    let cli_binary = std::env::current_dir()
        .unwrap()
        .join(TestContext::get_cli_binary());

    let mut cmd = Command::new(&cli_binary);
    cmd.arg("controller")
        .arg("start")
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg("19080")
        .arg("--engine-port")
        .arg("19123")
        .env("PATH", temp_path.as_ref())
        .current_dir(ctx.temp_dir.path()); // Run from temp dir where relative paths fail

    // This should fail because engine-manager binary doesn't exist in PATH or relative paths
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Could not find pie_engine_manager binary"));
}

// Additional tests that would be useful for end-to-end testing
// (These would be implemented as the actual functionality is built)

#[tokio::test]
#[ignore] // Ignored until Phase 2/3 implementation
async fn test_controller_with_mock_engine_manager() {
    // This test would create a mock engine-manager binary and test the full flow
    // Implementation would be added in Phase 2
}

#[tokio::test]
async fn test_backend_registration_flow() {
    use std::process::{Command, Stdio};
    use tokio::time::{timeout, Duration};

    // Start engine manager service for testing
    let test_port = 18081; // Use a different port to avoid conflicts

    // Check if engine manager binary exists
    let engine_manager_check = Command::new("cargo")
        .args(&["check", "--bin", "pie_engine_manager"])
        .current_dir("../engine-manager")
        .output();

    if engine_manager_check.is_err() {
        println!("Skipping test: engine-manager binary not available");
        return;
    }

    // Start the engine manager in the background
    let mut engine_manager = Command::new("cargo")
        .args(&["run", "--bin", "pie_engine_manager", "--", "--port", &test_port.to_string()])
        .current_dir("../engine-manager")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to start engine manager");

    // Wait for service to be ready with timeout
    let service_url = format!("http://127.0.0.1:{}", test_port);
    let health_url = format!("{}/health", service_url);

    let service_ready = timeout(Duration::from_secs(10), async {
        for _ in 0..20 {
            if let Ok(response) = reqwest::get(&health_url).await {
                if response.status().is_success() {
                    return true;
                }
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        false
    }).await;

    if !service_ready.unwrap_or(false) {
        engine_manager.kill().expect("Failed to kill engine manager");
        let _ = engine_manager.wait();
        panic!("Engine manager service did not become ready in time");
    }

    // Test backend list with running service
    let mut cmd = Command::new(TestContext::get_cli_binary());
    cmd.arg("backend")
        .arg("list")
        .arg("--management-service-url")
        .arg(&service_url);

    let output = cmd.output().expect("Failed to execute command");

    // Should succeed and show no backends
    assert!(output.status.success(),
        "Backend list command failed: {}",
        String::from_utf8_lossy(&output.stderr));

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("No backends currently registered") || stdout.contains("Registered Backends"),
        "Unexpected output: {}", stdout);

    // Clean up: terminate the engine manager
    engine_manager.kill().expect("Failed to kill engine manager");
    let _ = engine_manager.wait();
}

#[tokio::test]
#[ignore] // Ignored until Phase 4 implementation
async fn test_model_management_flow() {
    // This test would verify model loading/unloading through the CLI
    // Implementation would be added in Phase 4
}

#[tokio::test]
#[ignore] // Ignored until Phase 5 implementation
async fn test_application_deployment_flow() {
    // This test would verify application deployment and execution
    // Implementation would be added in Phase 5
}
