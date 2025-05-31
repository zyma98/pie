//! ZMQ client logic for CLI communication with the management service.

use crate::types::{ManagementCommand, ManagementResponse};
use crate::config::Config;
use std::collections::HashMap;
use super::cli::Commands;
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, ZmqMessage};
use tokio::time::{timeout, Duration};

pub struct ZmqClient {
    socket: DealerSocket,
    endpoint: String,
}

impl ZmqClient {
    pub async fn new(service_endpoint: &str) -> Result<Self, String> {
        let mut socket = DealerSocket::new();
        
        // Connect to the service with timeout
        match timeout(Duration::from_secs(3), socket.connect(service_endpoint)).await {
            Ok(Ok(())) => {},
            Ok(Err(e)) => {
                return Err(format!("Failed to connect to {}: {}", service_endpoint, e));
            },
            Err(_) => {
                return Err(format!("Timeout: Could not connect to service at {} (is the service running?)", service_endpoint));
            }
        }
        
        Ok(Self { 
            socket, 
            endpoint: service_endpoint.to_string() 
        })
    }

    pub async fn send_command(&mut self, command: ManagementCommand) -> Result<ManagementResponse, String> {
        let serialized_cmd = serde_json::to_string(&command)
            .map_err(|e| format!("Failed to serialize command: {}", e))?;

        let msg = ZmqMessage::from(serialized_cmd.into_bytes());

        // Try to send the command with timeout
        match timeout(Duration::from_secs(3), self.socket.send(msg)).await {
            Ok(Ok(())) => {},
            Ok(Err(e)) => {
                return Err(format!("Failed to send command to {}: {}", self.endpoint, e));
            },
            Err(_) => {
                return Err(format!("Timeout: Could not send command to service at {} (is the service running?)", self.endpoint));
            }
        }

        // Determine timeout based on command type - model loading needs much longer
        let receive_timeout = if command.command == "load-model" {
            Duration::from_secs(60) // 60 seconds for model loading
        } else {
            Duration::from_secs(10) // 10 seconds for other operations
        };

        // Try to receive the response with appropriate timeout
        match timeout(receive_timeout, self.socket.recv()).await {
            Ok(Ok(response_msg)) => {
                let response_bytes = response_msg.get(0)
                    .ok_or_else(|| "Received empty response".to_string())?;
                let response_str = std::str::from_utf8(response_bytes)
                    .map_err(|_| "Received non-UTF8 response".to_string())?;
                serde_json::from_str(response_str)
                    .map_err(|e| format!("Failed to deserialize response: {}", e))
            },
            Ok(Err(e)) => {
                Err(format!("Failed to receive response from {}: {}", self.endpoint, e))
            },
            Err(_) => {
                let timeout_msg = if command.command == "load-model" {
                    "Timeout: Model loading took longer than 60 seconds"
                } else {
                    "Timeout: No response from service within 10 seconds"
                };
                Err(format!("{} (endpoint: {})", timeout_msg, self.endpoint))
            }
        }
    }
}

fn get_service_endpoint() -> String {
    // Try to load from config first
    if let Ok(config) = Config::load_default() {
        return config.endpoints.cli_management.clone();
    }
    
    // Fall back to default IPC endpoint (not TCP)
    "ipc:///tmp/symphony-cli".to_string()
}

pub async fn send_command_to_service(command_enum: Commands, json: bool) -> Result<String, String> {
    // 1. Determine the service endpoint
    let endpoint = get_service_endpoint();

    // 2. Create a ZmqClient instance
    let mut client = ZmqClient::new(&endpoint).await?;

    // 3. Convert cli::Commands enum to a types::ManagementCommand
    let core_command = match &command_enum {
        Commands::Status => ManagementCommand::new("status".to_string(), HashMap::new()),
        Commands::LoadModel { model_name, config_path } => {
            let mut params = HashMap::new();
            params.insert("model_name".to_string(), serde_json::Value::String(model_name.clone()));
            if let Some(path) = config_path {
                params.insert("config_path".to_string(), serde_json::Value::String(path.clone()));
            }
            ManagementCommand::new("load-model".to_string(), params)
        }
        Commands::UnloadModel { model_name } => {
            let mut params = HashMap::new();
            params.insert("model_name".to_string(), serde_json::Value::String(model_name.clone()));
            ManagementCommand::new("unload-model".to_string(), params)
        }
        Commands::ListModels => ManagementCommand::new("list-models".to_string(), HashMap::new()),
        Commands::StopService => ManagementCommand::new("stop-service".to_string(), HashMap::new()),
        Commands::StartService { .. } => {
            return Err("StartService command should be handled locally, not sent to service".to_string());
        }
    };

    // 4. Send the command and get a response
    match client.send_command(core_command).await {
        Ok(response) => {
            if json {
                // Return raw JSON response
                if response.success {
                    if let Some(data) = response.data {
                        Ok(serde_json::to_string_pretty(&data)
                            .unwrap_or_else(|_| r#"{"status": "success"}"#.to_string()))
                    } else {
                        Ok(r#"{"status": "success"}"#.to_string())
                    }
                } else {
                    let error_msg = response.error.unwrap_or_else(|| "Unknown error".to_string());
                    Err(error_msg)
                }
            } else {
                // Format the ManagementResponse into a user-friendly string
                if response.success {
                    if let Some(data) = response.data {
                        format_response_pretty(&command_enum, &data)
                    } else {
                        Ok("✓ Success".to_string())
                    }
                } else {
                    let error_msg = response.error.unwrap_or_else(|| "Unknown error".to_string());
                    Err(format!("Service error: {}", error_msg))
                }
            }
        }
        Err(e) => Err(e),
    }
}

/// Format response data in a user-friendly way based on the command type
fn format_response_pretty(command: &Commands, data: &serde_json::Value) -> Result<String, String> {
    match command {
        Commands::Status => {
            let mut result = String::from("✓ Service is running");
            
            // Add model information if available
            if let Some(models) = data.get("models").and_then(|m| m.as_array()) {
                let model_count = models.len();
                result.push_str(&format!("\n  Models loaded: {}", model_count));
                
                if model_count > 0 && model_count <= 3 {
                    // Show details for up to 3 models
                    for model in models {
                        if let Some(model_name) = model.get("model_name").and_then(|n| n.as_str()) {
                            result.push_str(&format!("\n  • {}", model_name));
                            if let Some(uptime) = model.get("uptime").and_then(|u| u.as_u64()) {
                                result.push_str(&format!(" (uptime: {}s)", uptime));
                            }
                        }
                    }
                } else if model_count > 3 {
                    result.push_str("\n  (use 'list-models' to see details)");
                }
            }
            
            Ok(result)
        }
        Commands::ListModels => {
            if let Some(models) = data.get("models").and_then(|m| m.as_array()) {
                if models.is_empty() {
                    Ok("No models currently loaded".to_string())
                } else {
                    let mut result = String::from("Loaded Models:\n");
                    for model in models {
                        if let Some(model_obj) = model.as_object() {
                            let model_name = model_obj.get("model_name")
                                .and_then(|n| n.as_str())
                                .unwrap_or("unknown");
                            let model_type = model_obj.get("model_type")
                                .and_then(|t| t.as_str())
                                .unwrap_or("unknown");
                            let uptime = model_obj.get("uptime")
                                .and_then(|u| u.as_u64())
                                .unwrap_or(0);
                            let is_alive = model_obj.get("is_alive")
                                .and_then(|a| a.as_bool())
                                .unwrap_or(false);
                            let status_icon = if is_alive { "✓" } else { "✗" };
                            
                            result.push_str(&format!("  {} {} ({})\n", status_icon, model_name, model_type));
                            result.push_str(&format!("    Uptime: {}s", uptime));
                            
                            if let Some(endpoint) = model_obj.get("endpoint").and_then(|e| e.as_str()) {
                                result.push_str(&format!(" | Endpoint: {}", endpoint));
                            }
                            result.push('\n');
                        } else if let Some(name) = model.as_str() {
                            // Fallback for simple string models
                            result.push_str(&format!("  • {}\n", name));
                        }
                    }
                    Ok(result.trim_end().to_string())
                }
            } else {
                Ok("No models currently loaded".to_string())
            }
        }
        Commands::LoadModel { model_name, .. } => {
            if let Some(message) = data.get("message") {
                Ok(format!("✓ {}", message.as_str().unwrap_or("Model loaded successfully")))
            } else {
                Ok(format!("✓ Model '{}' loaded successfully", model_name))
            }
        }
        Commands::UnloadModel { model_name } => {
            if let Some(message) = data.get("message") {
                Ok(format!("✓ {}", message.as_str().unwrap_or("Model unloaded successfully")))
            } else {
                Ok(format!("✓ Model '{}' unloaded successfully", model_name))
            }
        }
        Commands::StopService => {
            if let Some(message) = data.get("message") {
                Ok(format!("✓ {}", message.as_str().unwrap_or("Service shutdown requested")))
            } else {
                Ok("✓ Service shutdown requested".to_string())
            }
        }
        Commands::StartService { .. } => {
            Ok("✓ Service started".to_string())
        }
    }
}
