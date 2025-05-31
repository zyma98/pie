//! ZMQ client logic for CLI communication with the management service.

use crate::types::{ManagementCommand, ManagementResponse};
use crate::config::Config;
use std::collections::HashMap;
use super::cli::Commands;

pub struct ZmqClient {
    socket: zmq::Socket,
    endpoint: String,
}

impl ZmqClient {
    pub fn new(service_endpoint: &str) -> Result<Self, String> {
        let context = zmq::Context::new();
        let socket = context.socket(zmq::DEALER).map_err(|e| e.to_string())?;
        socket.connect(service_endpoint).map_err(|e| e.to_string())?;
        
        // Set timeouts for send/recv (5 seconds)
        socket.set_rcvtimeo(5000).map_err(|e| e.to_string())?;
        socket.set_sndtimeo(5000).map_err(|e| e.to_string())?;
        
        Ok(Self { 
            socket, 
            endpoint: service_endpoint.to_string() 
        })
    }

    pub fn send_command(&self, command: ManagementCommand) -> Result<ManagementResponse, String> {
        let serialized_cmd = serde_json::to_string(&command)
            .map_err(|e| format!("Failed to serialize command: {}", e))?;

        self.socket.send(&serialized_cmd, 0)
            .map_err(|e| format!("Failed to send command: {}", e))?;

        let mut msg = zmq::Message::new();
        match self.socket.recv(&mut msg, 0) {
            Ok(_) => {
                let response_str = msg.as_str().ok_or_else(|| "Received non-UTF8 response".to_string())?;
                serde_json::from_str(response_str)
                    .map_err(|e| format!("Failed to deserialize response: {}", e))
            },
            Err(zmq::Error::EAGAIN) => {
                Err(format!("Timeout: No response from service at {} (is the service running?)", self.endpoint))
            },
            Err(e) => {
                Err(format!("Failed to receive response from {}: {}", self.endpoint, e))
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

pub fn send_command_to_service(command_enum: Commands) -> Result<String, String> {
    // 1. Determine the service endpoint
    let endpoint = get_service_endpoint();

    // 2. Create a ZmqClient instance
    let client = ZmqClient::new(&endpoint)?;

    // 3. Convert cli::Commands enum to a types::ManagementCommand
    let core_command = match command_enum {
        Commands::Status => ManagementCommand::new("status".to_string(), HashMap::new()),
        Commands::LoadModel { model_name, config_path } => {
            let mut params = HashMap::new();
            params.insert("model_name".to_string(), serde_json::Value::String(model_name));
            if let Some(path) = config_path {
                params.insert("config_path".to_string(), serde_json::Value::String(path));
            }
            ManagementCommand::new("load-model".to_string(), params)
        }
        Commands::UnloadModel { model_name } => {
            let mut params = HashMap::new();
            params.insert("model_name".to_string(), serde_json::Value::String(model_name));
            ManagementCommand::new("unload-model".to_string(), params)
        }
        Commands::ListModels => ManagementCommand::new("list-models".to_string(), HashMap::new()),
        Commands::StopService => ManagementCommand::new("shutdown".to_string(), HashMap::new()),
        Commands::StartService { .. } => {
            return Err("StartService command should be handled locally, not sent to service".to_string());
        }
    };

    // 4. Send the command and get a response
    match client.send_command(core_command) {
        Ok(response) => {
            // Format the ManagementResponse into a user-friendly string
            if response.success {
                if let Some(data) = response.data {
                    Ok(serde_json::to_string_pretty(&data)
                        .unwrap_or_else(|_| "Success".to_string()))
                } else {
                    Ok("Success".to_string())
                }
            } else {
                let error_msg = response.error.unwrap_or_else(|| "Unknown error".to_string());
                Err(format!("Service error: {}", error_msg))
            }
        }
        Err(e) => Err(e),
    }
}
