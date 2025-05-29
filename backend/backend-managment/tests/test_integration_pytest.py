#!/usr/bin/env python3
"""
Integration tests for Symphony Management Service and CLI using pytest.
"""

import os
import sys
import json
import time
import tempfile
import threading
import pytest
from unittest.mock import Mock, patch
import zmq

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from management_service import ManagementService
from management_cli import ManagementCLI


@pytest.fixture
def temp_config():
    """Create a temporary config file for integration testing."""
    temp_dir = tempfile.mkdtemp()
    config_file = os.path.join(temp_dir, "integration_test_config.json")
    
    test_config = {
        "model_backends": {
            "llama": "test_backend.py",
            "test": "test_backend.py"
        },
        "endpoints": {
            "client_handshake": "ipc:///tmp/test-integration-client",
            "cli_management": "ipc:///tmp/test-integration-cli"
        },
        "logging": {
            "level": "ERROR",  # Clean output during tests
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        }
    }
    
    with open(config_file, 'w') as f:
        json.dump(test_config, f)
        
    yield config_file
    
    # Cleanup
    if os.path.exists(config_file):
        os.unlink(config_file)
    os.rmdir(temp_dir)


@pytest.fixture
def mock_backend_script(temp_config):
    """Create a mock backend script for testing."""
    # Load config to get backend path
    with open(temp_config) as f:
        config = json.load(f)
    
    service = ManagementService(config_path=temp_config)
    backend_dir = service.backend_base_path
    os.makedirs(backend_dir, exist_ok=True)
    
    backend_script = os.path.join(backend_dir, "test_backend.py")
    
    # Create a simple mock backend that just sleeps
    backend_content = '''#!/usr/bin/env python3
import sys
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipc-endpoint", required=True)
    parser.add_argument("--config")
    args = parser.parse_args()
    
    print(f"Mock backend started on {args.ipc_endpoint}")
    
    # Just sleep to simulate a running backend
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Mock backend shutting down")

if __name__ == "__main__":
    main()
'''
    
    with open(backend_script, 'w') as f:
        f.write(backend_content)
    
    # Make it executable
    os.chmod(backend_script, 0o755)
    
    yield backend_script
    
    # Cleanup
    if os.path.exists(backend_script):
        os.unlink(backend_script)
    if os.path.exists(backend_dir):
        os.rmdir(backend_dir)


class TestIntegration:
    """Integration tests for Management Service and CLI."""
    
    def test_service_cli_status_integration(self, temp_config):
        """Test service and CLI status integration."""
        service = ManagementService(config_path=temp_config)
        cli = ManagementCLI(service_endpoint="ipc:///tmp/test-integration-cli")
        
        # Initialize service sockets without starting main loop
        assert service.initialize_sockets() is True
        
        # Create a background thread to handle messages
        def message_handler():
            while not service.shutdown_requested:
                result = service.handle_single_message(timeout=1000)  # Increase timeout
                if result:
                    print(f"Handled message: {result}")
        
        handler_thread = threading.Thread(target=message_handler, daemon=True)
        handler_thread.start()
        
        # Give service more time to initialize
        time.sleep(0.5)
        
        try:
            # Test CLI status command with debugging
            print("Testing if service is running...")
            is_running = cli._is_service_running()
            print(f"Service running status: {is_running}")
            assert is_running is True
            
            result = cli.status()
            assert result is True
            
        finally:
            service.shutdown_requested = True
            time.sleep(0.2)  # Give time for cleanup
            # Stop service
            service.shutdown_requested = True
            time.sleep(0.2)
    
    @patch('management_service.subprocess.Popen')
    def test_service_cli_load_model_integration(self, mock_popen, temp_config, mock_backend_script):
        """Test service and CLI load model integration."""
        # Mock the subprocess for backend creation
        mock_process = Mock()
        mock_process.poll.return_value = None  # Process is running
        
        # Mock stdout to return empty strings (simulating no output)
        mock_stdout = Mock()
        mock_stdout.readline.side_effect = ['', '', '']  # Empty strings to end iteration
        mock_process.stdout = mock_stdout
        
        mock_popen.return_value = mock_process
        
        service = ManagementService(config_path=temp_config)
        cli = ManagementCLI(service_endpoint="ipc:///tmp/test-integration-cli")
        
        # Initialize service sockets without starting main loop
        assert service.initialize_sockets() is True
        
        # Create a background thread to handle messages
        def message_handler():
            while not service.shutdown_requested:
                service.handle_single_message(timeout=100)
        
        handler_thread = threading.Thread(target=message_handler, daemon=True)
        handler_thread.start()
        
        # Give service time to initialize
        time.sleep(0.1)
        
        try:
            # Test CLI load model command
            result = cli.load_model("test-model")
            assert result is True
            
            # Verify model was loaded in service
            assert "test-model" in service.model_instances
            
            # Test CLI status shows loaded model
            result = cli.status()
            assert result is True
            
            # Test CLI unload model command
            result = cli.unload_model("test-model")
            assert result is True
            
            # Verify model was unloaded
            assert "test-model" not in service.model_instances
            
        finally:
            # Stop service
            service.shutdown_requested = True
            time.sleep(0.1)
    
    def test_service_cli_stop_integration(self, temp_config):
        """Test service and CLI stop integration."""
        service = ManagementService(config_path=temp_config)
        cli = ManagementCLI(service_endpoint="ipc:///tmp/test-integration-cli")
        
        # Initialize service sockets without starting main loop
        assert service.initialize_sockets() is True
        
        # Create a background thread to handle messages
        def message_handler():
            while not service.shutdown_requested:
                service.handle_single_message(timeout=100)
        
        handler_thread = threading.Thread(target=message_handler, daemon=True)
        handler_thread.start()
        
        # Give service time to initialize
        time.sleep(0.1)
        
        # Verify service is running
        assert cli._is_service_running() is True
        
        # Stop service via CLI
        result = cli.stop_service()
        assert result is True
        
        # Give time for shutdown
        time.sleep(0.5)
        
        # Verify service is stopped
        assert service.shutdown_requested is True
    
    def test_multiple_cli_commands(self, temp_config):
        """Test multiple CLI commands in sequence."""
        service = ManagementService(config_path=temp_config)
        cli = ManagementCLI(service_endpoint="ipc:///tmp/test-integration-cli")
        
        # Initialize service sockets without starting main loop
        assert service.initialize_sockets() is True
        
        # Create a background thread to handle messages
        def message_handler():
            while not service.shutdown_requested:
                service.handle_single_message(timeout=100)
        
        handler_thread = threading.Thread(target=message_handler, daemon=True)
        handler_thread.start()
        
        # Give service time to initialize
        time.sleep(0.1)
        
        try:
            # Test multiple status calls
            for _ in range(3):
                result = cli.status()
                assert result is True
                time.sleep(0.1)
            
            # Test unknown command handling (should not crash service)
            response = cli._send_command("unknown-command")
            assert response["success"] is False
            assert "Unknown command" in response.get("error", "")
            
        finally:
            # Stop service
            service.shutdown_requested = True
            time.sleep(0.1)
    
    def test_service_resilience(self, temp_config):
        """Test service resilience to malformed CLI requests."""
        service = ManagementService(config_path=temp_config)
        
        # Initialize service sockets without starting main loop
        assert service.initialize_sockets() is True
        
        # Create a background thread to handle messages
        def message_handler():
            while not service.shutdown_requested:
                service.handle_single_message(timeout=100)
        
        handler_thread = threading.Thread(target=message_handler, daemon=True)
        handler_thread.start()
        
        # Give service time to initialize
        time.sleep(0.1)
        
        try:
            # Send malformed JSON to CLI endpoint
            context = zmq.Context()
            req_socket = context.socket(zmq.REQ)
            req_socket.setsockopt(zmq.RCVTIMEO, 500)
            
            try:
                req_socket.connect("ipc:///tmp/test-integration-cli")
                
                # Send invalid JSON
                req_socket.send(b"invalid json")
                
                # Service should handle gracefully and not crash
                time.sleep(0.1)
                
                # Verify service is still running by sending valid request
                cli = ManagementCLI(service_endpoint="ipc:///tmp/test-integration-cli")
                result = cli.status()
                assert result is True
                
            except zmq.Again:
                # Timeout is expected for malformed request
                pass
            finally:
                req_socket.close()
                context.term()
                
        finally:
            # Stop service
            service.shutdown_requested = True
            time.sleep(0.1)
