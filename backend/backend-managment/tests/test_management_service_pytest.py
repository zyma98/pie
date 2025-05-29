#!/usr/bin/env python3
"""
Unit tests for the Symphony Management Service using pytest.
"""

import os
import sys
import json
import time
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add parent directory to path to import management_service
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import management_service
from management_service import ManagementService, ModelInstance, ManagementCommand


@pytest.fixture
def mock_process():
    """Create a mock process fixture."""
    mock_proc = Mock()
    mock_proc.poll.return_value = None  # Process is alive
    return mock_proc


@pytest.fixture
def temp_config():
    """Create a temporary config file for testing."""
    temp_dir = tempfile.mkdtemp()
    config_file = os.path.join(temp_dir, "test_config.json")
    
    test_config = {
        "model_backends": {
            "llama": "l4m_backend.py",
            "test": "test_backend.py"
        },
        "endpoints": {
            "client_handshake": "ipc:///tmp/test-symphony-ipc",
            "cli_management": "ipc:///tmp/test-symphony-cli"
        },
        "logging": {
            "level": "DEBUG",
            "format": "%(asctime)s [%(levelname)8s] %(name)s: %(message)s"
        }
    }
    
    with open(config_file, 'w') as f:
        json.dump(test_config, f)
        
    yield config_file
    
    # Cleanup
    if os.path.exists(config_file):
        os.unlink(config_file)
    os.rmdir(temp_dir)


class TestModelInstance:
    """Test the ModelInstance dataclass."""
    
    def test_model_instance_creation(self, mock_process):
        """Test ModelInstance creation."""
        instance = ModelInstance(
            model_name="test-model",
            model_type="llama", 
            endpoint="ipc:///tmp/test",
            process=mock_process
        )
        
        assert instance.model_name == "test-model"
        assert instance.model_type == "llama"
        assert instance.endpoint == "ipc:///tmp/test"
        assert instance.started_at is not None
        
    def test_is_alive(self, mock_process):
        """Test is_alive method."""
        instance = ModelInstance(
            model_name="test-model",
            model_type="llama",
            endpoint="ipc:///tmp/test", 
            process=mock_process
        )
        
        # Process is alive
        mock_process.poll.return_value = None
        assert instance.is_alive() is True
        
        # Process is dead
        mock_process.poll.return_value = 1
        assert instance.is_alive() is False
        
    def test_terminate(self, mock_process):
        """Test terminate method."""
        instance = ModelInstance(
            model_name="test-model",
            model_type="llama",
            endpoint="ipc:///tmp/test",
            process=mock_process
        )
        
        instance.terminate()
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once_with(timeout=10)


class TestManagementCommand:
    """Test the ManagementCommand dataclass."""
    
    def test_command_creation(self):
        """Test ManagementCommand creation."""
        cmd = ManagementCommand(
            command="status",
            params={"key": "value"}
        )
        
        assert cmd.command == "status"
        assert cmd.params == {"key": "value"}
        assert cmd.correlation_id is not None
        
    def test_correlation_id_auto_generation(self):
        """Test that correlation_id is auto-generated."""
        cmd1 = ManagementCommand(command="test", params={})
        cmd2 = ManagementCommand(command="test", params={})
        
        assert cmd1.correlation_id != cmd2.correlation_id


class TestManagementService:
    """Test the ManagementService class."""
        
    def test_service_initialization(self, temp_config):
        """Test service initialization."""
        service = ManagementService(config_path=temp_config)
        
        assert service.client_endpoint == "ipc:///tmp/test-symphony-ipc"
        assert service.cli_endpoint == "ipc:///tmp/test-symphony-cli"
        assert service.model_backends["llama"] == "l4m_backend.py"
        assert service.model_backends["test"] == "test_backend.py"
        
    def test_load_config_file_not_found(self):
        """Test config loading with non-existent file."""
        with pytest.raises(SystemExit):
            ManagementService(config_path="/nonexistent/config.json")
            
    def test_load_config_invalid_json(self):
        """Test config loading with invalid JSON."""
        temp_dir = tempfile.mkdtemp()
        bad_config_file = os.path.join(temp_dir, "bad_config.json")
        
        try:
            with open(bad_config_file, 'w') as f:
                f.write("invalid json {")
                
            with pytest.raises(SystemExit):
                ManagementService(config_path=bad_config_file)
        finally:
            if os.path.exists(bad_config_file):
                os.unlink(bad_config_file)
            os.rmdir(temp_dir)
        
    def test_generate_unique_endpoint(self, temp_config):
        """Test unique endpoint generation."""
        service = ManagementService(config_path=temp_config)
        
        endpoint1 = service._generate_unique_endpoint()
        endpoint2 = service._generate_unique_endpoint()
        
        assert endpoint1 != endpoint2
        assert endpoint1.startswith("ipc:///tmp/symphony-model-")
        assert endpoint2.startswith("ipc:///tmp/symphony-model-")
        
    def test_get_model_type(self, temp_config):
        """Test model type determination."""
        service = ManagementService(config_path=temp_config)
        
        assert service._get_model_type("llama-7b") == "llama"
        assert service._get_model_type("deepseek-coder") == "deepseek"
        assert service._get_model_type("unknown-model") == "llama"
        
    @patch('management_service.subprocess.Popen')
    def test_create_model_instance_success(self, mock_popen, temp_config):
        """Test successful model instance creation."""
        service = ManagementService(config_path=temp_config)
        
        # Mock successful process with proper stdout
        mock_process = Mock()
        mock_process.poll.return_value = None
        
        # Mock stdout to return empty strings (simulating no output)
        mock_stdout = Mock()
        mock_stdout.readline.side_effect = ['', '', '']  # Empty strings to end iteration
        mock_process.stdout = mock_stdout
        
        mock_popen.return_value = mock_process
        
        # Create a mock backend script
        backend_path = os.path.join(service.backend_base_path, "l4m_backend.py")
        os.makedirs(service.backend_base_path, exist_ok=True)
        
        with open(backend_path, 'w') as f:
            f.write("#!/usr/bin/env python3\nprint('test backend')")
        
        try:
            endpoint = service._create_model_instance("test-model")
            
            assert endpoint is not None
            assert endpoint.startswith("ipc:///tmp/symphony-model-")
            assert "test-model" in service.model_instances
            
            instance = service.model_instances["test-model"]
            assert instance.model_name == "test-model"
            assert instance.endpoint == endpoint
            
        finally:
            # Cleanup
            if os.path.exists(backend_path):
                os.unlink(backend_path)
            if os.path.exists(service.backend_base_path):
                os.rmdir(service.backend_base_path)
        
    def test_create_model_instance_script_not_found(self, temp_config):
        """Test model instance creation with missing script."""
        service = ManagementService(config_path=temp_config)
        
        endpoint = service._create_model_instance("nonexistent-model")
        assert endpoint is None
        
    def test_handle_status_command(self, temp_config, mock_process):
        """Test status command handling."""
        service = ManagementService(config_path=temp_config)
        
        # Add a mock model instance
        mock_process.poll.return_value = None
        instance = ModelInstance(
            model_name="test-model",
            model_type="llama",
            endpoint="ipc:///tmp/test",
            process=mock_process
        )
        service.model_instances["test-model"] = instance
        
        command = ManagementCommand(command="status", params={})
        response = service._handle_status_command(command)
        
        assert response["success"] is True
        assert response["correlation_id"] == command.correlation_id
        assert len(response["data"]["models"]) == 1
        assert response["data"]["models"][0]["name"] == "test-model"
        
    def test_handle_load_model_command(self, temp_config):
        """Test load-model command handling."""
        service = ManagementService(config_path=temp_config)
        
        command = ManagementCommand(
            command="load-model",
            params={"model_name": "test-model"}
        )
        
        with patch.object(service, '_get_or_create_model_instance') as mock_get_or_create:
            mock_get_or_create.return_value = "ipc:///tmp/test-endpoint"
            
            response = service._handle_load_model_command(command)
            
            assert response["success"] is True
            assert response["data"]["endpoint"] == "ipc:///tmp/test-endpoint"
            mock_get_or_create.assert_called_once_with("test-model", None)
            
    def test_handle_load_model_command_missing_name(self, temp_config):
        """Test load-model command with missing model name."""
        service = ManagementService(config_path=temp_config)
        
        command = ManagementCommand(command="load-model", params={})
        response = service._handle_load_model_command(command)
        
        assert response["success"] is False
        assert "model_name parameter required" in response["error"]
        
    def test_handle_unload_model_command(self, temp_config):
        """Test unload-model command handling."""
        service = ManagementService(config_path=temp_config)
        
        # Add a mock model instance
        mock_process = Mock()
        mock_process.poll.return_value = None  # Process is alive
        instance = ModelInstance(
            model_name="test-model",
            model_type="llama",
            endpoint="ipc:///tmp/test",
            process=mock_process
        )
        service.model_instances["test-model"] = instance
        
        command = ManagementCommand(
            command="unload-model",
            params={"model_name": "test-model"}
        )
        
        response = service._handle_unload_model_command(command)
        
        assert response["success"] is True
        assert "test-model" not in service.model_instances
        mock_process.terminate.assert_called_once()
        
    def test_handle_unload_model_command_not_found(self, temp_config):
        """Test unload-model command with non-existent model."""
        service = ManagementService(config_path=temp_config)
        
        command = ManagementCommand(
            command="unload-model",
            params={"model_name": "nonexistent-model"}
        )
        
        response = service._handle_unload_model_command(command)
        
        assert response["success"] is False
        assert "not found" in response["error"]
        
    def test_handle_stop_service_command(self, temp_config):
        """Test stop-service command handling."""
        service = ManagementService(config_path=temp_config)
        
        command = ManagementCommand(command="stop-service", params={})
        response = service._handle_stop_service_command(command)
        
        assert response["success"] is True
        assert service.shutdown_requested is True
        
    def test_check_model_instances_health(self, temp_config):
        """Test model instance health checking."""
        service = ManagementService(config_path=temp_config)
        
        # Add alive and dead model instances
        alive_process = Mock()
        alive_process.poll.return_value = None
        
        dead_process = Mock()
        dead_process.poll.return_value = 1
        
        alive_instance = ModelInstance(
            model_name="alive-model",
            model_type="llama",
            endpoint="ipc:///tmp/alive",
            process=alive_process
        )
        
        dead_instance = ModelInstance(
            model_name="dead-model", 
            model_type="llama",
            endpoint="ipc:///tmp/dead",
            process=dead_process
        )
        
        service.model_instances["alive-model"] = alive_instance
        service.model_instances["dead-model"] = dead_instance
        
        service._check_model_instances_health()
        
        # Only alive instance should remain
        assert "alive-model" in service.model_instances
        assert "dead-model" not in service.model_instances
