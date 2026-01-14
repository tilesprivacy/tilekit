import os
import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

# Add the project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from server.backend.mlx_runner import MLXRunner

class TestLoRA(unittest.TestCase):
    @patch("server.backend.mlx_runner.engine_load_model")
    def test_lora_path_passed(self, mock_load):
        """Verify that adapter_path is actually passed to the engine loader."""
        model_path = "/tmp/fake_model"
        adapter_path = "/tmp/fake_adapter"
        
        runner = MLXRunner(model_path, adapter_path=adapter_path)
        runner.load_model()
        
        mock_load.assert_called_once_with(
            model_path, adapter_path=adapter_path
        )
        print("LoRA path propagation verification (MLXRunner -> engine): PASSED")

    @patch("mlx_lm.utils.load")
    def test_model_kit_passes_adapter(self, mock_mlx_load):
        """Verify that ModelKit passes the adapter path to mlx_lm."""
        from server.backend.mlx_engine.model_kit.model_kit import ModelKit
        
        # Fixed mock return value to avoid unpack error
        mock_mlx_load.return_value = (MagicMock(), MagicMock())
        
        # Mocking config.json
        with patch("pathlib.Path.read_text", return_value='{"model_type": "llama"}'):
            with patch("json.loads", return_value={"model_type": "llama"}):
                model_kit = ModelKit(Path("/tmp/fake_model"), adapter_path="/tmp/fake_adapter")
                
                mock_mlx_load.assert_called_once()
                args, kwargs = mock_mlx_load.call_args
                self.assertEqual(kwargs.get("adapter_path"), Path("/tmp/fake_adapter"))
                print("ModelKit LoRA path propagation (ModelKit -> mlx_lm): PASSED")

if __name__ == "__main__":
    unittest.main()
