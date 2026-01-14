import os
import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from server.backend.mlx_runner import MLXRunner
from server.cache_utils import get_model_path

def test_inference():
    model_spec = "driaforall/mem-agent-mlx-4bit"
    print(f"Testing inference with model: {model_spec}")
    
    try:
        model_path, model_name, commit_hash = get_model_path(model_spec)
        if model_path is None or not model_path.exists():
            print(f"Error: Model {model_spec} not found in cache. Please download it first.")
            return

        runner = MLXRunner(str(model_path), verbose=True)
        
        with runner:
            print("\n--- Testing Streaming Generation ---")
            prompt = "Why is the sky blue? Answer in one sentence."
            full_response = ""
            for token in runner.generate_streaming(prompt, max_tokens=50):
                print(token, end="", flush=True)
                full_response += token
            print("\n--- Streaming Done ---\n")
            
            print("\n--- Memory Usage ---")
            print(runner.get_memory_usage())

            print("\n--- Testing JSON Structured Output ---")
            json_schema = {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "confidence": {"type": "number"}
                },
                "required": ["answer", "confidence"]
            }
            import json
            prompt = "What is the capital of France? Return in JSON format."
            full_response = ""
            for token in runner.generate_streaming(prompt, max_tokens=100, json_schema=json.dumps(json_schema)):
                print(token, end="", flush=True)
                full_response += token
            print("\n--- JSON Done ---\n")
            
            try:
                parsed = json.loads(full_response)
                print(f"Parsed JSON: {parsed}")
                assert "answer" in parsed
                assert "confidence" in parsed
                print("JSON Schema verification PASSED")
            except Exception as e:
                print(f"JSON Schema verification FAILED: {e}")

    except Exception as e:
        print(f"Tests failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_inference()
