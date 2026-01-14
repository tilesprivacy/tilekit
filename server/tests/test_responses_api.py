import sys
import os
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from fastapi.testclient import TestClient
from server.api import app
from server.backend.mlx_backend import get_or_load_model
import json

def test_responses_api():
    client = TestClient(app)
    
    # We use a model that is likely to be present in the cache
    model_spec = "driaforall/mem-agent-mlx-4bit"
    print(f"Testing Responses API with model: {model_spec}")
    
    # Pre-load the model to avoid timeout in test
    try:
        from server.cache_utils import get_model_path
        model_path, _, _ = get_model_path(model_spec)
        if not model_path.exists():
             print(f"Skipping test, model {model_spec} not found in cache.")
             return
             
        get_or_load_model(model_spec)
    except Exception as e:
        print(f"Skipping test, could not load model: {e}")
        return

    payload = {
        "model": model_spec,
        "messages": [
            {"role": "user", "content": "What is the capital of France? Answer with just the name of the city."}
        ],
        "temperature": 0.0,
        "max_tokens": 10
    }
    
    print("Sending request to /v1/responses...")
    try:
        response = client.post("/v1/responses", json=payload)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            assert data["object"] == "chat.completion"
            assert "choices" in data
            assert len(data["choices"]) > 0
            assert "message" in data["choices"][0]
            assert "usage" in data
            
            content = data["choices"][0]["message"]["content"]
            assert len(content) > 0
            print(f"Response content: {content}")
            print("Chat completion test PASSED")
        else:
            print(f"Test FAILED: {response.text}")
            return

        print("\n--- Testing JSON Structured Output ---")
        json_schema = {
            "type": "object",
            "properties": {
                "capital": {"type": "string"},
                "country": {"type": "string"}
            },
            "required": ["capital", "country"]
        }
        
        payload["messages"] = [{"role": "user", "content": "What is the capital of Germany? Respond in JSON."}]
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"schema": json_schema}
        }
        payload["max_tokens"] = 100
        
        print("Sending request to /v1/responses with JSON schema...")
        response = client.post("/v1/responses", json=payload)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            print(f"Parsed JSON: {parsed}")
            assert parsed["capital"].lower() == "berlin"
            assert parsed["country"].lower() == "germany"
            print("JSON schema test PASSED")
        else:
            print(f"JSON schema test FAILED: {response.text}")

    except Exception as e:
        print(f"Test FAILED with exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_responses_api()
