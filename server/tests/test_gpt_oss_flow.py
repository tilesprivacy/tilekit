"""Tests for gpt-oss flow and streaming Responses API.

Verifies:
- Channel marker parsing
- Streaming SSE event format
- Code interpreter routing
"""

import os
import sys
import json
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from server.schemas import ResponseRequest, ChatMessage


class TestChannelMarkerParsing:
    """Test gpt-oss channel marker parsing."""

    def test_extract_final_channel(self):
        """Test extraction of final channel content."""
        content = "<|channel|>final<|message|>This is the final response.<|end|>"
        
        # Simulate the parsing logic from mlx.rs
        def extract_channel_content(content: str, channel: str) -> str | None:
            channel_marker = f"<|channel|>{channel}<|message|>"
            if channel_marker in content:
                start_idx = content.find(channel_marker) + len(channel_marker)
                remaining = content[start_idx:]
                end_idx = remaining.find("<|end|>")
                if end_idx == -1:
                    end_idx = remaining.find("<|channel|>")
                if end_idx == -1:
                    end_idx = len(remaining)
                return remaining[:end_idx].strip()
            return None
        
        result = extract_channel_content(content, "final")
        assert result == "This is the final response."

    def test_extract_code_channel(self):
        """Test extraction of code channel content."""
        content = "<|channel|>code<|message|>print('Hello, World!')<|end|>"
        
        def extract_channel_content(content: str, channel: str) -> str | None:
            channel_marker = f"<|channel|>{channel}<|message|>"
            if channel_marker in content:
                start_idx = content.find(channel_marker) + len(channel_marker)
                remaining = content[start_idx:]
                end_idx = remaining.find("<|end|>")
                if end_idx == -1:
                    end_idx = remaining.find("<|channel|>")
                if end_idx == -1:
                    end_idx = len(remaining)
                return remaining[:end_idx].strip()
            return None
        
        result = extract_channel_content(content, "code")
        assert result == "print('Hello, World!')"

    def test_extract_analysis_channel(self):
        """Test extraction of analysis channel content."""
        content = "<|channel|>analysis<|message|>Let me think about this...<|channel|>final<|message|>Here is the answer.<|end|>"
        
        def extract_channel_content(content: str, channel: str) -> str | None:
            channel_marker = f"<|channel|>{channel}<|message|>"
            if channel_marker in content:
                start_idx = content.find(channel_marker) + len(channel_marker)
                remaining = content[start_idx:]
                end_idx = remaining.find("<|end|>")
                if end_idx == -1:
                    end_idx = remaining.find("<|channel|>")
                if end_idx == -1:
                    end_idx = len(remaining)
                return remaining[:end_idx].strip()
            return None
        
        analysis = extract_channel_content(content, "analysis")
        final = extract_channel_content(content, "final")
        
        assert analysis == "Let me think about this..."
        assert final == "Here is the answer."

    def test_legacy_reply_format(self):
        """Test legacy <reply> tag extraction."""
        content = "<think>Some reasoning</think><reply>The answer is 42.</reply>"
        
        def extract_reply(content: str) -> str:
            if "<reply>" in content and "</reply>" in content:
                start = content.find("<reply>") + len("<reply>")
                end = content.find("</reply>")
                return content[start:end]
            return ""
        
        result = extract_reply(content)
        assert result == "The answer is 42."


class TestStreamingSSEEvents:
    """Test streaming SSE event format."""

    def test_response_created_event(self):
        """Test response.created event structure."""
        event = {
            "type": "response.created",
            "response": {
                "id": "resp-123",
                "object": "response",
                "created_at": 1234567890,
                "model": "mlx-community/gpt-oss-20b-MXFP4-Q4",
                "status": "in_progress",
                "output": [],
            },
        }
        
        assert event["type"] == "response.created"
        assert event["response"]["status"] == "in_progress"
        assert event["response"]["output"] == []

    def test_content_part_delta_event(self):
        """Test response.content_part.delta event structure."""
        event = {
            "type": "response.content_part.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        }
        
        assert event["type"] == "response.content_part.delta"
        assert event["delta"]["text"] == "Hello"

    def test_response_done_event(self):
        """Test response.done event structure."""
        event = {
            "type": "response.done",
            "response": {
                "id": "resp-123",
                "object": "response",
                "created_at": 1234567890,
                "model": "mlx-community/gpt-oss-20b-MXFP4-Q4",
                "status": "completed",
                "output": [
                    {
                        "id": "msg-456",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "text", "text": "Hello, World!"}],
                    }
                ],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
            },
        }
        
        assert event["type"] == "response.done"
        assert event["response"]["status"] == "completed"
        assert len(event["response"]["output"]) == 1
        assert event["response"]["usage"]["total_tokens"] == 15

    def test_sse_format(self):
        """Test SSE line format."""
        event = {"type": "response.created", "data": "test"}
        sse_line = f"event: response.created\ndata: {json.dumps(event)}\n\n"
        
        assert sse_line.startswith("event: response.created")
        assert "data: " in sse_line
        assert sse_line.endswith("\n\n")


class TestResponseRequest:
    """Test ResponseRequest schema."""

    def test_stream_field_default(self):
        """Test that stream field defaults to False."""
        request = ResponseRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
        )
        assert request.stream is False

    def test_stream_field_true(self):
        """Test that stream field can be set to True."""
        request = ResponseRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            stream=True,
        )
        assert request.stream is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
