from .mlx_runner import MLXRunner
from ..cache_utils import get_model_path
from fastapi import HTTPException
from ..schemas import ChatMessage, ChatCompletionRequest, downloadRequest, GenerationMetrics
from ..hf_downloader import pull_model

import logging
import json
import time
import uuid
from collections.abc import AsyncGenerator

logger = logging.getLogger("app")

from typing import Any, Dict, Iterator, List, Optional, Union

_model_cache: Dict[str, MLXRunner] = {}
_default_max_tokens: Optional[int] = None  # Use dynamic model-aware limits by default
_current_model_path: Optional[str] = None


def download_model(model_name: str):
    """Download the model"""
    if pull_model(model_name):
        return {"message": "Model downloaded"}
    else:
        raise HTTPException(status_code=400, detail="Downloading model failed")


def get_or_load_model(model_spec: str, verbose: bool = False) -> MLXRunner:
    """Get model from cache or load it if not cached."""
    global _model_cache, _current_model_path

    # Use the existing model path resolution from cache_utils

    try:
        model_path, model_name, commit_hash = get_model_path(model_spec)
        if not model_path.exists():
            logger.info(f"Model {model_spec} not found in cache")
            raise HTTPException(
                status_code=404, detail=f"Model {model_spec} not found in cache"
            )
    except Exception as e:
        logger.info(f"Model {model_spec} not found in: {str(e)}")
        raise HTTPException(
            status_code=404, detail=f"Model {model_spec} not found: {str(e)}"
        )

    # Check if it's an MLX model

    model_path_str = str(model_path)

    # Check if we need to load a different model
    if _current_model_path != model_path_str:
        # Proactively clean up any previously loaded runner to release memory
        if _model_cache:
            try:
                for _old_runner in list(_model_cache.values()):
                    try:
                        _old_runner.cleanup()
                    except Exception:
                        pass
            finally:
                _model_cache.clear()

        # Load new model
        if verbose:
            print(f"Loading model: {model_name}")

        logger.info(f"Loading model: {model_name}")
        runner = MLXRunner(model_path_str, verbose=verbose)
        runner.load_model()

        _model_cache[model_path_str] = runner
        _current_model_path = model_path_str
    else:
        logger.info(f"Model {model_name} already in memory")

    return _model_cache[model_path_str]

async def generate_chat_stream(
    messages: List[ChatMessage], request: ChatCompletionRequest
) -> AsyncGenerator[str, None]:
    """Generate streaming chat completion response."""

    _messages = messages
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    runner = get_or_load_model(request.model)
    if request.chat_start:
        _messages.extend(request.messages)
    # Convert messages to dict format for runner
    message_dicts = format_chat_messages_for_runner(_messages)

    # Let the runner format with chat templates
    prompt = runner._format_conversation(message_dicts, use_chat_template=True)

    # Yield initial response
    initial_response = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }
        ],
    }

    yield f"data: {json.dumps(initial_response)}\n\n"

    # Stream tokens
    metrics = None
    try:
        for token in runner.generate_streaming(
            prompt=prompt,
            max_tokens=runner.get_effective_max_tokens(
                request.max_tokens or _default_max_tokens, interactive=False
            ),
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            use_chat_template=False,  # Already applied in _format_conversation
            use_chat_stop_tokens=False,  # Server mode shouldn't stop on chat markers
        ):
            # Check if this is metrics object (last item yielded)
            if isinstance(token, GenerationMetrics):
                metrics = token
                continue

            chunk_response = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {"index": 0, "delta": {"content": token}, "finish_reason": None}
                ],
            }

            yield f"data: {json.dumps(chunk_response)}\n\n"

            # Check for stop sequences
            if request.stop:
                stop_sequences = (
                    request.stop if isinstance(request.stop, list) else [request.stop]
                )
                if any(stop in token for stop in stop_sequences):
                    break

    except Exception as e:
        error_response = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
            "error": str(e),
        }
        yield f"data: {json.dumps(error_response)}\n\n"

    # Final response with metrics
    final_response = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }

    # Include benchmarking metrics if available
    if metrics:
        final_response["metrics"] = {
            "ttft_ms": metrics.ttft_ms,
            "total_tokens": metrics.total_tokens,
            "tokens_per_second": metrics.tokens_per_second,
            "total_latency_s": metrics.total_latency_s,
        }

    yield f"data: {json.dumps(final_response)}\n\n"
    yield "data: [DONE]\n\n"

def format_chat_messages_for_runner(
    messages: List[ChatMessage],
) -> List[Dict[str, str]]:
    """Convert chat messages to format expected by MLXRunner.

    Returns messages in dict format for the runner to apply chat templates.
    """
    return [{"role": msg.role, "content": msg.content} for msg in messages]


def count_tokens(text: str) -> int:
    """Rough token count estimation."""
    return int(len(text.split()) * 1.3)  # Approximation, convert to int

