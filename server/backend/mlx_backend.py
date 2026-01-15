from .mlx_runner import MLXRunner
from ..cache_utils import get_model_path
from fastapi import HTTPException
from ..schemas import ChatMessage, ChatCompletionRequest, downloadRequest, ResponseRequest
from ..hf_downloader import pull_model

import logging
import asyncio
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
    try:
        json_schema = None
        if request.response_format:
            if request.response_format.get("type") == "json_schema":
                schema_info = request.response_format.get("json_schema", {})
                json_schema = json.dumps(schema_info.get("schema", {}))
            elif request.response_format.get("type") == "json_object":
                # Fallback for json_object type
                json_schema = "{}" 

        accumulated_text = ""
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
            json_schema=json_schema,
        ):
            accumulated_text += token
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
                if any(stop in accumulated_text for stop in stop_sequences):
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

    # Final response
    final_response = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }

    yield f"data: {json.dumps(final_response)}\n\n"
    yield "data: [DONE]\n\n"


async def generate_response(request: ResponseRequest) -> Dict[str, Any]:
    """Generate complete non-streaming chat completion response."""
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    runner = get_or_load_model(request.model)

    # Convert messages to dict format for runner
    message_dicts = format_chat_messages_for_runner(request.messages)

    # Let the runner format with chat templates
    prompt = runner._format_conversation(message_dicts, use_chat_template=True)

    json_schema = None
    if request.response_format:
        if request.response_format.get("type") == "json_schema":
            schema_info = request.response_format.get("json_schema", {})
            json_schema = json.dumps(schema_info.get("schema", {}))
        elif request.response_format.get("type") == "json_object":
            # Fallback for json_object type
            json_schema = "{}"

    response_text = await asyncio.to_thread(
        runner.generate_batch,
        prompt=prompt,
        max_tokens=runner.get_effective_max_tokens(
            request.max_tokens or _default_max_tokens, interactive=False
        ),
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
        use_chat_template=False,
        json_schema=json_schema,
    )

    # Handle stop sequences if provided
    if request.stop:
        stop_sequences = (
            request.stop if isinstance(request.stop, list) else [request.stop]
        )
        min_index = len(response_text)
        found_stop = False
        for stop in stop_sequences:
            index = response_text.find(stop)
            if index != -1:
                min_index = min(min_index, index)
                found_stop = True
        
        if found_stop:
            response_text = response_text[:min_index]

    prompt_tokens = count_tokens(prompt)
    completion_tokens = count_tokens(response_text)

    return {
        "id": completion_id,
        "object": "response",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def generate_response_stream(
    request: ResponseRequest,
) -> AsyncGenerator[str, None]:
    """Generate streaming response with OpenAI-aligned SSE events.
    
    Events emitted:
    - response.created: Initial response object
    - response.content_part.delta: Streaming text chunks
    - response.done: Final response complete
    """
    response_id = f"resp-{uuid.uuid4()}"
    created = int(time.time())
    runner = get_or_load_model(request.model)

    # Convert messages to dict format for runner
    message_dicts = format_chat_messages_for_runner(request.messages)

    # Let the runner format with chat templates
    prompt = runner._format_conversation(message_dicts, use_chat_template=True)

    # Emit response.created event
    created_event = {
        "type": "response.created",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "model": request.model,
            "status": "in_progress",
            "output": [],
        },
    }
    yield f"event: response.created\ndata: {json.dumps(created_event)}\n\n"

    # Emit output_item.added for the message
    output_item_added = {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "id": f"msg-{uuid.uuid4()}",
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
            "content": [],
        },
    }
    yield f"event: response.output_item.added\ndata: {json.dumps(output_item_added)}\n\n"

    # Emit content_part.added
    content_part_added = {
        "type": "response.content_part.added",
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "text", "text": ""},
    }
    yield f"event: response.content_part.added\ndata: {json.dumps(content_part_added)}\n\n"

    # Stream tokens
    accumulated_text = ""
    try:
        json_schema = None
        if request.response_format:
            if request.response_format.get("type") == "json_schema":
                schema_info = request.response_format.get("json_schema", {})
                json_schema = json.dumps(schema_info.get("schema", {}))
            elif request.response_format.get("type") == "json_object":
                json_schema = "{}"

        for token in runner.generate_streaming(
            prompt=prompt,
            max_tokens=runner.get_effective_max_tokens(
                request.max_tokens or _default_max_tokens, interactive=False
            ),
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            use_chat_template=False,
            use_chat_stop_tokens=False,
            json_schema=json_schema,
        ):
            accumulated_text += token

            # Emit content_part.delta
            delta_event = {
                "type": "response.content_part.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": {"type": "text_delta", "text": token},
            }
            yield f"event: response.content_part.delta\ndata: {json.dumps(delta_event)}\n\n"

            # Check for stop sequences
            if request.stop:
                stop_sequences = (
                    request.stop if isinstance(request.stop, list) else [request.stop]
                )
                if any(stop in accumulated_text for stop in stop_sequences):
                    break

    except Exception as e:
        error_event = {
            "type": "error",
            "error": {"type": "server_error", "message": str(e)},
        }
        yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
        return

    # Emit content_part.done
    content_part_done = {
        "type": "response.content_part.done",
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "text", "text": accumulated_text},
    }
    yield f"event: response.content_part.done\ndata: {json.dumps(content_part_done)}\n\n"

    # Emit output_item.done
    output_item_done = {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "id": output_item_added["item"]["id"],
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "text", "text": accumulated_text}],
        },
    }
    yield f"event: response.output_item.done\ndata: {json.dumps(output_item_done)}\n\n"

    # Emit response.done
    prompt_tokens = count_tokens(prompt)
    completion_tokens = count_tokens(accumulated_text)
    done_event = {
        "type": "response.done",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "model": request.model,
            "status": "completed",
            "output": [output_item_done["item"]],
            "usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        },
    }
    yield f"event: response.done\ndata: {json.dumps(done_event)}\n\n"


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

