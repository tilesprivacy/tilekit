from fastapi import FastAPI, HTTPException

from .schemas import ChatMessage, ChatCompletionRequest, StartRequest, downloadRequest, ResponseRequest
from .config import SYSTEM_PROMPT
import logging
import sys
from typing import Optional

from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .hf_downloader import pull_model

from server.mem_agent.utils import (
    create_memory_if_not_exists,
    format_results,
)
from server.mem_agent.engine import execute_sandboxed_code

from . import runtime

logger = logging.getLogger("app")
_current_model_path: Optional[str] = None
_default_max_tokens: Optional[int] = None  # Use dynamic model-aware limits by default
_memory_path = ""

_messages: list[ChatMessage] = []


app = FastAPI()


@app.get("/ping")
async def ping():
    return {"message": "Badda-Bing Badda-Bang"}


@app.post("/download")
async def download(request: downloadRequest):
    """Download the model"""
    runtime.backend.download_model(request.model)

@app.post("/start")
async def start_model(request: StartRequest):
    """Load the model and start the agent"""
    global _messages, _runner, _memory_path

    _messages = [ChatMessage(role="system", content=SYSTEM_PROMPT)]
    _memory_path = request.memory_path
    logger.info(f"{runtime.backend}")
    runtime.backend.get_or_load_model(request.model)
    return {"message": "Model loaded"}


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion."""
    global _messages, _memory_path
    try:

        if request.stream:
            result = ({}, "")
            if request.python_code:
                result = execute_sandboxed_code(
                    code=request.python_code,
                    allowed_path=_memory_path,
                    import_module="server.mem_agent.tools",
                )

            _messages.append(
                ChatMessage(role="user", content=format_results(result[0], result[1]))
            )

            # Streaming response
            return StreamingResponse(
                runtime.backend.generate_chat_stream(_messages, request),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache"},
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/responses")
async def create_response(request: ResponseRequest):
    """Create a non-streaming completion response."""
    try:
        response = await runtime.backend.generate_response(request)
        return response
    except Exception as e:
        logger.exception("Error in generate_response")
        raise HTTPException(status_code=500, detail=str(e)) from e
