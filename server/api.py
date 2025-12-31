from fastapi import FastAPI, HTTPException
from .config import SYSTEM_PROMPT
import logging
import json
import time
import uuid
import sys
from collections.abc import AsyncGenerator
from typing import Any, Dict, List, Optional, Union

from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .cache_utils import get_model_path
from .hf_downloader import pull_model

from server.mem_agent.utils import (
    extract_python_code,
    extract_reply,
    extract_thoughts,
    create_memory_if_not_exists,
    format_results,
)
from server.mem_agent.engine import execute_sandboxed_code

# Global model cache and configuration

if sys.platform == "darwin":
    from .mlx_api import generate_chat_stream, get_or_load_model


logger = logging.getLogger("app")
_current_model_path: Optional[str] = None
_default_max_tokens: Optional[int] = None  # Use dynamic model-aware limits by default
_max_tool_turns = 5
_memory_path = ""


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    repetition_penalty: Optional[float] = 1.1


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


_messages: list[ChatMessage] = []


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    chat_start: bool
    python_code: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    repetition_penalty: Optional[float] = 1.1


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    # usage: Dict[str, int]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "mlx-knife"
    permission: List = []
    context_length: Optional[int] = None


class StartRequest(BaseModel):
    model: str
    memory_path: str


class downloadRequest(BaseModel):
    model: str


app = FastAPI()


@app.get("/ping")
async def ping():
    return {"message": "Badda-Bing Badda-Bang"}


@app.post("/download")
async def download(request: downloadRequest):
    """Download the model"""
    try:
        if pull_model(request.model):
            return {"message": "Model downloaded"}
        else:
            raise HTTPException(status_code=400, detail="Downloading model failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/start")
async def start_model(request: StartRequest):
    """Load the model and start the agent"""
    global _messages, _runner, _memory_path

    _messages = [ChatMessage(role="system", content=SYSTEM_PROMPT)]
    _memory_path = request.memory_path

    get_or_load_model(request.model)
    return {"message": "Model loaded"}


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion."""
    global _messages, _max_tool_turns, _memory_path
    try:

        if request.stream:
            result = ({}, "")
            if request.python_code:
                create_memory_if_not_exists()
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
                generate_chat_stream(request.model, request.messages, request),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache"},
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
