from fastapi import FastAPI, HTTPException

from .schemas import ChatMessage, ChatCompletionRequest, StartRequest, downloadRequest, ResponseRequest
from .config import SYSTEM_PROMPT, TILES_TRACK
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
from server.mem_agent.engine import VenvStackExecutor
import uuid

from . import runtime

logger = logging.getLogger("app")
_current_model_path: Optional[str] = None
_default_max_tokens: Optional[int] = None  # Use dynamic model-aware limits by default
_memory_path = ""
_executor: Optional[VenvStackExecutor] = None

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
    """Load the model and start the agent.
    
    Uses the system_prompt from the request if provided,
    otherwise falls back to the default SYSTEM_PROMPT.
    """
    global _messages, _runner, _memory_path, _executor

    # Use dynamic system prompt if provided, else fall back to default
    system_prompt = request.system_prompt if request.system_prompt else SYSTEM_PROMPT
    _messages = [ChatMessage(role="system", content=system_prompt)]
    _memory_path = request.memory_path
    
    # Initialize persistent code executor
    session_id = str(uuid.uuid4())
    
    # Check for insider track regression fix
    use_system_python = (TILES_TRACK == "insider")
    _executor = VenvStackExecutor(
        session_id=session_id, 
        workspace_path=_memory_path,
        use_system_python=use_system_python
    )

    
    logger.info(f"Initialized VenvStackExecutor for session {session_id}")
    logger.info(f"{runtime.backend}")
    runtime.backend.get_or_load_model(request.model)
    return {"message": "Model loaded"}




@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion."""
    global _messages, _memory_path, _executor
    try:
        # If this is a fresh start, reset the message history to (System + current messages)
        if request.chat_start:
            logger.info("[CHAT] Starting fresh interaction, resetting message history")
            # Preserve the system message (first message) if it exists
            system_msg = _messages[0] if _messages and _messages[0].role == "system" else None
            _messages = []
            if system_msg:
                _messages.append(system_msg)
            
            # Add the new user messages from the request
            _messages.extend(request.messages)
            logger.debug(f"[CHAT] New message history length: {len(_messages)}")


        if request.stream:
            if request.python_code:
                logger.info(f"[CODE_INTERPRETER] Received code to execute:\n{request.python_code}")
                
                # Ensure executor is initialized
                if not _executor:
                    logger.warning("[CODE_INTERPRETER] Executor not initialized, creating default")
                    _executor = VenvStackExecutor(session_id="default", workspace_path=_memory_path)
                
                # Execute code using the persistent venvstack sandbox
                try:
                    result_vars, result_error = _executor.execute(code=request.python_code)
                    logger.info(f"[CODE_INTERPRETER] Execution result: vars={result_vars}, error={result_error}")
                except Exception as exec_error:
                    logger.error(f"[CODE_INTERPRETER] Execution failed: {exec_error}")
                    result_vars, result_error = ({}, str(exec_error))

                # For a relay turn, the "result" is the user content for the NEXT turn
                formatted_result = format_results(result_vars, result_error)
                logger.debug(f"[CODE_INTERPRETER] Formatted result for model: {formatted_result[:200]}...")
                
                _messages.append(
                    ChatMessage(role="user", content=formatted_result)
                )
            else:
                logger.debug("[CODE_INTERPRETER] No python_code in request (standard turn)")

            # Streaming response
            return StreamingResponse(
                runtime.backend.generate_chat_stream(_messages, request),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache"},
            )
    except Exception as e:
        logger.exception("[CODE_INTERPRETER] Error in create_chat_completion")
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/v1/responses")
async def create_response(request: ResponseRequest):
    """Create a response with optional streaming support.
    
    When stream=true, returns SSE events:
    - response.created
    - response.output_item.added
    - response.content_part.delta (for each token)
    - response.done
    """
    try:
        if request.stream:
            return StreamingResponse(
                runtime.backend.generate_response_stream(request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            response = await runtime.backend.generate_response(request)
            return response
    except Exception as e:
        logger.exception("Error in create_response")
        raise HTTPException(status_code=500, detail=str(e)) from e

