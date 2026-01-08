from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

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


@dataclass
class GenerationMetrics:
    """Benchmarking metrics for token generation."""
    ttft_ms: float  # Time to first token in milliseconds
    total_tokens: int  # Total tokens generated
    tokens_per_second: float  # Throughput
    total_latency_s: float  # End-to-end latency in seconds
