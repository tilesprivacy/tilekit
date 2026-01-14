"""
Enhanced MLX model runner using the vendored mlx_engine.
Provides ollama-like run experience with streaming and interactive chat.
"""

import sys
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Dict, Optional, List, Any

# Add current directory to sys.path to support absolute imports in vendored mlx_engine
sys.path.append(os.path.dirname(__file__))

if sys.platform == "darwin":
    import mlx.core as mx
else:
    mx = None

from .mlx_engine import load_model as engine_load_model
from .mlx_engine import create_generator as engine_create_generator
from .mlx_engine import tokenize as engine_tokenize

from ..reasoning_utils import ReasoningExtractor, StreamingReasoningParser


def get_model_context_length(model_path: str) -> int:
    """Extract max_position_embeddings from model config."""
    config_path = os.path.join(model_path, "config.json")

    try:
        with open(config_path) as f:
            config = json.load(f)

        context_keys = [
            "max_position_embeddings",
            "n_positions",
            "context_length",
            "max_sequence_length",
            "seq_len",
        ]

        for key in context_keys:
            if key in config:
                return config[key]

        return 4096

    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return 4096


class MLXRunner:
    """Direct MLX model runner using mlx_engine."""

    def __init__(
        self, model_path: str, adapter_path: Optional[str] = None, verbose: bool = False
    ):
        self.model_path = Path(model_path)
        self.adapter_path = adapter_path
        self.model_kit = None
        self._memory_baseline = None
        self._context_length = None
        self.verbose = verbose
        self._model_loaded = False
        self._context_entered = False

    def __enter__(self):
        if self._context_entered:
            raise RuntimeError("MLXRunner context manager cannot be entered multiple times")
        self._context_entered = True
        try:
            self.load_model()
            return self
        except Exception:
            self._context_entered = False
            self.cleanup()
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._context_entered = False
        self.cleanup()
        return False

    def load_model(self):
        """Load the model via mlx_engine."""
        if self._model_loaded:
            return

        if self.verbose:
            print(f"Loading model from {self.model_path}...")
        start_time = time.time()

        try:
            mx.clear_cache()
        except Exception:
            pass
        self._memory_baseline = mx.get_active_memory() / 1024**3

        try:
            # mlx_engine.load_model returns a ModelKit or VisionModelKit
            self.model_kit = engine_load_model(
                str(self.model_path), adapter_path=self.adapter_path
            )

            load_time = time.time() - start_time
            current_memory = mx.get_active_memory() / 1024**3
            model_memory = current_memory - self._memory_baseline

            if self.verbose:
                print(f"Model loaded in {load_time:.1f}s")
                print(f"Memory: {model_memory:.1f}GB model, {current_memory:.1f}GB total")

            self._context_length = get_model_context_length(str(self.model_path))
            self._model_loaded = True

        except Exception as e:
            self.model_kit = None
            self._model_loaded = False
            mx.clear_cache()
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}") from e

    def cleanup(self):
        """Clean up model resources."""
        if self.verbose and self._model_loaded:
            print("Cleaning up model...")

        self.model_kit = None
        self._model_loaded = False

        import gc
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass

    def get_effective_max_tokens(
        self, requested_tokens: Optional[int], interactive: bool = False
    ) -> int:
        if not self._context_length:
            return requested_tokens or (4096 if interactive else 2048)

        # In interactive mode, we aim for a larger context but leave room for the prompt
        limit = self._context_length if interactive else self._context_length // 2
        return min(requested_tokens or limit, limit)

    def _get_chat_stop_tokens(self) -> List[str]:
        """Get chat stop tokens from tokenizer."""
        stop_tokens = []
        if not self.model_kit:
            return stop_tokens

        tokenizer = self.model_kit.tokenizer
        if hasattr(tokenizer, "eos_token") and tokenizer.eos_token:
            stop_tokens.append(tokenizer.eos_token)

        # Common chat tokens if not present in eos_token_ids
        for token in ["<|end|>", "<|im_end|>", "</s>", "<|eot_id|>"]:
            if token not in stop_tokens:
                # Basic check if token exists in vocab
                try:
                    if tokenizer.encode(token, add_special_tokens=False):
                        stop_tokens.append(token)
                except Exception:
                    pass
        return stop_tokens

    def generate_streaming(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        repetition_context_size: int = 20,
        use_chat_template: bool = True,
        use_chat_stop_tokens: bool = False,
        interactive: bool = False,
        hide_reasoning: bool = False,
        json_schema: Optional[str] = None,
    ) -> Iterator[str]:
        """Generate text using mlx_engine's generator."""
        if not self.model_kit:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Apply chat template if available and requested
        if (
            use_chat_template
            and hasattr(self.model_kit.tokenizer, "chat_template")
            and self.model_kit.tokenizer.chat_template
        ):
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.model_kit.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        prompt_tokens = engine_tokenize(self.model_kit, formatted_prompt)
        effective_max_tokens = self.get_effective_max_tokens(max_tokens, interactive)

        # Handle stop strings/tokens
        stop_strings = []
        if use_chat_stop_tokens:
            stop_strings.extend(self._get_chat_stop_tokens())

        # Initialize reasoning parser
        reasoning_parser = None
        model_name = getattr(self.model_kit.tokenizer, "name_or_path", "") or ""
        model_type = ReasoningExtractor.detect_model_type(str(model_name).lower())
        if model_type:
            reasoning_parser = StreamingReasoningParser(model_type, hide_reasoning=hide_reasoning)

        generator = engine_create_generator(
            self.model_kit,
            prompt_tokens,
            max_tokens=effective_max_tokens,
            temp=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            json_schema=json_schema,
            stop_strings=stop_strings if stop_strings else None,
        )

        for result in generator:
            if result.text:
                if reasoning_parser:
                    for formatted_token in reasoning_parser.process_token(result.text):
                        yield formatted_token
                else:
                    yield result.text
            
            if result.stop_condition:
                break

        if reasoning_parser:
            yield from reasoning_parser.finalize()

    def generate_batch(self, *args, **kwargs) -> str:
        """Simple wrapper for generate_streaming to collect all tokens."""
        return "".join(self.generate_streaming(*args, **kwargs))

    def _format_conversation(self, messages: List[Dict[str, str]], use_chat_template: bool = True) -> str:
        """Format conversation using tokenizer's template."""
        if not self.model_kit:
            raise RuntimeError("Model needed for formatting")
            
        if use_chat_template and hasattr(self.model_kit.tokenizer, "chat_template") and self.model_kit.tokenizer.chat_template:
            return self.model_kit.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        
        # Fallback manual formatting
        formatted = ""
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            formatted += f"{role}: {content}\n"
        formatted += "Assistant: "
        return formatted

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics in GB."""
        try:
            current_memory = mx.get_active_memory() / 1024**3
            peak_memory = mx.get_peak_memory() / 1024**3
        except Exception:
            current_memory = 0.0
            peak_memory = 0.0

        return {
            "current_gb": current_memory,
            "peak_gb": peak_memory,
            "model_gb": (
                current_memory - self._memory_baseline if self._memory_baseline else 0
            ),
        }

    def interactive_chat(
        self,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        use_chat_template: bool = True,
    ):
        """Run an interactive chat session."""
        print("Starting interactive chat. Type 'exit' or 'quit' to end.\n")

        conversation_history = []
        if system_prompt:
            conversation_history.append({"role": "system", "content": system_prompt})

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() in ["exit", "quit", "q"]:
                    print("\nGoodbye!")
                    break

                if not user_input:
                    continue

                conversation_history.append({"role": "user", "content": user_input})
                prompt = self._format_conversation(conversation_history, use_chat_template=use_chat_template)

                print("\nAssistant: ", end="", flush=True)

                response_tokens = []
                for token in self.generate_streaming(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    use_chat_template=False, 
                    use_chat_stop_tokens=True,
                    interactive=True,
                ):
                    print(token, end="", flush=True)
                    response_tokens.append(token)

                print()
                assistant_response = "".join(response_tokens).strip()
                conversation_history.append({"role": "assistant", "content": assistant_response})

            except KeyboardInterrupt:
                print("\n\nChat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}")
                continue


def run_model_enhanced(
    model_path: str,
    prompt: Optional[str] = None,
    interactive: bool = False,
    max_tokens: int = 500,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    stream: bool = True,
    use_chat_template: bool = True,
    hide_reasoning: bool = False,
    verbose: bool = False,
) -> Optional[str]:
    """Enhanced run function with direct MLX integration."""
    try:
        with MLXRunner(model_path, verbose=verbose) as runner:
            if interactive or prompt is None:
                runner.interactive_chat(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    use_chat_template=use_chat_template,
                )
                return None

            if verbose:
                print(f"\nPrompt: {prompt}\n")
                print("Response: ", end="", flush=True)

            if stream:
                response_tokens = []
                try:
                    for token in runner.generate_streaming(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        use_chat_template=use_chat_template,
                        hide_reasoning=hide_reasoning,
                    ):
                        print(token, end="", flush=True)
                        response_tokens.append(token)
                except KeyboardInterrupt:
                    print("\n[INFO] Generation interrupted by user.")
                response = "".join(response_tokens)
            else:
                try:
                    response = runner.generate_batch(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        use_chat_template=use_chat_template,
                    )
                except KeyboardInterrupt:
                    print("\n[INFO] Generation interrupted by user.")
                    response = ""
                print(response)

            return response

    except Exception as e:
        print(f"\n[ERROR] {e}")
        return None


def get_gpu_status() -> Dict[str, float]:
    """Independent GPU status check."""
    try:
        return {
            "active_memory_gb": mx.get_active_memory() / 1024**3,
            "peak_memory_gb": mx.get_peak_memory() / 1024**3,
        }
    except Exception:
        return {"active_memory_gb": 0.0, "peak_memory_gb": 0.0}


def check_memory_available(required_gb: float) -> bool:
    """Pre-flight check before model loading."""
    try:
        current_memory = mx.get_active_memory() / 1024**3
        # Conservative estimate for total memory if not detectable
        estimated_total = 16.0  # Assume 16GB for modern Macs
        available = estimated_total - current_memory - 2.0  # 2GB headroom
        return available >= required_gb
    except Exception:
        return True
