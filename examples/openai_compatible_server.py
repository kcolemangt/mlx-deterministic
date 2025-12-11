#!/usr/bin/env python3
"""
OpenAI-Compatible API Server with Deterministic Inference

This creates an API server compatible with OpenAI's chat completions API format.
Supports deterministic inference using batch-invariant operations.

Usage:
    python examples/openai_compatible_server.py

    # With custom model
    python examples/openai_compatible_server.py --model mlx-community/Qwen3-8B-4bit

    # With local model
    python examples/openai_compatible_server.py --model /path/to/local/model
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from mlx_lm import load, generate
import uvicorn
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mlx_deterministic import enable_mlx_lm_deterministic_mode

app = FastAPI(title="MLX Deterministic Inference API")

# Enable CORS for LM Studio
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
MODEL = None
TOKENIZER = None
DETERMINISTIC_ENABLED = False

# ============================================================================
# OpenAI-Compatible Request/Response Models
# ============================================================================

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.0
    max_tokens: int = 512
    stream: bool = False
    # Custom parameter for deterministic mode
    deterministic: bool = False

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

class ModelInfo(BaseModel):
    id: str
    object: str
    created: int
    owned_by: str

class ModelsResponse(BaseModel):
    object: str
    data: List[ModelInfo]

# ============================================================================
# Model Management
# ============================================================================

def enable_deterministic_mode(model):
    """Enable deterministic inference by monkey-patching MLX-LM attention."""
    global DETERMINISTIC_ENABLED

    if DETERMINISTIC_ENABLED:
        return

    print("🔧 Enabling deterministic mode...")
    # Use the new MLX-LM specific function that patches scaled_dot_product_attention
    enable_mlx_lm_deterministic_mode(split_size=256)
    DETERMINISTIC_ENABLED = True

# ============================================================================
# API Endpoints
# ============================================================================

@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global MODEL, TOKENIZER

    model_name = os.getenv("MODEL_NAME", "mlx-community/Qwen2.5-7B-Instruct-4bit")

    print(f"🚀 Loading model: {model_name}")
    print("   This may take a minute...")

    try:
        MODEL, TOKENIZER = load(model_name)
        print(f"✓ Model loaded successfully!")
        print(f"  Model: {model_name}")
        print(f"  Deterministic: Not enabled (set deterministic=true in request)")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"   Make sure the model exists and is downloaded")
        raise

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "MLX Deterministic Inference API",
        "deterministic_mode": DETERMINISTIC_ENABLED,
        "endpoints": {
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    """Health check for LM Studio."""
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "deterministic_enabled": DETERMINISTIC_ENABLED
    }

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return ModelsResponse(
        object="list",
        data=[
            ModelInfo(
                id="mlx-deterministic",
                object="model",
                created=int(time.time()),
                owned_by="mlx-deterministic"
            )
        ]
    )

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.

    Supports deterministic inference with the 'deterministic' parameter.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Enable deterministic mode if requested
    if request.deterministic and not DETERMINISTIC_ENABLED:
        enable_deterministic_mode(MODEL)

    # Convert messages to prompt
    prompt = ""
    for msg in request.messages:
        if msg.role == "system":
            prompt += f"System: {msg.content}\n"
        elif msg.role == "user":
            prompt += f"User: {msg.content}\n"
        elif msg.role == "assistant":
            prompt += f"Assistant: {msg.content}\n"

    prompt += "Assistant:"

    # Generate response
    try:
        response_text = generate(
            model=MODEL,
            tokenizer=TOKENIZER,
            prompt=prompt,
            max_tokens=request.max_tokens,
            verbose=False
        )

        # Extract just the assistant's response
        if "Assistant:" in response_text:
            response_text = response_text.split("Assistant:")[-1].strip()

        # Create response
        return ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=response_text
                    ),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=len(prompt.split()),  # Rough estimate
                completion_tokens=len(response_text.split()),  # Rough estimate
                total_tokens=len(prompt.split()) + len(response_text.split())
            )
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MLX Deterministic Inference API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model", help="Model to load (default: Qwen2.5-7B-Instruct-4bit)")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic mode on startup")

    args = parser.parse_args()

    if args.model:
        os.environ["MODEL_NAME"] = args.model

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     MLX Deterministic Inference API Server                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\nServer will start on: http://{args.host}:{args.port}")
    print(f"OpenAI API compatible endpoint: /v1/chat/completions")
    print(f"Deterministic mode: {'Enabled on startup' if args.deterministic else 'Use deterministic=true in request'}")
    print(f"\nBase URL for OpenAI-compatible clients:")
    print(f"  http://localhost:{args.port}/v1")
    print("\nStarting server...\n")

    uvicorn.run(app, host=args.host, port=args.port)
