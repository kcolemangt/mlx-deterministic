# OpenAI-Compatible API Server

This provides an **OpenAI-compatible API server** with deterministic inference support for MLX models.

## 🎯 What This Does

Creates an API server that:
- ✅ Provides OpenAI-compatible endpoints
- ✅ Supports deterministic inference mode (batch-invariant operations)
- ✅ Uses MLX for Apple Silicon optimization
- ✅ Works with any OpenAI-compatible client

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /Users/joshua/projects/Deterministic
source venv/bin/activate
pip install fastapi uvicorn
```

### 2. Start the Server

```bash
# Basic usage (downloads Qwen2.5-7B-Instruct-4bit)
python examples/openai_compatible_server.py

# Or specify a different model
python examples/openai_compatible_server.py --model mlx-community/Llama-3.2-3B-Instruct-4bit

# Or use a local model path
python examples/openai_compatible_server.py --model /path/to/your/mlx/model

# Custom port
python examples/openai_compatible_server.py --port 8001
```

Server will start on: **http://localhost:8000** (or your specified port)

## 🎛️ API Usage

### Using curl

```bash
# Standard inference
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-deterministic",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 100
  }'

# Deterministic inference (enable batch-invariant ops)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-deterministic",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 100,
    "deterministic": true
  }'
```

### Using Python OpenAI Client

```python
from openai import OpenAI

# Point to your local server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Standard chat
response = client.chat.completions.create(
    model="mlx-deterministic",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(response.choices[0].message.content)

# With deterministic mode
response = client.chat.completions.create(
    model="mlx-deterministic",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    extra_body={"deterministic": True}  # Enable batch-invariant ops
)

print(response.choices[0].message.content)
```

## 🔧 Server Options

```bash
python examples/openai_compatible_server.py --help

Options:
  --host HOST            Host to bind to (default: 0.0.0.0)
  --port PORT            Port to bind to (default: 8000)
  --model MODEL          Model to load (default: Qwen2.5-7B-Instruct-4bit)
```

## 📊 Endpoints

The server provides OpenAI-compatible endpoints:

| Endpoint | Purpose |
|----------|---------|
| `GET /` | Health check |
| `GET /health` | Server status |
| `GET /v1/models` | List available models |
| `POST /v1/chat/completions` | Chat completions (main endpoint) |

## 🎯 Deterministic Mode

When you set `"deterministic": true` in your request:

1. **First request**: Automatically converts the model to use batch-invariant operations
2. **Subsequent requests**: Uses the converted model
3. **Result**: Identical outputs for identical inputs, regardless of batch size!

**Performance overhead**: ~20-35% slower, but reproducible

## 🧪 Testing Determinism

Test that it works:

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="x")

prompt = {"role": "user", "content": "What is 2+2?"}

# Run 10 times - should get identical responses
responses = []
for i in range(10):
    response = client.chat.completions.create(
        model="mlx-deterministic",
        messages=[prompt],
        extra_body={"deterministic": True}
    )
    responses.append(response.choices[0].message.content)

# Check uniqueness
unique = set(responses)
print(f"Unique responses: {len(unique)}/10")
print(f"Deterministic: {'✓ YES' if len(unique) == 1 else '✗ NO'}")
```

## 🔍 Monitoring

Check server status:

```bash
# Health check
curl http://localhost:8000/health

# Response:
{
  "status": "ok",
  "model_loaded": true,
  "deterministic_enabled": false
}
```

## 🐛 Troubleshooting

### "Model not loaded"
- Make sure the model path is correct
- For Hugging Face models, they will download automatically on first use

### "Connection refused"
- Check server is running: `curl http://localhost:8000/health`
- Verify port number
- Check firewall settings

### Slow responses
- First request is slower (model loading)
- Deterministic mode adds ~30% overhead
- Consider using smaller models for testing

## 📝 Supported Models

Any MLX-compatible model works:

```bash
# Qwen (recommended)
python examples/openai_compatible_server.py --model mlx-community/Qwen2.5-7B-Instruct-4bit

# Llama
python examples/openai_compatible_server.py --model mlx-community/Llama-3.2-3B-Instruct-4bit

# Mistral
python examples/openai_compatible_server.py --model mlx-community/Mistral-7B-Instruct-v0.3-4bit

# Local model
python examples/openai_compatible_server.py --model /path/to/your/mlx/model
```

## 🔗 Integration with Other Tools

This server works with any tool that supports OpenAI's API:

- **OpenAI Python SDK** ✅
- **Cursor** (AI code editor) ✅
- **Continue.dev** (VS Code extension) ✅
- **LangChain** (Python framework) ✅
- **Any OpenAI-compatible client** ✅

Just point the base URL to `http://localhost:8000/v1`!

## 📚 More Information

- [Main Integration Guide](../INTEGRATION_GUIDE.md)
- [MLX-LM Documentation](https://github.com/ml-explore/mlx-lm)
