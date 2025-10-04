# LM Studio Integration Guide

This guide shows how to use MLX Deterministic Inference as an endpoint for **LM Studio**.

## 🎯 What This Does

Creates an **OpenAI-compatible API server** that:
- ✅ Works with LM Studio's API endpoint feature
- ✅ Supports deterministic inference mode
- ✅ Uses MLX for Apple Silicon optimization
- ✅ Compatible with OpenAI client libraries

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /Users/joshua/projects/Deterministic
source venv/bin/activate
pip install fastapi uvicorn
```

### 2. Start the Server

```bash
# Basic usage (Qwen2.5-7B-Instruct-4bit)
python examples/lm_studio_compatible_server.py

# Or specify a different model
python examples/lm_studio_compatible_server.py --model mlx-community/Llama-3.2-3B-Instruct-4bit

# Enable deterministic mode by default
python examples/lm_studio_compatible_server.py --deterministic
```

Server will start on: **http://localhost:8000**

### 3. Configure LM Studio

In LM Studio:

1. **Settings** → **Developer** → **API Server**
2. Set **Base URL**: `http://localhost:8000/v1`
3. Click **Test Connection**

You should see: ✓ Connected

### 4. Use in LM Studio

Now you can use LM Studio as normal, but it will connect to your MLX deterministic inference server!

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
    "temperature": 0.0,
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
    "temperature": 0.0,
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
    ],
    temperature=0.0
)

print(response.choices[0].message.content)

# With deterministic mode
response = client.chat.completions.create(
    model="mlx-deterministic",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.0,
    extra_body={"deterministic": True}  # Enable batch-invariant ops
)

print(response.choices[0].message.content)
```

## 🔧 Server Options

```bash
python examples/lm_studio_compatible_server.py --help

Options:
  --host HOST            Host to bind to (default: 0.0.0.0)
  --port PORT            Port to bind to (default: 8000)
  --model MODEL          Model to load (default: Qwen2.5-7B-Instruct-4bit)
  --deterministic        Enable deterministic mode on startup
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
        temperature=0.0,
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
- Make sure the model is downloaded first
- Use MLX-LM to download: `mlx_lm.convert --model <model_name>`

### "Connection refused" in LM Studio
- Check server is running: `curl http://localhost:8000/health`
- Verify port is 8000
- Check firewall settings

### Slow responses
- First request is slower (model loading)
- Deterministic mode adds ~30% overhead
- Consider using smaller models for testing

## 📝 Supported Models

Any MLX-compatible model works:

```bash
# Qwen (recommended)
python examples/lm_studio_compatible_server.py --model mlx-community/Qwen2.5-7B-Instruct-4bit

# Llama
python examples/lm_studio_compatible_server.py --model mlx-community/Llama-3.2-3B-Instruct-4bit

# Mistral
python examples/lm_studio_compatible_server.py --model mlx-community/Mistral-7B-Instruct-v0.3-4bit
```

## 🎓 Advanced: Production Deployment

For production use:

```bash
# Use gunicorn for better performance
pip install gunicorn

gunicorn examples.lm_studio_compatible_server:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

## 🔗 Integration with Other Tools

This server works with any tool that supports OpenAI's API:

- **LM Studio** ✅
- **Cursor** (AI code editor)
- **Continue** (VS Code extension)
- **LangChain** (Python framework)
- **Anything using OpenAI client**

Just point the base URL to `http://localhost:8000/v1`!

## 📚 More Information

- [Main Integration Guide](../INTEGRATION_GUIDE.md)
- [Repository](https://github.com/ProbioticFarmer/mlx-deterministic)
- [MLX-LM Documentation](https://github.com/ml-explore/mlx-lm)
