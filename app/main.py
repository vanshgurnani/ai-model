import os, secrets, time
from collections import defaultdict, deque
from typing import Any
import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from app.providers import get_provider

APP_API_KEY = os.getenv("APP_API_KEY", "")
PUBLIC_MODEL_NAME = os.getenv("PUBLIC_MODEL_NAME", "llama-3.2-3b-instruct")
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "20"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))
app = FastAPI(title="Llama API Gateway", version="1.0.0", description="OpenAI-compatible API with Cloudflare Workers AI and Ollama providers.")
_request_times: dict[str, deque[float]] = defaultdict(deque)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = PUBLIC_MODEL_NAME
    messages: list[ChatMessage] = Field(min_length=1, max_length=40)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=512, ge=1, le=MAX_OUTPUT_TOKENS)
    stream: bool = False
    @field_validator("messages")
    @classmethod
    def validate_messages(cls, messages):
        if any(m.role not in {"system", "user", "assistant"} for m in messages): raise ValueError("Invalid message role")
        if any(len(m.content) > 20000 for m in messages): raise ValueError("Message too long")
        return messages

async def require_api_key(authorization: str | None = Header(default=None)) -> None:
    if not APP_API_KEY: raise HTTPException(status_code=503, detail="Server API key is not configured")
    if not authorization or not secrets.compare_digest(authorization, f"Bearer {APP_API_KEY}"):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

async def rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic(); q = _request_times[ip]
    while q and now - q[0] > 60: q.popleft()
    if len(q) >= MAX_REQUESTS_PER_MINUTE: raise HTTPException(status_code=429, detail="Rate limit exceeded")
    q.append(now)

@app.get("/")
async def root() -> dict[str, Any]:
    return {"name": "Llama API Gateway", "status": "online", "model": PUBLIC_MODEL_NAME, "provider": os.getenv("INFERENCE_PROVIDER", "cloudflare"), "docs": "/docs", "health": "/health"}

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "provider": os.getenv("INFERENCE_PROVIDER", "cloudflare")}

@app.get("/v1/models", dependencies=[Depends(require_api_key)])
async def models() -> dict[str, Any]:
    return {"object": "list", "data": await get_provider().list_models()}

@app.post("/v1/chat/completions", dependencies=[Depends(require_api_key), Depends(rate_limit)])
async def chat_completions(body: ChatCompletionRequest):
    payload = body.model_dump()
    payload["messages"] = [m.model_dump() for m in body.messages]
    provider = get_provider()

    if body.stream:
        if not getattr(provider, "supports_streaming", False):
            raise HTTPException(status_code=501, detail="Streaming is not supported by the configured inference provider")
        return StreamingResponse(
            provider.stream_chat(payload),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try: return await provider.chat(payload)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429: raise HTTPException(status_code=429, detail="Inference provider quota/rate limit reached")
        raise HTTPException(status_code=502, detail=f"Inference provider returned HTTP {exc.response.status_code}")
    except httpx.TimeoutException: raise HTTPException(status_code=504, detail="Inference provider timed out")
    except httpx.HTTPError: raise HTTPException(status_code=502, detail="Could not reach inference provider")
    except KeyError as exc: raise HTTPException(status_code=503, detail=f"Missing provider configuration: {exc.args[0]}")

@app.get("/health/provider", dependencies=[Depends(require_api_key)])
async def provider_health() -> dict[str, str]:
    provider = os.getenv("INFERENCE_PROVIDER", "cloudflare").lower()
    if provider not in {"cloudflare", "ollama"}: raise HTTPException(status_code=500, detail="Unknown provider")
    return {"provider": provider, "configured": "true"}
