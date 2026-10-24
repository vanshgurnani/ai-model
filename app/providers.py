import os
from abc import ABC, abstractmethod
from typing import Any
import httpx

class InferenceProvider(ABC):
    @abstractmethod
    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    @abstractmethod
    async def list_models(self) -> list[dict[str, Any]]: ...

class CloudflareWorkersAIProvider(InferenceProvider):
    def __init__(self) -> None:
        self.account_id = os.environ["CLOUDFLARE_ACCOUNT_ID"]
        self.api_token = os.environ["CLOUDFLARE_API_TOKEN"]
        self.model = os.getenv("CLOUDFLARE_MODEL", "@cf/meta/llama-3.2-3b-instruct")
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/v1"

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = dict(payload)
        body["model"] = self.model
        async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=15)) as client:
            response = await client.post(f"{self.base_url}/chat/completions", headers={"Authorization": f"Bearer {self.api_token}"}, json=body)
            response.raise_for_status()
            result = response.json()
        result["model"] = self.model
        return result

    async def list_models(self) -> list[dict[str, Any]]:
        return [{"id": self.model, "object": "model", "owned_by": "cloudflare-workers-ai"}]

class OllamaProvider(InferenceProvider):
    def __init__(self) -> None:
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = {"model": self.model, "messages": payload.get("messages", []), "stream": False,
                "options": {"temperature": payload.get("temperature", 0.7), "num_predict": payload.get("max_tokens", 512)}}
        async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=10)) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=body)
            response.raise_for_status()
            result = response.json()
        msg = result.get("message", {})
        prompt = result.get("prompt_eval_count", 0)
        completion = result.get("eval_count", 0)
        return {"id": "ollama-chat-completion", "object": "chat.completion", "created": 0, "model": self.model,
                "choices": [{"index": 0, "message": {"role": msg.get("role", "assistant"), "content": msg.get("content", "")}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}}

    async def list_models(self) -> list[dict[str, Any]]:
        return [{"id": self.model, "object": "model", "owned_by": "ollama"}]

def get_provider() -> InferenceProvider:
    name = os.getenv("INFERENCE_PROVIDER", "cloudflare").lower()
    if name == "cloudflare": return CloudflareWorkersAIProvider()
    if name == "ollama": return OllamaProvider()
    raise RuntimeError(f"Unsupported INFERENCE_PROVIDER: {name}")
