# Llama API Gateway

FastAPI gateway deployed on Render Free with Cloudflare Workers AI for Llama 3.2 inference. The model runs on Cloudflare, not on Render; Render hosts the API gateway.

## Features
- OpenAI-compatible `POST /v1/chat/completions`
- API-key authentication
- Basic per-IP rate limiting
- Provider abstraction: Cloudflare Workers AI now, Ollama later
- Health and model endpoints

See deployment instructions below.

## Deploy
1. Create a Cloudflare account and enable Workers AI.
2. Obtain Account ID and an API token with Workers AI permissions.
3. In Render, create a Blueprint from this repository.
4. Set `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`.
5. Copy the generated `APP_API_KEY` from Render's environment settings.
6. Test `/health`, then `/v1/chat/completions` with `Authorization: Bearer <APP_API_KEY>`.

Cloudflare free usage is limited and can become billable; check current Workers AI pricing and quotas. Render Free services sleep when idle.
