# Switching the agent to Nebius Token Factory

The agent only ever reads four environment variables (`glassbox/agent/config.py`). Swapping provider is a `.env` change; no code changes. The default (NVIDIA's hosted API, used for development) is **not** changed until you do the steps below.

Facts from the Token Factory docs (checked 2026-10-11): OpenAI-compatible, base URL `https://api.tokenfactory.nebius.com/v1/`, Bearer-token auth, keys from <https://tokenfactory.nebius.com> (Google or GitHub login), models listed by `GET /v1/models`, model IDs carry a publisher namespace (for example `meta-llama/Llama-3.3-70B-Instruct`).

## 1. Get the key and find model IDs
```bash
# in .env (kept out of git; mode 600): add the two lines you know, leave the model IDs for step 2
LLM_BASE_URL=https://api.tokenfactory.nebius.com/v1/
LLM_API_KEY=<your Token Factory key>
python scripts/list_models.py                       # every model the key can use
python scripts/list_models.py --grep nemotron       # is a Nemotron model offered?
```
The docs I could read do not say whether a Nemotron model is hosted, so check the list. Pick:
- `LLM_MODEL`: the **reasoning** model that writes reports and answers questions. Any strong instruction-following model is fine; prefer Nemotron if listed (the project's NVIDIA story).
- `LLM_FAST_MODEL`: the model that **runs the tool-calling steps**. It must support function calling. Test the candidates:
```bash
python scripts/list_models.py --test <model id>      # prints "tool calling WORKS" or fails
```

## 2. The exact `.env` after the swap
```bash
# --- Submission: Nebius Token Factory ---
LLM_BASE_URL=https://api.tokenfactory.nebius.com/v1/
LLM_API_KEY=<Token Factory key>
LLM_MODEL=<reasoning model id from /v1/models>
LLM_FAST_MODEL=<tool-calling model id from /v1/models>

# --- Development: NVIDIA hosted API (keep commented as the rollback) ---
# LLM_BASE_URL=https://integrate.api.nvidia.com/v1
# LLM_API_KEY=<NVIDIA key>
# LLM_MODEL=nvidia/nemotron-3-super-120b-a12b
# LLM_FAST_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
```
The NVIDIA key was pasted into a chat during development: rotate it after the hackathon.

## 3. Verify (each step prints the provider; none prints the key)
```bash
python scripts/agent_smoke.py data/demo demo_2       # one real agent run against a demo scan, number check included
uvicorn --factory glassbox.app.main:build_app --port 8000
#   startup log:  Glass-Box CT: language model provider=api.tokenfactory.nebius.com model=<id> fast_model=<id>; demo_dir=...; live_segmentation=off
curl -s localhost:8000/api/health                   # {"language_model": true, "provider": "api.tokenfactory.nebius.com", "model": ...}
```
The About page also shows the live provider and model. The hackathon requires a runtime Token Factory call in the submitted build: this log line and `/api/health` are the evidence, and every report records `provider_host`.

## 4. Rebuild the cached demo reports with the new provider
Cached reports are keyed on provider, models and prompt, so they are rebuilt for the new provider:
```bash
python scripts/precompute_reports.py data/demo       # builds the five demo reports with Token Factory; check "number check ok" on all five
```
Do this **before** building the Docker image so the baked reports are the Token Factory ones and load instantly.

## 5. Cut over / roll back
Cut over: the lines above. Roll back: comment the Token Factory block and uncomment the NVIDIA block, then restart. If the provider is down at request time the app falls back to the deterministic template (labelled under Evidence) and does not cache that fallback, so it retries the model next time.

## 6. In a container
Pass the same four variables as environment variables to the container or endpoint (never bake `.env` into the image; `.dockerignore` excludes it). See `docs/DEPLOY.md`.
