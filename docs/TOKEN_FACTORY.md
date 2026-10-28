# Nebius Token Factory in Glass-Box CT

The submitted build calls **Nebius Token Factory** for every language-model request, and nothing else. This page says how it is configured, where in the code the calls happen, and how to check it.

Token Factory facts (Nebius docs): OpenAI-compatible, base URL `https://api.tokenfactory.nebius.com/v1/`, Bearer-token auth, models listed by `GET /v1/models`, model IDs carry a publisher namespace.

## Models
| Role | Variable | Model ID |
|---|---|---|
| Reasoning: writes the report, writes Ask answers | `LLM_MODEL` | `nvidia/nemotron-3-super-120b-a12b` |
| Fast: runs the tool-calling steps of the agent and of Ask | `LLM_FAST_MODEL` | `nvidia/Nemotron-3_5-Lightning` (tool calling confirmed with `list_models.py --test`) |

Other Nemotron models the key can use (listed 2026-10-27, all tool-calling tested): `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`, `nvidia/Nemotron-3-Ultra-550b-a55b`. The IDs are not defaulted in code: production refuses to start unless both are set, so the model that ran is always the one that was configured and logged.

## Where the calls happen (all go through one client)
| What | File | Model used |
|---|---|---|
| Client creation, provider rules | `glassbox/agent/config.py` (`resolve_endpoint`, `from_env`, `make_client`) | n/a |
| Tool-calling loop that gathers the measurements | `glassbox/agent/agent.py`, `run_agent`, first `chat.completions.create` | `LLM_FAST_MODEL` |
| Report writing, with number check and up to 3 rewrites | `glassbox/agent/agent.py`, `run_agent`, second `chat.completions.create` | `LLM_MODEL` |
| Ask-a-question tool loop | `glassbox/agent/ask.py`, `ask_events` | `LLM_FAST_MODEL` (answer: `LLM_MODEL`) |
| Server start-up (builds the one client) | `glassbox/app/main.py`, `build_app` | n/a |
| Report cache and background warm-up | `glassbox/app/report.py` | whatever the client above is |

Every report records `provider_host` and the models in its `models` field, and the audit record carries them (`glassbox/trust/audit.py`).

## Production rule: Token Factory only, no fallback
`GLASSBOX_ENV=production` (set in the `Dockerfile`) makes `glassbox/agent/config.py` accept only `https://api.tokenfactory.nebius.com/...`. Any other `LLM_BASE_URL` (NVIDIA's API included), a missing key or a missing model ID stops the server at start-up with the reason; it does not quietly use another provider or run without a model. `make_client` checks the host again at the single place a client is created. `NEBIUS_API_KEY` is only ever sent to the Token Factory host. Tests: `tests/test_provider_config.py`.
If Token Factory is unreachable at request time, the report falls back to the deterministic template (labelled "template" under Evidence, `provider_host` null, not cached) and Ask returns an error. Nothing is sent to another provider.

## Environment
```bash
NEBIUS_API_KEY=<Token Factory key>            # secret; .env is git-ignored, Render holds it as an environment variable
LLM_MODEL=nvidia/nemotron-3-super-120b-a12b
LLM_FAST_MODEL=nvidia/Nemotron-3_5-Lightning
# LLM_BASE_URL is not needed: Token Factory is the default. GLASSBOX_ENV=production is set by the Dockerfile.
```
Local development against another OpenAI-compatible endpoint stays possible: leave `GLASSBOX_ENV` unset and set `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_FAST_MODEL` (see `.env.example`). A production image started with such a `.env` refuses to run, by design.

## Check it
```bash
python scripts/list_models.py --grep nemotron         # models this key can use (never prints the key)
python scripts/list_models.py --test <fast model id>   # "tool calling WORKS"
python scripts/agent_smoke.py data/demo demo_2         # one real agent run, number check included
GLASSBOX_ENV=production python scripts/live_smoke_roles.py data/demo demo_2   # one live call per role, prints model, host and latency
```
At start-up the server logs `Glass-Box CT: language model provider=api.tokenfactory.nebius.com model=<id> fast_model=<id>; ...`; `GET /api/health` returns `{"provider": "api.tokenfactory.nebius.com", "token_factory_only": true, ...}`; the About page shows the same.

## Cached demo reports
Cached reports are keyed on provider, models and prompt. Regenerate them through Token Factory, then rebuild the demo-data archive (`docs/DEPLOY.md`, section 0):
```bash
python scripts/precompute_reports.py data/demo         # all five must print "number check ok"
```
Reports keep the standing liver-report limitation note (`_verbatim_notes` in `glassbox/agent/agent.py` appends it when a model omits it).
