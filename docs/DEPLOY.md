# Deploying Glass-Box CT

The image (`Dockerfile`, about 360 MB) is the React front end + FastAPI + the five demo scans and their reports baked in. It needs **no GPU and no PyTorch**. It was built and run (keyless) on a Lambda instance; it has not been run on Nebius or on a fallback host, so every cloud-specific step below is from the providers' docs and unverified until you run it. I did not create any account or deploy anything.

## 0. Before you build (same for every host)
1. Set the final language-model provider in `.env` (`docs/TOKEN_FACTORY.md`), then bake the reports with it:
   `python scripts/precompute_reports.py data/demo` and check that all five say `number check ok`. The cached reports are keyed on provider, models and prompt.
2. Build from a checkout that has `data/demo` (it is git-ignored; the five scans are CC BY 4.0 HCC-TACE-Seg, attribution is in the app and README):
   ```bash
   docker build -t glassbox-ct .
   docker run --rm -p 8000:8000 --env-file .env glassbox-ct       # open http://localhost:8000
   python scripts/smoke_deployed.py http://localhost:8000          # all PASS
   ```
   `.dockerignore` is an allow-list: `.env`, `.git`, `results/`, caches and `node_modules` never enter the image. Secrets are passed as environment variables at run time, never baked.
3. Run-time variables (all optional; without them the baked reports are served and "Ask" is switched off): `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_FAST_MODEL`. `PORT` is honoured. `GLASSBOX_WARM_REPORTS=0` skips the background cache check at start-up.
4. The server logs one line at start: `Glass-Box CT: language model provider=<host> model=<id> ...`. Use it as the runtime evidence of the Token Factory call.

## A. Nebius AI Cloud (primary)
Source: Nebius docs, Serverless AI endpoints quickstart and Container Registry quickstart (fetched 2026-10-11). Serverless AI runs a container image behind a managed HTTPS URL; billing is per second while the endpoint exists, so **delete it when you are not demoing**.

```bash
# one-time: Docker, the Nebius CLI and jq installed; then
nebius profile create                                   # log in; pick the project
export REGION_ID=<your project's region id>

# 1. registry and push
export REGISTRY_PATH=$(nebius registry create --name glassbox --format json | jq -r ".metadata.id" | cut -d- -f 2)
nebius registry configure-helper                        # lets docker push without docker login
docker tag glassbox-ct cr.$REGION_ID.nebius.cloud/$REGISTRY_PATH/glassbox-ct:v1
docker push cr.$REGION_ID.nebius.cloud/$REGISTRY_PATH/glassbox-ct:v1      # run docker as your own user, not root

# 2. endpoint (CPU platform and preset are the ones the quickstart uses)
nebius ai endpoint create \
  --name glassbox-ct \
  --image cr.$REGION_ID.nebius.cloud/$REGISTRY_PATH/glassbox-ct:v1 \
  --platform cpu-d3 --preset 4vcpu-16gb \
  --container-port 8000 \
  --public \
  --subnet-id <subnet id> \
  --auth token --token "$AUTH_TOKEN"        # see the two caveats below
# environment variables for the language model: run `nebius ai endpoint create --help` for the exact flag name (not shown in the quickstart) and pass
# LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_FAST_MODEL

# 3. URL, test, logs, delete
export ENDPOINT_ID=$(nebius ai endpoint get-by-name --name glassbox-ct --format jsonpath='{.metadata.id}')
export ENDPOINT_URL=$(nebius ai endpoint get "$ENDPOINT_ID" --format json | jq -r '.status.public_endpoints[] | select(startswith("https://"))' | head -1)
python scripts/smoke_deployed.py "$ENDPOINT_URL" --token "$AUTH_TOKEN"
nebius ai endpoint logs $ENDPOINT_ID                    # the start-up line shows provider and model
nebius ai endpoint delete $ENDPOINT_ID                  # stops billing
```
Caveats to resolve when you run it:
- **Public demo link.** The quickstart protects the endpoint with a Bearer token, which a browser visiting the URL cannot send, so judges could not open it. Check `--help` for a no-auth option for a public demo, or put the page behind another front door. I could not confirm one from the docs.
- **Secrets.** Pass the Token Factory key through whatever secret/env mechanism the endpoint offers; do not put it in the image or the command history you share.
- **GPU variant (optional).** `Dockerfile.gpu` adds live segmentation of uploaded scans on a GPU endpoint (weights mounted at `/models/model.pt`). Not built or run here; GPU platform and preset names were not in the docs I read.

## B. Fallback host (any container host)
The image works anywhere that runs a Linux container and injects `PORT`. Two options; neither tried.
- **Fly.io:** `fly launch --no-deploy` (accept the Dockerfile, internal port 8000), `fly secrets set LLM_API_KEY=... LLM_BASE_URL=... LLM_MODEL=... LLM_FAST_MODEL=...`, `fly deploy`. Use at least 512 MB RAM. Then `python scripts/smoke_deployed.py https://<app>.fly.dev`.
- **Render:** New → Web Service → "Existing image" (push to `ghcr.io/<user>/glassbox-ct` first: `docker tag ... && docker push ...`) or "Build from repo" is **not** possible because `data/demo` is not in git; add the same four environment variables in the dashboard.
- Pushing the image to a public registry publishes the five scans (CC BY 4.0, attribution included) and the baked reports; it does not publish keys or the repo history.

## C. Keeping a key safe on a public page
- The question box is rate-limited in the server (6 per minute, 120 per hour, 300 characters) so a public page cannot run up the bill; lower `ASK_PER_MINUTE` / `ASK_PER_HOUR` in `glassbox/app/server.py` for a tighter limit.
- Use a Token Factory key restricted to this project if the console offers one, and revoke it after the hackathon.
