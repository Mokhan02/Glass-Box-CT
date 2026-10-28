# Deploying Glass-Box CT

The image (`Dockerfile`, about 360 MB) is the React front end + FastAPI + the five demo scans and their reports baked in. It needs **no GPU and no PyTorch**. It was built and run (keyless) on a Lambda instance; it has not been run on Nebius or on a fallback host, so every cloud-specific step below is from the providers' docs and unverified until you run it. I did not create any account or deploy anything.

## 0. Demo data and the build (same for every host)
**The five demo scans are not in git.** The Docker build downloads them from a public release asset, checks the SHA-256 and **fails the build on a mismatch** (`scripts/fetch_demo_data.py`, run by the `RUN` line in the `Dockerfile`; unpacked to `/app/demo`, owned by the non-root `app` user). Archive: `glassbox-demo-data-v2.tar.gz`, 56402527 bytes, SHA-256 `dccf71fab9465f4dadecfb32bace4cc776f4f3bb8c486e281b42053b3665fd21`. The two build arguments and their defaults:
- `DEMO_DATA_URL` = `https://github.com/Mokhan02/Glass-Box-CT/releases/download/v2/glassbox-demo-data-v2.tar.gz`
- `DEMO_DATA_SHA256` = `dccf71fab9465f4dadecfb32bace4cc776f4f3bb8c486e281b42053b3665fd21`
The URL must be reachable **without credentials** from the build machine, so this repository must be **public** when the release is published (release assets of a private repository need a token). The data is a release asset of this repository, not a file in git. Until the release below is published the build stops with `HTTP 404 (has the release been published, and is the repository public?)`.

### Publishing the data (one time)
The archive is `release/glassbox-demo-data-v2.tar.gz` on your Mac (git-ignored; keep a copy). **Upload that exact file**: rebuilding the tarball changes its hash, and then the SHA in the `Dockerfile` and here must change with it.
1. Before making the repository public: `gitleaks detect` on the full history must report no leaks, and `git ls-files` must not list `.env` or anything under `release/`.
2. GitHub → `Mokhan02/Glass-Box-CT` → Settings → General → Danger Zone → "Change visibility" → Public.
3. Releases → "Create a new release" → "Choose a tag": type `v2` and select "Create new tag: v2 on publish" → Release title `Demo data v2` → description: paste the contents of `docs/demo-data-README.md` (it carries the CC BY 4.0 attribution and the SHA-256) → drag in `release/glassbox-demo-data-v2.tar.gz` (GitHub allows 2 GB per asset; this is about 56 MB) → leave pre-release and draft unticked → **Publish release**.
4. Verify from any machine (no login):
   ```bash
   curl -L -o /tmp/d.tgz https://github.com/Mokhan02/Glass-Box-CT/releases/download/v2/glassbox-demo-data-v2.tar.gz
   shasum -a 256 /tmp/d.tgz        # must print dccf71fab9465f4dadecfb32bace4cc776f4f3bb8c486e281b42053b3665fd21
   ```
With the GitHub CLI instead (`brew install gh`, `gh auth login`): `gh repo edit Mokhan02/Glass-Box-CT --visibility public --accept-visibility-change-consequences`, then `gh release create v2 release/glassbox-demo-data-v2.tar.gz --repo Mokhan02/Glass-Box-CT --title "Demo data v2" --notes-file docs/demo-data-README.md`.
To use another copy (a new version, a mirror): `docker build --build-arg DEMO_DATA_URL=... --build-arg DEMO_DATA_SHA256=... .`

## 0b. Build and run
```bash
docker build -t glassbox-ct .                                   # downloads and verifies the demo data
docker run --rm -p 8000:8000 --env-file .env glassbox-ct       # http://localhost:8000
python scripts/smoke_deployed.py http://localhost:8000          # all PASS
```
`.dockerignore` is an allow-list: `.env`, `.git`, `results/`, caches and `node_modules` never enter the build context. Secrets are passed as environment variables at run time, never baked.
1. The cached reports in archive `v2` were written through Nebius Token Factory with `LLM_MODEL=nvidia/nemotron-3-super-120b-a12b` and `LLM_FAST_MODEL=nvidia/Nemotron-3_5-Lightning`. A report is reused only when provider, both model IDs and the prompt match, so **set exactly these two IDs on the server**; any other value makes it regenerate the reports on first use (about 35 s each). To change models: set them in `.env`, run `GLASSBOX_ENV=production python scripts/precompute_reports.py data/demo`, rebuild the archive as the next version, publish it, and update the two build arguments.
2. Run-time variables: `NEBIUS_API_KEY`, `LLM_MODEL`, `LLM_FAST_MODEL`. The image sets `GLASSBOX_ENV=production`, so it **requires** them and accepts Nebius Token Factory only; it stops at start-up with the reason otherwise (docs/TOKEN_FACTORY.md). For a keyless local look, run with `-e GLASSBOX_ENV=development`. **`PORT` is honoured**: the container runs `uvicorn ... --host 0.0.0.0 --port ${PORT:-8000}` (checked with `PORT=9123` and with `PORT` unset). `GLASSBOX_WARM_REPORTS=0` skips the background cache check at start-up.
3. The server logs one line at start: `Glass-Box CT: language model provider=<host> model=<id> ...`. Use it as the runtime evidence of the Token Factory call.

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
# NEBIUS_API_KEY, LLM_MODEL, LLM_FAST_MODEL

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
- **Fly.io:** `fly launch --no-deploy` (accept the Dockerfile, internal port 8000), `fly secrets set NEBIUS_API_KEY=... LLM_MODEL=... LLM_FAST_MODEL=...`, `fly deploy`. Use at least 512 MB RAM. Then `python scripts/smoke_deployed.py https://<app>.fly.dev`.
- **Render (build from the repository):** New → Web Service → connect `Mokhan02/Glass-Box-CT` → Language: Docker (it finds the `Dockerfile`). Render injects `PORT` (the server listens on it) and passes environment variables to the Docker build as build arguments, so the demo-data URL and SHA-256 need no setting unless you want to override them. Set `NEBIUS_API_KEY`, `LLM_MODEL`, `LLM_FAST_MODEL` as environment variables (runtime secrets; never commit them). Health check path: `/api/health`. The build fails with a clear message if the data release is not published yet or the checksum differs; publish the release (section 0), then "Manual Deploy" → "Deploy latest commit". Memory: the image is about 500 MB; a 512 MB instance may be tight, try 1 GB if the service restarts (not tested on Render).
- Pushing the image to a public registry publishes the five scans (CC BY 4.0, attribution included) and the baked reports; it does not publish keys or the repo history.

## C. Keeping a key safe on a public page
- The question box is rate-limited in the server (6 per minute, 120 per hour, 300 characters) so a public page cannot run up the bill; lower `ASK_PER_MINUTE` / `ASK_PER_HOUR` in `glassbox/app/server.py` for a tighter limit.
- Use a Token Factory key restricted to this project if the console offers one, and revoke it after the hackathon.
