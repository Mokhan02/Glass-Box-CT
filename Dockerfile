# Glass-Box CT demo server: React front end (built here) + FastAPI + the five pre-computed demo scans and reports baked in.
# Research use only. The five demo scans (HCC-TACE-Seg, TCIA, CC BY 4.0, resampled) are NOT in git: the build downloads them from a public release asset, verifies the SHA-256 and fails on a mismatch
# (scripts/fetch_demo_data.py; archive and attribution: https://github.com/Mokhan02/Glass-Box-CT). Override the two build args to use another copy.
#   docker build -t glassbox-ct .
#   docker run --rm -p 8000:8000 --env-file .env glassbox-ct      # .env supplies the language model (see docs/TOKEN_FACTORY.md); without it the baked reports are served and Ask is off
# syntax=docker/dockerfile:1

FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt
RUN useradd --create-home --uid 10001 app
COPY LICENSE ./LICENSE
COPY glassbox ./glassbox
COPY --from=web /web/dist ./frontend/dist
ARG DEMO_DATA_URL=https://github.com/Mokhan02/Glass-Box-CT/releases/download/v1/glassbox-demo-data-v1.tar.gz
ARG DEMO_DATA_SHA256=8391cd183116664dad64a79d48d9e2996ce5d4750a8219dca76e54f9d321cdb5
COPY scripts/fetch_demo_data.py /tmp/fetch_demo_data.py
RUN python /tmp/fetch_demo_data.py --url "$DEMO_DATA_URL" --sha256 "$DEMO_DATA_SHA256" --out /app && chown -R app:app /app/demo && rm /tmp/fetch_demo_data.py
ENV GLASSBOX_DEMO_DIR=/app/demo GLASSBOX_FRONTEND_DIST=/app/frontend/dist GLASSBOX_ENV=production
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT','8000'), timeout=4)"
# PORT is honoured because most hosts inject it
CMD ["sh", "-c", "exec uvicorn --factory glassbox.app.main:build_app --host 0.0.0.0 --port ${PORT:-8000}"]
