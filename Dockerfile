# Glass-Box CT web app. Research use only. NOT built or run in this repository's development environment (no Docker available there);
# the same dependency versions (torch 2.14.1 + CUDA 13.0, MONAI 1.6.1) were validated on an A100 outside a container.
FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime
WORKDIR /app
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt
COPY glassbox ./glassbox
# Weights are not baked in (size, and NVIDIA's license terms): mount them at /models/model.pt. Sample scans and their results go in /data/demo.
ENV GLASSBOX_DEMO_DIR=/data/demo GLASSBOX_WEIGHTS=/models/model.pt GLASSBOX_LIVE=0
EXPOSE 8000
CMD ["uvicorn", "--factory", "glassbox.app.main:build_app", "--host", "0.0.0.0", "--port", "8000"]
