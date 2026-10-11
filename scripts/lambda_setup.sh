#!/usr/bin/env bash
# One-time setup on the Lambda instance. Run from the repo root on the instance:
#   bash scripts/lambda_setup.sh /path/to/filesystem/glassbox
# Everything big (venv, weights, data) lives under that path so it survives instance termination.
set -euo pipefail
WORK="${1:?usage: lambda_setup.sh <persistent-dir>}"
mkdir -p "$WORK"/{cache/weights,cache/ts,outputs}
[ -e cache ] || ln -s "$WORK/cache" cache          # repo paths use ./cache
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h "$WORK" | tail -1
python3 -m pip install --user -q uv
export PATH="$HOME/.local/bin:$PATH"
[ -d "$WORK/venv" ] || uv venv --python 3.11 "$WORK/venv"
uv pip install --python "$WORK/venv/bin/python" -q \
  "monai[nibabel,einops,tqdm]" torch numpy scikit-learn nibabel remotezip pytest
[ -s cache/weights/model.pt ] || curl -sSfL -o cache/weights/model.pt \
  https://huggingface.co/nvidia/NV-Segment-CT/resolve/main/vista3d_pretrained_model/model.pt
[ -s cache/bundle_metadata.json ] || curl -sSfL -o cache/bundle_metadata.json \
  https://raw.githubusercontent.com/Project-MONAI/model-zoo/dev/models/vista3d/configs/metadata.json
"$WORK/venv/bin/python" scripts/fetch_ts.py --dest cache/ts --ids s0235
"$WORK/venv/bin/python" -c "import torch;print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "setup ok. Next: $WORK/venv/bin/python scripts/gate_check.py s0235 cuda"
