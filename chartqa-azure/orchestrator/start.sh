# #!/usr/bin/env bash
# set -euo pipefail
# uvicorn app:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}

# 오케스트레이터 로컬
cd orchestrator
pip install -r requirements.txt
uvicorn app:app --reload --port 8000