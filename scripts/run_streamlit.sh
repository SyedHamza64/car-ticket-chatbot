#!/usr/bin/env bash
# Self-healing Streamlit runner - restarts on crash
# Usage: ./scripts/run_streamlit.sh  (or: nohup ./scripts/run_streamlit.sh &)

cd "$(dirname "$0")/.." || exit 1
LOG="${PWD}/.streamlit_crash.log"
PORT="${STREAMLIT_PORT:-8501}"

echo "[$(date -Iseconds)] Starting Streamlit (port $PORT). Log: $LOG" | tee -a "$LOG"
while true; do
  .venv/bin/python -m streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port="$PORT"
  EXIT=$?
  echo "[$(date -Iseconds)] Streamlit exited with code $EXIT. Restarting in 5s..." | tee -a "$LOG"
  sleep 5
done
