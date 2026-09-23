#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
MODEL_DIR="$(cd "$ROOT_DIR/../.." && pwd)/Models"
MISTRAL_MODEL_PATH="${MISTRAL_MODEL_PATH:-$MODEL_DIR/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf}"
LLAMA_URL="http://127.0.0.1:8080"
BACKEND_URL="http://127.0.0.1:8000"
LLAMA_PID=""
BACKEND_PID=""

usage() {
  cat <<'EOF'
Run Autism AI with one command:

  ./run_app.sh mock       Flutter with predictable mock replies
  ./run_app.sh mistral    Start Mistral and connect Flutter directly
  ./run_app.sh backend    Start Mistral, FastAPI, and integrated Flutter

Running ./run_app.sh without an option displays a menu.

The backend option requires the complete verified research corpus. Use mock or
mistral for normal UI testing while that corpus is unavailable.
EOF
}

cleanup() {
  if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  if [[ -n "$LLAMA_PID" ]] && kill -0 "$LLAMA_PID" 2>/dev/null; then
    kill "$LLAMA_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

need_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing command: $1"
    echo "$2"
    exit 1
  fi
}

wait_for_url() {
  local url="$1"
  local label="$2"
  local attempts="${3:-60}"
  local process_pid="${4:-}"

  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if curl --silent --output /dev/null "$url"; then
      return 0
    fi
    if [[ -n "$process_pid" ]] && ! kill -0 "$process_pid" 2>/dev/null; then
      echo "$label stopped before it became ready."
      return 1
    fi
    sleep 1
  done

  echo "$label did not become reachable at $url."
  return 1
}

start_mistral() {
  need_command llama-server "Install llama.cpp first (Homebrew: brew install llama.cpp)."

  if curl --silent --fail --output /dev/null "$LLAMA_URL/health"; then
    echo "Using the Mistral server already running on port 8080."
    return
  fi

  if [[ ! -f "$MISTRAL_MODEL_PATH" ]]; then
    echo "Mistral model not found:"
    echo "  $MISTRAL_MODEL_PATH"
    echo "Set MISTRAL_MODEL_PATH to the GGUF file location and try again."
    exit 1
  fi

  mkdir -p "$BACKEND_DIR/logs"
  echo "Starting Mistral. Its log is backend/logs/llama-server.log"
  llama-server \
    -m "$MISTRAL_MODEL_PATH" \
    --alias mistral \
    --host 127.0.0.1 \
    --port 8080 \
    --ctx-size 8192 \
    >"$BACKEND_DIR/logs/llama-server.log" 2>&1 &
  LLAMA_PID=$!

  if ! wait_for_url "$LLAMA_URL/health" "Mistral" 90 "$LLAMA_PID"; then
    tail -n 30 "$BACKEND_DIR/logs/llama-server.log" || true
    exit 1
  fi
  echo "Mistral is ready."
}

start_backend() {
  local python="$BACKEND_DIR/.venv/bin/python"
  if [[ ! -x "$python" ]]; then
    echo "Backend environment not found at backend/.venv."
    echo "Follow backend/DEVELOPMENT.md setup once, then rerun this command."
    exit 1
  fi

  if curl --silent --output /dev/null "$BACKEND_URL/health"; then
    echo "Using the FastAPI server already running on port 8000."
    return
  fi

  mkdir -p "$BACKEND_DIR/logs"
  echo "Starting FastAPI. Its log is backend/logs/api.log"
  (
    cd "$BACKEND_DIR"
    HF_HOME="$BACKEND_DIR/.cache/huggingface" \
      HF_HUB_OFFLINE=1 \
      TOKENIZERS_PARALLELISM=false \
      exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
  ) >"$BACKEND_DIR/logs/api.log" 2>&1 &
  BACKEND_PID=$!

  if ! wait_for_url "$BACKEND_URL/health" "FastAPI" 90 "$BACKEND_PID"; then
    tail -n 40 "$BACKEND_DIR/logs/api.log" || true
    exit 1
  fi
}

run_flutter() {
  need_command flutter "Install Flutter and ensure it is available on PATH."
  cd "$ROOT_DIR"
  flutter run -d chrome --web-hostname localhost "$@"
}

MODE="${1:-}"
if [[ "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "$MODE" ]]; then
  echo "Choose how to run Autism AI:"
  echo "  1) Mock chat (fastest)"
  echo "  2) Direct Mistral chat"
  echo "  3) Integrated backend (requires complete corpus)"
  read -r -p "Enter 1, 2, or 3: " choice
  case "$choice" in
    1) MODE="mock" ;;
    2) MODE="mistral" ;;
    3) MODE="backend" ;;
    *) echo "Invalid choice."; exit 1 ;;
  esac
fi

case "$MODE" in
  mock)
    echo "Starting Flutter with mock chat."
    run_flutter --dart-define=CHAT_PROVIDER=mock
    ;;
  mistral|local)
    start_mistral
    echo "Starting Flutter with direct Mistral chat."
    run_flutter \
      --dart-define=CHAT_PROVIDER=local \
      --dart-define=LOCAL_LLM_BASE_URL="$LLAMA_URL"
    ;;
  backend)
    start_mistral
    start_backend

    health_file="$(mktemp)"
    health_status="$(curl --silent --output "$health_file" --write-out '%{http_code}' "$BACKEND_URL/health")"
    if [[ "$health_status" != "200" ]]; then
      echo
      echo "The integrated backend is not ready (HTTP $health_status):"
      cat "$health_file"
      echo
      echo "The current known blocker is the incomplete research corpus."
      echo "Run './run_app.sh mistral' for real model chat, or './run_app.sh mock'."
      rm -f "$health_file"
      exit 1
    fi
    rm -f "$health_file"

    echo "Starting Flutter with the integrated backend."
    run_flutter \
      --dart-define=CHAT_PROVIDER=backend \
      --dart-define=AUTISM_AI_BACKEND_URL="$BACKEND_URL" \
      --dart-define=AUTISM_AI_MODEL=mistral
    ;;
  *)
    usage
    exit 1
    ;;
esac
