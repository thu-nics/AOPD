#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$SCRIPT_DIR/../common/paths.sh"
VARIANT="${VARIANT:?Set VARIANT to agentic_opd or outcome}"
if [[ "$VARIANT" != "agentic_opd" && "$VARIANT" != "outcome" ]]; then
    echo "ERROR: VARIANT must be agentic_opd or outcome" >&2
    exit 1
fi
DEFAULT_COMPACT_STATE_GROUP_ROWS=false
DEFAULT_PREFER_NONREPEAT_ARGMAX=0
DEFAULT_PROGRESS_INTERVENTION=0
if [[ "$VARIANT" == "agentic_opd" ]]; then
    DEFAULT_COMPACT_STATE_GROUP_ROWS=true
    DEFAULT_PREFER_NONREPEAT_ARGMAX=1
    DEFAULT_PROGRESS_INTERVENTION=1
fi

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the student model directory}"
AWM_BASE_URL="${AWM_BASE_URL:-}"
AWM_HOST="${AWM_HOST:-127.0.0.1}"
AWM_PORT="${AWM_PORT:-}"
MANAGE_AWM_SERVER="${MANAGE_AWM_SERVER:-1}"
AWM_SERVER_START_TIMEOUT="${AWM_SERVER_START_TIMEOUT:-120}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/awm}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/runs/$RUN_STAMP}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-$RUN_DIR/tensorboard}"
TRAIN_SPLIT="${TRAIN_SPLIT:-all}"
VAL_SPLIT="${VAL_SPLIT:-all}"
TRAIN_DATA="${TRAIN_DATA:-}"
TRAIN_SELECTION_MANIFEST="${TRAIN_SELECTION_MANIFEST:-}"
TRAIN_STEPS="${TRAIN_STEPS:-}"
TRAIN_TASK_COUNT="${TRAIN_TASK_COUNT:-}"
TRAIN_TASK_FRACTION="${TRAIN_TASK_FRACTION:-}"
FINAL_POOL_DIR="${FINAL_POOL_DIR:-$REPO_ROOT/runs/awm_data_processing/03_static_feasibility_judge}"
USE_RAW_SPLIT="${USE_RAW_SPLIT:-0}"
TRAIN_BATCH="${TRAIN_BATCH:-8}"
VAL_BATCH="${VAL_BATCH:-8}"
PPO_MINI_BATCH="${PPO_MINI_BATCH:-32}"
PPO_MICRO="${PPO_MICRO:-1}"
LOGPROB_MICRO="${LOGPROB_MICRO:-1}"
PPO_MAX_TOKENS_PER_GPU="${PPO_MAX_TOKENS_PER_GPU:-16384}"
LOGPROB_MAX_TOKENS_PER_GPU="${LOGPROB_MAX_TOKENS_PER_GPU:-32768}"
TP_SIZE="${TP_SIZE:-2}"
SP_SIZE="${SP_SIZE:-2}"
N_GPUS="${N_GPUS:-2}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.65}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32000}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$MAX_MODEL_LEN}"
MAX_HISTORY_EXCHANGES="${MAX_HISTORY_EXCHANGES:-}"
ENABLE_ENVSCALER="${ENABLE_ENVSCALER:-0}"
ENVSCALER_ROOT="${ENVSCALER_ROOT:-$REPO_ROOT/../EnvScaler}"
ENVSCALER_POOL="${ENVSCALER_POOL:-$REPO_ROOT/runs/envscaler_data_processing/02_static_feasibility_judge/envscaler_training_pool.parquet}"
ENVSCALER_MANIFEST="${ENVSCALER_MANIFEST:-$REPO_ROOT/runs/envscaler_data_processing/02_static_feasibility_judge/health_manifest.json}"
AWM_PER_STEP="${AWM_PER_STEP:-58}"
ENVSCALER_PER_STEP="${ENVSCALER_PER_STEP:-6}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-25}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
SMOKE="${SMOKE:-0}"
RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
SHUFFLE="${SHUFFLE:-true}"

MAX_CKPTS="${MAX_CKPTS:-null}"
SAVE_BEFORE_VALIDATION="${SAVE_BEFORE_VALIDATION:-false}"
EXPERT_CACHE_DIR="${EXPERT_CACHE_DIR:-$RUN_DIR/cache}"
RUNTIME_JUDGE_REFERENCE_TRIALS="${RUNTIME_JUDGE_REFERENCE_TRIALS:-}"
RUNTIME_JUDGE_CACHE_PATH="${RUNTIME_JUDGE_CACHE_PATH:-$EXPERT_CACHE_DIR/runtime_judge.jsonl}"
RUNTIME_JUDGE_CONFIDENCE_THRESHOLD="${RUNTIME_JUDGE_CONFIDENCE_THRESHOLD:-80}"
RUNTIME_JUDGE_PROVIDER="${RUNTIME_JUDGE_PROVIDER:-deepseek}"
RUNTIME_JUDGE_MODEL="${RUNTIME_JUDGE_MODEL:-deepseek-v4-flash}"
RUNTIME_JUDGE_API_BASE="${RUNTIME_JUDGE_API_BASE:-https://api.deepseek.com}"
RUNTIME_JUDGE_API_KEY_ENV="${RUNTIME_JUDGE_API_KEY_ENV:-DEEPSEEK_API_KEY}"
RUNTIME_JUDGE_MAX_TOKENS="${RUNTIME_JUDGE_MAX_TOKENS:-8192}"
ORACLE_PROVIDER="${ORACLE_PROVIDER:-deepseek}"
ORACLE_MODEL="${ORACLE_MODEL:-deepseek-v4-flash}"
ORACLE_API_BASE="${ORACLE_API_BASE:-https://api.deepseek.com}"
ORACLE_API_KEY_ENV="${ORACLE_API_KEY_ENV:-DEEPSEEK_API_KEY}"
ORACLE_ENABLE_THINKING="${ORACLE_ENABLE_THINKING:-true}"
ORACLE_REASONING_EFFORT="${ORACLE_REASONING_EFFORT:-max}"
ORACLE_THINKING_BUDGET="${ORACLE_THINKING_BUDGET:-null}"
ORACLE_TEMPERATURE="${ORACLE_TEMPERATURE:-null}"
ORACLE_TOP_P="${ORACLE_TOP_P:-null}"
ORACLE_PRESENCE_PENALTY="${ORACLE_PRESENCE_PENALTY:-null}"
ORACLE_MAX_TOKENS="${ORACLE_MAX_TOKENS:-4096}"
TEACHER_VALIDITY_MAX_RETRIES="${TEACHER_VALIDITY_MAX_RETRIES:-2}"
MATCHER_PROVIDER="${MATCHER_PROVIDER:-deepseek}"
MATCHER_MODEL="${MATCHER_MODEL:-deepseek-v4-flash}"
MATCHER_API_BASE="${MATCHER_API_BASE:-https://api.deepseek.com}"
MATCHER_API_KEY_ENV="${MATCHER_API_KEY_ENV:-DEEPSEEK_API_KEY}"
TEACHER_REWARD_MODE="${TEACHER_REWARD_MODE:-frequency_weighted}"
FREQUENCY_BONUS_SCALE="${FREQUENCY_BONUS_SCALE:-0.5}"
STATE_GROUP_ADVANTAGE_MODE="${STATE_GROUP_ADVANTAGE_MODE:-mean_then_batch_whiten}"
MIN_EFFECTIVE_STATE_GROUPS="${MIN_EFFECTIVE_STATE_GROUPS:-1}"
STATE_GROUP_DIAGNOSTIC_ONLY="${STATE_GROUP_DIAGNOSTIC_ONLY:-0}"
AWM_USE_PRIVILEGED_TEACHER_CONTEXT="${AWM_USE_PRIVILEGED_TEACHER_CONTEXT:-false}"
ENVSCALER_USE_PRIVILEGED_TEACHER_CONTEXT="${ENVSCALER_USE_PRIVILEGED_TEACHER_CONTEXT:-false}"
COMPACT_STATE_GROUP_ROWS="${COMPACT_STATE_GROUP_ROWS:-$DEFAULT_COMPACT_STATE_GROUP_ROWS}"
PREFER_NONREPEAT_ARGMAX="${PREFER_NONREPEAT_ARGMAX:-$DEFAULT_PREFER_NONREPEAT_ARGMAX}"
TEACHER_MULTI_CALL_FALLBACK="${TEACHER_MULTI_CALL_FALLBACK:-$DEFAULT_PROGRESS_INTERVENTION}"
TEACHER_MULTI_CALL_FALLBACK_MIN_STREAK="${TEACHER_MULTI_CALL_FALLBACK_MIN_STREAK:-2}"
REPEAT_REWARD_CAP="${REPEAT_REWARD_CAP:-$DEFAULT_PROGRESS_INTERVENTION}"
REPEAT_REWARD_CAP_MIN_STREAK="${REPEAT_REWARD_CAP_MIN_STREAK:-3}"
REPEAT_REWARD_CAP_VALUE="${REPEAT_REWARD_CAP_VALUE:-0.0}"
REPEAT_TERMINATION="${REPEAT_TERMINATION:-$DEFAULT_PROGRESS_INTERVENTION}"
REPEAT_TERMINATION_MAX_STREAK="${REPEAT_TERMINATION_MAX_STREAK:-4}"
TERMINAL_JUDGE_MODEL="${TERMINAL_JUDGE_MODEL:-deepseek-v4-flash}"
TERMINAL_JUDGE_API_BASE="${TERMINAL_JUDGE_API_BASE:-https://api.deepseek.com}"
TERMINAL_JUDGE_API_KEY_ENV="${TERMINAL_JUDGE_API_KEY_ENV:-DEEPSEEK_API_KEY}"
TERMINAL_JUDGE_REASONING_EFFORT="${TERMINAL_JUDGE_REASONING_EFFORT:-max}"
TERMINAL_JUDGE_MAX_TOKENS="${TERMINAL_JUDGE_MAX_TOKENS:-8192}"
TERMINAL_JUDGE_TIMEOUT_SECONDS="${TERMINAL_JUDGE_TIMEOUT_SECONDS:-300}"
TERMINAL_JUDGE_MAX_RETRIES="${TERMINAL_JUDGE_MAX_RETRIES:-5}"
TAU2_ROOT="${TAU2_ROOT:-$REPO_ROOT/../tau2-bench}"
TAU2_DATA_DIR="${TAU2_DATA_DIR:-$TAU2_ROOT/data}"
TAU_USER_LLM="${TAU_USER_LLM:-}"
TAU_VAL_DOMAINS="${TAU_VAL_DOMAINS:-airline}"
TAU_VAL_TRIALS="${TAU_VAL_TRIALS:-1}"
TAU_VAL_NUM_TASKS="${TAU_VAL_NUM_TASKS:-}"
AWM_SERVER_PID=""

stop_managed_awm_server() {
    if [[ -z "$AWM_SERVER_PID" ]]; then
        return
    fi
    if kill -0 "$AWM_SERVER_PID" 2>/dev/null; then
        kill -TERM -- "-$AWM_SERVER_PID" 2>/dev/null || kill -TERM "$AWM_SERVER_PID" 2>/dev/null || true
        for _ in {1..40}; do
            if ! kill -0 "$AWM_SERVER_PID" 2>/dev/null; then
                break
            fi
            sleep 0.25
        done
        if kill -0 "$AWM_SERVER_PID" 2>/dev/null; then
            kill -KILL -- "-$AWM_SERVER_PID" 2>/dev/null || kill -KILL "$AWM_SERVER_PID" 2>/dev/null || true
        fi
    fi
    wait "$AWM_SERVER_PID" 2>/dev/null || true
    AWM_SERVER_PID=""
}

trap stop_managed_awm_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ ! -x "$PYTHON" || ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: invalid PYTHON=$PYTHON or MODEL_PATH=$MODEL_PATH" >&2
    exit 1
fi
if [[ "$RUN_DIR" == *$'\n'* || "$RUN_DIR" == *$'\r'* || "$RUN_DIR" == *$'\t'* ]]; then
    echo "ERROR: RUN_DIR must not contain newline, carriage-return, or tab characters" >&2
    exit 1
fi
if [[ -z "${!TERMINAL_JUDGE_API_KEY_ENV:-}" ]]; then
    echo "ERROR: $TERMINAL_JUDGE_API_KEY_ENV is required for the AWM terminal SQL+LLM judge" >&2
    exit 1
fi
if [[ "$VARIANT" == "agentic_opd" ]]; then
    if [[ -z "$TAU_USER_LLM" ]]; then
        echo "ERROR: TAU_USER_LLM is required for periodic Tau validation" >&2
        exit 1
    fi
    for api_key_env_var in ORACLE_API_KEY_ENV MATCHER_API_KEY_ENV RUNTIME_JUDGE_API_KEY_ENV; do
        required_api_key_env="${!api_key_env_var}"
        if [[ -z "$required_api_key_env" || -z "${!required_api_key_env:-}" ]]; then
            echo "ERROR: $required_api_key_env is required by $api_key_env_var" >&2
            exit 1
        fi
    done
fi
if [[ "$VARIANT" == "agentic_opd" && "$TAU_USER_LLM" == openrouter/* && -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "ERROR: OPENROUTER_API_KEY is required for TAU_USER_LLM=$TAU_USER_LLM" >&2
    exit 1
fi
if (( N_GPUS % TP_SIZE != 0 || N_GPUS % SP_SIZE != 0 )); then
    echo "ERROR: N_GPUS must be divisible by TP_SIZE and SP_SIZE" >&2
    exit 1
fi
for length_name in MAX_MODEL_LEN MAX_RESPONSE_LENGTH MAX_NUM_BATCHED_TOKENS PPO_MAX_TOKENS_PER_GPU LOGPROB_MAX_TOKENS_PER_GPU; do
    if [[ ! "${!length_name}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: $length_name must be a positive integer" >&2
        exit 1
    fi
done
if [[ -n "$MAX_HISTORY_EXCHANGES" && ! "$MAX_HISTORY_EXCHANGES" =~ ^[0-9]+$ ]]; then
    echo "ERROR: MAX_HISTORY_EXCHANGES must be empty or a non-negative integer" >&2
    exit 1
fi
if [[ ! "$RUNTIME_JUDGE_CONFIDENCE_THRESHOLD" =~ ^[0-9]+$ ]] || (( RUNTIME_JUDGE_CONFIDENCE_THRESHOLD > 100 )); then
    echo "ERROR: RUNTIME_JUDGE_CONFIDENCE_THRESHOLD must be an integer in [0, 100]" >&2
    exit 1
fi
if [[ ! "$RUNTIME_JUDGE_MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || (( RUNTIME_JUDGE_MAX_TOKENS < 8192 )); then
    echo "ERROR: RUNTIME_JUDGE_MAX_TOKENS must be an integer >= 8192" >&2
    exit 1
fi
if [[ "$TEACHER_REWARD_MODE" != "appearance" && "$TEACHER_REWARD_MODE" != "frequency_weighted" ]]; then
    echo "ERROR: TEACHER_REWARD_MODE must be appearance or frequency_weighted" >&2
    exit 1
fi
if [[ "$STATE_GROUP_ADVANTAGE_MODE" != "group_whiten" && "$STATE_GROUP_ADVANTAGE_MODE" != "mean_then_batch_whiten" ]]; then
    echo "ERROR: STATE_GROUP_ADVANTAGE_MODE must be group_whiten or mean_then_batch_whiten" >&2
    exit 1
fi
if [[ ! "$MIN_EFFECTIVE_STATE_GROUPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MIN_EFFECTIVE_STATE_GROUPS must be a positive integer" >&2
    exit 1
fi
case "$STATE_GROUP_DIAGNOSTIC_ONLY" in
    0)
        STATE_GROUP_DIAGNOSTIC_ONLY_HYDRA=false
        ;;
    1)
        STATE_GROUP_DIAGNOSTIC_ONLY_HYDRA=true
        ;;
    *)
        echo "ERROR: STATE_GROUP_DIAGNOSTIC_ONLY must be 0 or 1" >&2
        exit 1
        ;;
esac
for toggle_name in PREFER_NONREPEAT_ARGMAX TEACHER_MULTI_CALL_FALLBACK REPEAT_REWARD_CAP REPEAT_TERMINATION; do
    if [[ "${!toggle_name}" != "0" && "${!toggle_name}" != "1" ]]; then
        echo "ERROR: $toggle_name must be 0 or 1" >&2
        exit 1
    fi
done
for streak_name in TEACHER_MULTI_CALL_FALLBACK_MIN_STREAK REPEAT_REWARD_CAP_MIN_STREAK REPEAT_TERMINATION_MAX_STREAK; do
    if [[ ! "${!streak_name}" =~ ^([2-9]|[1-9][0-9]+)$ ]]; then
        echo "ERROR: $streak_name must be an integer >= 2" >&2
        exit 1
    fi
done
if ! "$PYTHON" -c 'import math, sys; value=float(sys.argv[1]); raise SystemExit(0 if math.isfinite(value) else 1)' "$REPEAT_REWARD_CAP_VALUE"; then
    echo "ERROR: REPEAT_REWARD_CAP_VALUE must be finite" >&2
    exit 1
fi
if [[ "$REPEAT_REWARD_CAP" == "1" && "$REPEAT_TERMINATION" == "1" ]] \
    && (( REPEAT_TERMINATION_MAX_STREAK < REPEAT_REWARD_CAP_MIN_STREAK )); then
    echo "ERROR: REPEAT_TERMINATION_MAX_STREAK must be >= REPEAT_REWARD_CAP_MIN_STREAK" >&2
    exit 1
fi
PREFER_NONREPEAT_ARGMAX_HYDRA=false
TEACHER_MULTI_CALL_FALLBACK_HYDRA=false
REPEAT_REWARD_CAP_HYDRA=false
REPEAT_TERMINATION_HYDRA=false
if [[ "$PREFER_NONREPEAT_ARGMAX" == "1" ]]; then PREFER_NONREPEAT_ARGMAX_HYDRA=true; fi
if [[ "$TEACHER_MULTI_CALL_FALLBACK" == "1" ]]; then TEACHER_MULTI_CALL_FALLBACK_HYDRA=true; fi
if [[ "$REPEAT_REWARD_CAP" == "1" ]]; then REPEAT_REWARD_CAP_HYDRA=true; fi
if [[ "$REPEAT_TERMINATION" == "1" ]]; then REPEAT_TERMINATION_HYDRA=true; fi

if [[ "$VARIANT" == "agentic_opd" ]] && ! "$PYTHON" -c 'import math, sys; value=float(sys.argv[1]); raise SystemExit(0 if math.isfinite(value) and value >= 0 else 1)' "$FREQUENCY_BONUS_SCALE"; then
    echo "ERROR: FREQUENCY_BONUS_SCALE must be finite and non-negative" >&2
    exit 1
fi
if [[ "$RESUME_MODE" == "resume_path" && -z "$RESUME_FROM_PATH" ]]; then
    echo "ERROR: RESUME_FROM_PATH is required when RESUME_MODE=resume_path" >&2
    exit 1
fi
if [[ -n "$RESUME_FROM_PATH" && ! -d "$RESUME_FROM_PATH" ]]; then
    echo "ERROR: RESUME_FROM_PATH not found: $RESUME_FROM_PATH" >&2
    exit 1
fi
if [[ -z "$MAX_PROMPT_LENGTH" ]]; then
    if (( MAX_RESPONSE_LENGTH >= MAX_MODEL_LEN )); then
        echo "ERROR: MAX_RESPONSE_LENGTH must be smaller than MAX_MODEL_LEN" >&2
        exit 1
    fi
    MAX_PROMPT_LENGTH=$((MAX_MODEL_LEN - MAX_RESPONSE_LENGTH))
elif [[ ! "$MAX_PROMPT_LENGTH" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_PROMPT_LENGTH must be a positive integer" >&2
    exit 1
fi
if (( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH > MAX_MODEL_LEN )); then
    echo "ERROR: MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH must not exceed MAX_MODEL_LEN" >&2
    exit 1
fi
for budget_name in PPO_MAX_TOKENS_PER_GPU LOGPROB_MAX_TOKENS_PER_GPU; do
    if (( ${!budget_name} * SP_SIZE < MAX_MODEL_LEN )); then
        echo "ERROR: $budget_name * SP_SIZE must be at least MAX_MODEL_LEN=$MAX_MODEL_LEN" >&2
        exit 1
    fi
done
if [[ "$ENABLE_ENVSCALER" != "0" && "$ENABLE_ENVSCALER" != "1" ]]; then
    echo "ERROR: ENABLE_ENVSCALER must be 0 or 1" >&2
    exit 1
fi
if [[ "$ENABLE_ENVSCALER" == "1" && "$VARIANT" != "agentic_opd" ]]; then
    echo "ERROR: EnvScaler mixing is supported only for agentic OPD training" >&2
    exit 1
fi
if [[ "$ENABLE_ENVSCALER" == "1" ]]; then
    if [[ ! -d "$ENVSCALER_ROOT/.git" ]]; then
        echo "ERROR: EnvScaler checkout not found at $ENVSCALER_ROOT; run examples/envscaler/setup/install_envscaler.sh" >&2
        exit 1
    fi
    if [[ ! "$AWM_PER_STEP" =~ ^[0-9]+$ || ! "$ENVSCALER_PER_STEP" =~ ^[0-9]+$ ]]; then
        echo "ERROR: mixed per-step counts must be non-negative integers" >&2
        exit 1
    fi
    if (( AWM_PER_STEP + ENVSCALER_PER_STEP != TRAIN_BATCH )); then
        echo "ERROR: AWM_PER_STEP + ENVSCALER_PER_STEP must equal TRAIN_BATCH" >&2
        exit 1
    fi
    if [[ ! -f "$ENVSCALER_POOL" || ! -f "$ENVSCALER_MANIFEST" ]]; then
        echo "ERROR: run examples/envscaler/data/run_static_feasibility_judge.sh first" >&2
        exit 1
    fi
    "$PYTHON" -c 'from agent_system.environments.env_package.envscaler.source import validate_envscaler_source; import sys; validate_envscaler_source(sys.argv[1])' "$ENVSCALER_ROOT"
fi
if [[ "$MANAGE_AWM_SERVER" != "0" && "$MANAGE_AWM_SERVER" != "1" ]]; then
    echo "ERROR: MANAGE_AWM_SERVER must be 0 or 1" >&2
    exit 1
fi
if [[ "$MANAGE_AWM_SERVER" == "1" && ! -d "$OPENENV_ROOT/.git" ]]; then
    echo "ERROR: OpenEnv checkout not found at $OPENENV_ROOT; run examples/awm/setup/install_awm.sh" >&2
    exit 1
fi
if [[ ! "$AWM_SERVER_START_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: AWM_SERVER_START_TIMEOUT must be a positive integer" >&2
    exit 1
fi
if ! "$PYTHON" -c 'import agent_world_model_env, openenv' >/dev/null 2>&1; then
    echo "ERROR: AWM dependencies are missing; run examples/awm/setup/install_awm.sh" >&2
    exit 1
fi
if [[ "$VARIANT" == "agentic_opd" ]] && ! TAU2_DATA_DIR="$TAU2_DATA_DIR" \
    "$PYTHON" -c 'import tau2; import rank_bm25' >/dev/null 2>&1; then
    echo "ERROR: Tau dependencies are missing; run examples/tau_bench/install_tau2.sh" >&2
    exit 1
fi

mkdir -p "$RUN_DIR/ckpt" "$EXPERT_CACHE_DIR" "$DATA_DIR"
if [[ "$MANAGE_AWM_SERVER" == "1" ]]; then
    if [[ -n "$AWM_BASE_URL" ]]; then
        echo "ERROR: AWM_BASE_URL cannot be set when MANAGE_AWM_SERVER=1; set AWM_HOST/AWM_PORT or use MANAGE_AWM_SERVER=0" >&2
        exit 1
    fi
    AWM_PORT="$("$PYTHON" - "$AWM_HOST" "$AWM_PORT" <<'PY'
import socket
import sys

host = sys.argv[1]
requested = int(sys.argv[2]) if sys.argv[2] else 0
if requested < 0 or requested > 65535:
    raise SystemExit("AWM_PORT must be between 1 and 65535")
family = socket.AF_INET6 if ":" in host else socket.AF_INET
with socket.socket(family, socket.SOCK_STREAM) as sock:
    sock.bind((host, requested))
    print(sock.getsockname()[1])
PY
)"
    AWM_BASE_URL="http://$AWM_HOST:$AWM_PORT"
    AWM_SERVER_RUN_ID="$(basename "$RUN_DIR")-$$"
    AWM_SERVER_LOG="$RUN_DIR/awm_server.log"
    printf '\n=== managed AWM server run_id=%s base_url=%s ===\n' \
        "$AWM_SERVER_RUN_ID" "$AWM_BASE_URL" >>"$AWM_SERVER_LOG"
    setsid env \
        AWM_HOST="$AWM_HOST" \
        AWM_PORT="$AWM_PORT" \
        AWM_SERVER_RUN_ID="$AWM_SERVER_RUN_ID" \
        AWM_TERMINAL_JUDGE_MODEL="$TERMINAL_JUDGE_MODEL" \
        AWM_TERMINAL_JUDGE_API_BASE="$TERMINAL_JUDGE_API_BASE" \
        AWM_TERMINAL_JUDGE_REASONING_EFFORT="$TERMINAL_JUDGE_REASONING_EFFORT" \
        AWM_TERMINAL_JUDGE_MAX_TOKENS="$TERMINAL_JUDGE_MAX_TOKENS" \
        AWM_TERMINAL_JUDGE_TIMEOUT_SECONDS="$TERMINAL_JUDGE_TIMEOUT_SECONDS" \
        AWM_TERMINAL_JUDGE_MAX_RETRIES="$TERMINAL_JUDGE_MAX_RETRIES" \
        bash "$SCRIPT_DIR/../runtime/start_server.sh" \
        >>"$AWM_SERVER_LOG" 2>&1 &
    AWM_SERVER_PID=$!
    AWM_SERVER_PROTOCOL_JSON=""
    for ((attempt = 1; attempt <= AWM_SERVER_START_TIMEOUT; attempt++)); do
        if ! kill -0 "$AWM_SERVER_PID" 2>/dev/null; then
            echo "ERROR: managed AWM server exited during startup; see $AWM_SERVER_LOG" >&2
            tail -n 80 "$AWM_SERVER_LOG" >&2 || true
            exit 1
        fi
        if AWM_SERVER_PROTOCOL_JSON="$("$PYTHON" "$SCRIPT_DIR/../runtime/check_server.py" \
            --base-url "$AWM_BASE_URL" --data-dir "$AWM_DATA_DIR" \
            --expected-terminal-model "$TERMINAL_JUDGE_MODEL" \
            --expected-run-id "$AWM_SERVER_RUN_ID" --timeout 1 2>/dev/null)"; then
            break
        fi
        AWM_SERVER_PROTOCOL_JSON=""
        sleep 1
    done
    if [[ -z "$AWM_SERVER_PROTOCOL_JSON" ]]; then
        echo "ERROR: managed AWM server did not become healthy within ${AWM_SERVER_START_TIMEOUT}s; see $AWM_SERVER_LOG" >&2
        tail -n 80 "$AWM_SERVER_LOG" >&2 || true
        exit 1
    fi
    AWM_SERVER_PROTOCOL_JSON="$AWM_SERVER_PROTOCOL_JSON" "$PYTHON" - \
        "$RUN_DIR/awm_server_manifest.json" "$AWM_BASE_URL" "$AWM_SERVER_RUN_ID" <<'PY'
import json
import os
from pathlib import Path
import sys

output = Path(sys.argv[1])
server_protocol = json.loads(os.environ["AWM_SERVER_PROTOCOL_JSON"])
manifest = {
    "kind": "awm_managed_run_server",
    "protocol_version": 1,
    "managed": True,
    "base_url": sys.argv[2],
    "run_id": sys.argv[3],
    "log_path": "awm_server.log",
    "logical_time": server_protocol["logical_time"],
    "terminal_judge": server_protocol["terminal_judge"],
}
output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
else
    AWM_BASE_URL="${AWM_BASE_URL:-http://127.0.0.1:8000}"
    if ! "$PYTHON" "$SCRIPT_DIR/../runtime/check_server.py" \
        --base-url "$AWM_BASE_URL" --data-dir "$AWM_DATA_DIR" \
        --expected-terminal-model "$TERMINAL_JUDGE_MODEL" \
        >/dev/null 2>&1; then
        echo "ERROR: external AWM server is not healthy at $AWM_BASE_URL" >&2
        exit 1
    fi
fi
if [[ ! -f "$DATA_DIR/manifest.json" ]]; then
    "$PYTHON" "$SCRIPT_DIR/../data/prepare_data.py" \
        --data-dir "$AWM_DATA_DIR" --output-dir "$DATA_DIR" --local-files-only
fi
if [[ "$SMOKE" == "1" ]]; then
    TRAIN_STEPS=1
    TRAIN_TASK_COUNT="${SMOKE_TRAIN_TASKS:-8}"
    TRAIN_BATCH=4
    VAL_BATCH="${SMOKE_VAL_BATCH:-2}"
    PPO_MINI_BATCH=4
    SAVE_FREQ=-1
    TEST_FREQ=-1
fi

if [[ "$USE_RAW_SPLIT" != "0" && "$USE_RAW_SPLIT" != "1" ]]; then
    echo "ERROR: USE_RAW_SPLIT must be 0 or 1" >&2
    exit 1
fi
if [[ "$VAL_BEFORE_TRAIN" != "true" && "$VAL_BEFORE_TRAIN" != "false" ]]; then
    echo "ERROR: VAL_BEFORE_TRAIN must be true or false" >&2
    exit 1
fi
if [[ -n "$TRAIN_TASK_COUNT" && -n "$TRAIN_TASK_FRACTION" ]]; then
    echo "ERROR: set only one of TRAIN_TASK_COUNT or TRAIN_TASK_FRACTION" >&2
    exit 1
fi
if [[ -z "$TRAIN_DATA" && "$USE_RAW_SPLIT" == "0" ]]; then
    TRAIN_DATA="$FINAL_POOL_DIR/awm_training_pool.parquet"
    TRAIN_SELECTION_MANIFEST="$FINAL_POOL_DIR/health_manifest.json"
fi

"$PYTHON" "$SCRIPT_DIR/../data/prepare_data.py" \
    --data-dir "$AWM_DATA_DIR" \
    --output-dir "$DATA_DIR" \
    --local-files-only \
    --verify-only
if [[ "$VARIANT" == "agentic_opd" ]]; then
    TAU_VAL_DIR="$RUN_DIR/data/tau_validation"
    tau_val_args=(
        --output-dir "$TAU_VAL_DIR"
        --source-root "$TAU2_ROOT"
        --train-steps 1
        --airline 4
        --retail 4
        --validation-domains "$TAU_VAL_DOMAINS"
        --validation-trials "$TAU_VAL_TRIALS"
        --validation-batch-size "$VAL_BATCH"
    )
    if [[ -n "$TAU_VAL_NUM_TASKS" ]]; then
        tau_val_args+=(--validation-num-tasks "$TAU_VAL_NUM_TASKS")
    fi
    TAU2_DATA_DIR="$TAU2_DATA_DIR" "$PYTHON" \
        "$REPO_ROOT/examples/tau_bench/train/prepare_data.py" "${tau_val_args[@]}"
    read -r TAU_VAL_AIRLINE TAU_VAL_RETAIL < <("$PYTHON" -c "import json,sys; c=json.load(open(sys.argv[1]))[\"validation_plan\"][\"counts\"]; print(c[\"airline\"], c[\"retail\"])" "$TAU_VAL_DIR/manifest.json")
    VAL_FILE="$TAU_VAL_DIR/validation.parquet"
else
    VAL_FILE="$DATA_DIR/awm_${VAL_SPLIT}.parquet"
fi
if [[ -n "$TRAIN_DATA" ]]; then
    if [[ -z "$TRAIN_SELECTION_MANIFEST" ]]; then
        echo "ERROR: TRAIN_SELECTION_MANIFEST is required with TRAIN_DATA" >&2
        exit 1
    fi
    if [[ ! -f "$TRAIN_DATA" || ! -f "$TRAIN_SELECTION_MANIFEST" ]]; then
        echo "ERROR: healthy training pool is missing; run examples/awm/data/run_static_feasibility_judge.sh first" >&2
        exit 1
    fi
    "$PYTHON" "$SCRIPT_DIR/../data/verify_training_pool.py" \
        --data "$TRAIN_DATA" \
        --manifest "$TRAIN_SELECTION_MANIFEST"
    TRAIN_FILE="$TRAIN_DATA"
    if [[ -n "$TRAIN_TASK_COUNT" || -n "$TRAIN_TASK_FRACTION" ]]; then
        SLICE_DATA="$RUN_DIR/data/awm_training_pool_slice.parquet"
        SLICE_MANIFEST="$RUN_DIR/data/training_slice_manifest.json"
        slice_args=()
        if [[ -n "$TRAIN_TASK_COUNT" ]]; then
            slice_args+=(--task-count "$TRAIN_TASK_COUNT")
        else
            slice_args+=(--fraction "$TRAIN_TASK_FRACTION")
        fi
        "$PYTHON" "$SCRIPT_DIR/../data/slice_training_pool.py" \
            --data "$TRAIN_DATA" \
            --manifest "$TRAIN_SELECTION_MANIFEST" \
            --output-data "$SLICE_DATA" \
            --output-manifest "$SLICE_MANIFEST" \
            "${slice_args[@]}"
        TRAIN_FILE="$SLICE_DATA"
        TRAIN_SELECTION_MANIFEST="$SLICE_MANIFEST"
    fi
else
    if [[ -n "$TRAIN_TASK_COUNT" || -n "$TRAIN_TASK_FRACTION" ]]; then
        echo "ERROR: training-task slicing requires a verified healthy pool" >&2
        exit 1
    fi
    TRAIN_FILE="$DATA_DIR/awm_${TRAIN_SPLIT}.parquet"
fi
for path in "$TRAIN_FILE" "$VAL_FILE"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: prepared training/validation data does not exist: $path" >&2
        exit 1
    fi
done
SOURCE_TASK_COUNT="$("$PYTHON" -c 'import pandas as pd, sys; print(len(pd.read_parquet(sys.argv[1])))' "$TRAIN_FILE")"
SOURCE_STEPS_PER_EPOCH=$((SOURCE_TASK_COUNT / TRAIN_BATCH))
if (( SOURCE_STEPS_PER_EPOCH <= 0 )); then
    echo "ERROR: training pool must contain at least TRAIN_BATCH=$TRAIN_BATCH tasks" >&2
    exit 1
fi
if [[ -z "$TRAIN_STEPS" ]]; then
    TRAIN_STEPS="$SOURCE_STEPS_PER_EPOCH"
fi
if (( TRAIN_STEPS <= 0 )); then
    echo "ERROR: TRAIN_STEPS must be positive" >&2
    exit 1
fi
if [[ "$VARIANT" == "agentic_opd" && "$SMOKE" != "1" ]]; then
    if [[ -z "$TRAIN_SELECTION_MANIFEST" ]]; then
        echo "ERROR: formal agentic OPD training requires a verified healthy pool" >&2
        exit 1
    fi
    SCHEDULE_DATA="$RUN_DIR/data/awm_training_schedule.parquet"
    SCHEDULE_MANIFEST="$RUN_DIR/data/training_schedule_manifest.json"
    if [[ "$ENABLE_ENVSCALER" == "1" ]]; then
        SCHEDULE_DATA="$RUN_DIR/data/awm_envscaler_training_schedule.parquet"
        "$PYTHON" "$REPO_ROOT/examples/envscaler/data/materialize_mixed_schedule.py" \
            --awm-data "$TRAIN_FILE" \
            --envscaler-data "$ENVSCALER_POOL" \
            --envscaler-manifest "$ENVSCALER_MANIFEST" \
            --output-data "$SCHEDULE_DATA" \
            --output-manifest "$SCHEDULE_MANIFEST" \
            --train-steps "$TRAIN_STEPS" \
            --awm-per-step "$AWM_PER_STEP" \
            --envscaler-per-step "$ENVSCALER_PER_STEP"
    else
        "$PYTHON" "$SCRIPT_DIR/../data/materialize_training_schedule.py" \
            --data "$TRAIN_FILE" \
            --manifest "$TRAIN_SELECTION_MANIFEST" \
            --output-data "$SCHEDULE_DATA" \
            --output-manifest "$SCHEDULE_MANIFEST" \
            --train-steps "$TRAIN_STEPS" \
            --train-batch-size "$TRAIN_BATCH"
    fi
    TRAIN_FILE="$SCHEDULE_DATA"
fi
TASK_COUNT="$("$PYTHON" -c 'import pandas as pd, sys; print(len(pd.read_parquet(sys.argv[1])))' "$TRAIN_FILE")"
if (( TASK_COUNT % TRAIN_BATCH != 0 )); then
    echo "WARNING: each epoch drops $((TASK_COUNT % TRAIN_BATCH)) tail task(s)" >&2
fi
STEPS_PER_EPOCH=$((TASK_COUNT / TRAIN_BATCH))
TRAIN_EPOCHS=$(((TRAIN_STEPS + STEPS_PER_EPOCH - 1) / STEPS_PER_EPOCH))

export AWM_DATA_DIR TAU2_DATA_DIR TENSORBOARD_DIR TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
CONFIG_NAME="awm_${VARIANT}"
if [[ "$ENABLE_ENVSCALER" == "1" ]]; then
    CONFIG_NAME=awm_envscaler_agentic_opd
fi
LOGGER='["console","tensorboard"]'
if [[ "$SMOKE" == "1" ]]; then LOGGER='["console"]'; fi
MIXED_OVERRIDES=()
if [[ "$ENABLE_ENVSCALER" == "1" ]]; then
    MIXED_OVERRIDES=(
        "env.envscaler.source_root=$ENVSCALER_ROOT"
        "env.envscaler.oracle.use_privileged_context=$ENVSCALER_USE_PRIVILEGED_TEACHER_CONTEXT"
        "env.agentic_mix.trajectory_counts.awm=$AWM_PER_STEP"
        "env.agentic_mix.trajectory_counts.envscaler=$ENVSCALER_PER_STEP"
    )
fi
VALIDATION_OVERRIDES=()
AGENTIC_OPD_REWARD_OVERRIDES=()
if [[ "$VARIANT" == "agentic_opd" ]]; then
    AGENTIC_OPD_REWARD_OVERRIDES=(
        "env.teacher_reward.mode=$TEACHER_REWARD_MODE"
        "env.teacher_reward.frequency_bonus_scale=$FREQUENCY_BONUS_SCALE"
        "env.rollout.prefer_nonrepeat_argmax=$PREFER_NONREPEAT_ARGMAX_HYDRA"
        "env.rollout.teacher_multi_call_fallback.enabled=$TEACHER_MULTI_CALL_FALLBACK_HYDRA"
        "env.rollout.teacher_multi_call_fallback.min_repeat_streak=$TEACHER_MULTI_CALL_FALLBACK_MIN_STREAK"
        "env.rollout.repeat_reward_cap.enabled=$REPEAT_REWARD_CAP_HYDRA"
        "env.rollout.repeat_reward_cap.min_streak=$REPEAT_REWARD_CAP_MIN_STREAK"
        "env.rollout.repeat_reward_cap.value=$REPEAT_REWARD_CAP_VALUE"
        "env.rollout.repeat_termination.enabled=$REPEAT_TERMINATION_HYDRA"
        "env.rollout.repeat_termination.max_streak=$REPEAT_TERMINATION_MAX_STREAK"
    )
    VALIDATION_OVERRIDES=(
        "env.validation.env_name=tau"
        "env.tau.source_root=$TAU2_ROOT"
        "env.tau.user_llm=$TAU_USER_LLM"
        "env.tau.validation_domains=[$TAU_VAL_DOMAINS]"
        "env.tau.validation_trials=$TAU_VAL_TRIALS"
        "env.tau.validation_counts.airline=$TAU_VAL_AIRLINE"
        "env.tau.validation_counts.retail=$TAU_VAL_RETAIL"
        "env.awm.runtime_failures.path=$RUN_DIR/runtime_failures.jsonl"
        "env.awm.runtime_failures.judge.data_dir=$AWM_DATA_DIR"
        "env.awm.runtime_failures.judge.reference_trials_path=$RUNTIME_JUDGE_REFERENCE_TRIALS"
        "env.awm.runtime_failures.judge.cache_path=$RUNTIME_JUDGE_CACHE_PATH"
        "env.awm.runtime_failures.judge.confidence_threshold=$RUNTIME_JUDGE_CONFIDENCE_THRESHOLD"
        "env.awm.runtime_failures.judge.provider=$RUNTIME_JUDGE_PROVIDER"
        "env.awm.runtime_failures.judge.model=$RUNTIME_JUDGE_MODEL"
        "env.awm.runtime_failures.judge.api_base=$RUNTIME_JUDGE_API_BASE"
        "env.awm.runtime_failures.judge.api_key_env=$RUNTIME_JUDGE_API_KEY_ENV"
        "env.awm.runtime_failures.judge.max_tokens=$RUNTIME_JUDGE_MAX_TOKENS"
    )
fi


echo "AWM $VARIANT run: $RUN_DIR"
if [[ "$ENABLE_ENVSCALER" == "1" ]]; then
    echo "Agentic mix per step: AWM=$AWM_PER_STEP EnvScaler=$ENVSCALER_PER_STEP"
fi
echo "Training split tasks=$TASK_COUNT batch=$TRAIN_BATCH steps=$TRAIN_STEPS epochs=$TRAIN_EPOCHS"
echo "Context budget prompt=$MAX_PROMPT_LENGTH response=$MAX_RESPONSE_LENGTH model=$MAX_MODEL_LEN batched=$MAX_NUM_BATCHED_TOKENS"
if [[ "$VARIANT" == "agentic_opd" ]]; then
    echo "Teacher reward mode=$TEACHER_REWARD_MODE frequency bonus scale=$FREQUENCY_BONUS_SCALE"
    echo "Teacher provider=$ORACLE_PROVIDER model=$ORACLE_MODEL matcher=$MATCHER_PROVIDER/$MATCHER_MODEL"
fi
"$PYTHON" -m verl.trainer.main_ppo \
    --config-name "$CONFIG_NAME" \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size="$TRAIN_BATCH" \
    data.val_batch_size="$VAL_BATCH" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.truncation=error \
    data.return_raw_chat=True \
    data.shuffle="$SHUFFLE" \
    +data.dataloader_num_workers=0 \
    data.apply_chat_template_kwargs.enable_thinking=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKENS_PER_GPU" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="$SP_SIZE" \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$LOGPROB_MAX_TOKENS_PER_GPU" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$LOGPROB_MAX_TOKENS_PER_GPU" \
    env.awm.base_url="$AWM_BASE_URL" \
    env.context.history_policy=token_budget \
    env.context.max_history_exchanges="${MAX_HISTORY_EXCHANGES:-null}" \
    env.awm.verifier_mode=sql \
    env.awm.terminal_judge.model="$TERMINAL_JUDGE_MODEL" \
    env.awm.terminal_judge.api_base="$TERMINAL_JUDGE_API_BASE" \
    env.awm.terminal_judge.api_key_env="$TERMINAL_JUDGE_API_KEY_ENV" \
    env.awm.terminal_judge.reasoning_effort="$TERMINAL_JUDGE_REASONING_EFFORT" \
    env.awm.terminal_judge.max_tokens="$TERMINAL_JUDGE_MAX_TOKENS" \
    env.awm.terminal_judge.timeout_seconds="$TERMINAL_JUDGE_TIMEOUT_SECONDS" \
    env.awm.terminal_judge.max_retries="$TERMINAL_JUDGE_MAX_RETRIES" \
    env.awm.oracle.provider="$ORACLE_PROVIDER" \
    env.awm.oracle.model="$ORACLE_MODEL" \
    env.awm.oracle.api_base="$ORACLE_API_BASE" \
    env.awm.oracle.api_key_env="$ORACLE_API_KEY_ENV" \
    env.awm.oracle.enable_thinking="$ORACLE_ENABLE_THINKING" \
    env.awm.oracle.reasoning_effort="$ORACLE_REASONING_EFFORT" \
    env.awm.oracle.thinking_budget="$ORACLE_THINKING_BUDGET" \
    env.awm.oracle.temperature="$ORACLE_TEMPERATURE" \
    env.awm.oracle.top_p="$ORACLE_TOP_P" \
    env.awm.oracle.presence_penalty="$ORACLE_PRESENCE_PENALTY" \
    env.awm.oracle.max_tokens="$ORACLE_MAX_TOKENS" \
    env.awm.oracle.teacher_validity_max_retries="$TEACHER_VALIDITY_MAX_RETRIES" \
    env.awm.oracle.matcher_provider="$MATCHER_PROVIDER" \
    env.awm.oracle.matcher_model="$MATCHER_MODEL" \
    env.awm.oracle.matcher_api_base="$MATCHER_API_BASE" \
    env.awm.oracle.matcher_api_key_env="$MATCHER_API_KEY_ENV" \
    env.awm.oracle.cache_path="$EXPERT_CACHE_DIR/teacher.jsonl" \
    env.awm.oracle.use_privileged_context="$AWM_USE_PRIVILEGED_TEACHER_CONTEXT" \
    algorithm.state_group.advantage_mode="$STATE_GROUP_ADVANTAGE_MODE" \
    algorithm.state_group.min_effective_groups="$MIN_EFFECTIVE_STATE_GROUPS" \
    algorithm.state_group.compact_policy_rows="$COMPACT_STATE_GROUP_ROWS" \
    algorithm.state_group.diagnostic_only="$STATE_GROUP_DIAGNOSTIC_ONLY_HYDRA" \
    env.awm.oracle.matcher_cache_path="$EXPERT_CACHE_DIR/matcher.jsonl" \
    env.rollout.n=4 \
    "${MIXED_OVERRIDES[@]}" \
    "${AGENTIC_OPD_REWARD_OVERRIDES[@]}" \
    "${VALIDATION_OVERRIDES[@]}" \
    trainer.total_training_steps="$TRAIN_STEPS" \
    trainer.total_epochs="$TRAIN_EPOCHS" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.val_before_train="$VAL_BEFORE_TRAIN" \
    trainer.save_before_validation="$SAVE_BEFORE_VALIDATION" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.balance_batch=False \
    trainer.project_name=awm \
    trainer.experiment_name="$(basename "$RUN_DIR")" \
    trainer.default_local_dir="$RUN_DIR/ckpt" \
    trainer.max_actor_ckpt_to_keep="$MAX_CKPTS" \
    trainer.max_critic_ckpt_to_keep="$MAX_CKPTS" \
    trainer.logger="$LOGGER" \
    trainer.resume_mode="$RESUME_MODE" \
    trainer.resume_from_path="${RESUME_FROM_PATH:-null}" \
    hydra.run.dir="$RUN_DIR/hydra" \
    "$@" 2>&1 | tee "$RUN_DIR/train.log"
