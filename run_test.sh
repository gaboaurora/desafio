#!/usr/bin/env bash
#SBATCH --job-name=baseline-test
#SBATCH --comment="Linea base pass@8 en test, incluidas plantillas SQL no vistas, para medir la generalizacion antes del entrenamiento"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=outputs/slurm-baseline-test-%j.out
#SBATCH --error=outputs/slurm-baseline-test-%j.err

# Línea base de test (pass@1, pass@2, pass@4 y pass@8) en una H100 asignada por SLURM.
# El repo tiene que estar en un disco visible desde los nodos de cómputo.
#
# En el servidor, desde la raíz del repo:
#   bash run_test.sh
#   sbatch --partition=PARTICION --account=CUENTA run_test.sh
#
# Si sbatch rechaza el GRES, el nombre del tipo de GPU es el del clúster:
#   sinfo -o "%P %G"
# y se cambia la línea `#SBATCH --gres`. Para módulos del sitio:
#   CUDA_MODULE=cuda/12.4 OLLAMA_MODULE=ollama bash run_test.sh
# El .venv de Windows no corre aquí. En el servidor:
#   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  cd "$(dirname "$0")"
  echo "No hay asignación SLURM. Enviando el job a la cola (1 GPU H100)..."
  exec sbatch --chdir="$(pwd)" "$0" "$@"
fi

cd "${SLURM_SUBMIT_DIR}"
if [[ ! -f evaluation/run_baseline.py ]]; then
  echo "El directorio de trabajo (${SLURM_SUBMIT_DIR}) no es la raíz del repo. Envíalo desde ahí." >&2
  exit 1
fi

# Instalación de usuario (sin root). Un módulo cargado después puede ir delante.
export PATH="${HOME}/.local/bin:${PATH}"

if ! type module >/dev/null 2>&1; then
  for _mod in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash; do
    if [[ -f "${_mod}" ]]; then
      # shellcheck disable=SC1090
      source "${_mod}"
      break
    fi
  done
fi
if type module >/dev/null 2>&1; then
  [[ -n "${CUDA_MODULE:-}" ]] && module load "${CUDA_MODULE}"
  [[ -n "${OLLAMA_MODULE:-}" ]] && module load "${OLLAMA_MODULE}"
fi

# La GPU la fija SLURM. Un .env no debe pisar CUDA_VISIBLE_DEVICES.
_cuda_from_slurm="${CUDA_VISIBLE_DEVICES-__unset__}"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source <(sed 's/\r$//' .env)
  set +a
fi
if [[ "${_cuda_from_slurm}" != "__unset__" ]]; then
  export CUDA_VISIBLE_DEVICES="${_cuda_from_slurm}"
elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

if [[ -x .venv/bin/python ]]; then
  PY=.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

MODEL="${TARGET_MODEL_NAME:-qwen2.5-coder:1.5b-instruct-q4_K_M}"
# -1: todas las capas que quepan en la H100.
NUM_GPU="${OLLAMA_NUM_GPU:--1}"
OUT_DIR="outputs/baseline_test"
mkdir -p "${OUT_DIR}"

PORT=$((12000 + SLURM_JOB_ID % 20000))
export OLLAMA_HOST="http://127.0.0.1:${PORT}"
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KEEP_ALIVE=24h
export OLLAMA_NUM_PARALLEL=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

echo "Job ${SLURM_JOB_ID} · nodo $(hostname) · ${SLURM_NODELIST:-} · $(date -Is)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} · OLLAMA_HOST=${OLLAMA_HOST}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi no está en el nodo. Vuelve a enviar con CUDA_MODULE=<módulo cuda>." >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
if ! nvidia-smi -L | grep -qi 'H100'; then
  echo "AVISO: la GPU asignada no se identifica como H100. Revisa #SBATCH --gres y 'sinfo -o \"%P %G\"'." >&2
  nvidia-smi -L >&2
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "No está 'ollama' en el PATH del nodo. Instálalo en \$HOME/.local o define OLLAMA_MODULE." >&2
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "Hace falta curl en el nodo de cómputo." >&2
  exit 1
fi
if ! "${PY}" -c "import ollama, dotenv, tqdm" >/dev/null 2>&1; then
  echo "Faltan dependencias en ${PY}. En el servidor: ${PY} -m pip install -r requirements.txt" >&2
  exit 1
fi

OLLAMA_PID=""
cleanup() {
  if [[ -n "${OLLAMA_PID}" ]] && kill -0 "${OLLAMA_PID}" 2>/dev/null; then
    kill "${OLLAMA_PID}" 2>/dev/null || true
    wait "${OLLAMA_PID}" 2>/dev/null || true
  fi
  OLLAMA_PID=""
}
trap cleanup EXIT
trap 'cleanup; exit 143' INT TERM

echo "Iniciando ollama serve en ${OLLAMA_HOST} (log: ${OUT_DIR}/ollama.log)"
ollama serve >>"${OUT_DIR}/ollama.log" 2>&1 &
OLLAMA_PID=$!

ready=0
for _ in $(seq 1 60); do
  if curl -sf "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "${OLLAMA_PID}" 2>/dev/null; then
    echo "ollama serve terminó al arrancar. Revisa ${OUT_DIR}/ollama.log." >&2
    exit 1
  fi
  sleep 1
done
if [[ "${ready}" -ne 1 ]]; then
  echo "Ollama no respondió en ${OLLAMA_HOST}. Revisa ${OUT_DIR}/ollama.log." >&2
  exit 1
fi

if ! ollama list | awk 'NR>1 {print $1}' | grep -Fxq "${MODEL}"; then
  echo "Descargando ${MODEL} (si el nodo no tiene salida a internet, haz 'ollama pull' en el login)..."
  ollama pull "${MODEL}"
fi

echo "Precargando ${MODEL} en la GPU (num_gpu=${NUM_GPU})..."
warmup="$(curl -sS -w "\n%{http_code}" "${OLLAMA_HOST}/api/chat" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"OK\"}],\"stream\":false,\"options\":{\"num_gpu\":${NUM_GPU},\"num_predict\":8}}")"
code="${warmup##*$'\n'}"
if [[ "${code}" != "200" ]]; then
  echo "No se pudo cargar ${MODEL} (HTTP ${code})." >&2
  echo "${warmup%$'\n'*}" >&2
  exit 1
fi
ps_out="$(ollama ps || true)"
echo "${ps_out}"
if grep -qi '100% CPU' <<<"${ps_out}" || ! grep -qi 'GPU' <<<"${ps_out}"; then
  echo "Ollama no dejó el modelo en la H100." >&2
  nvidia-smi >&2 || true
  exit 1
fi

echo "Test completo · modelo ${MODEL} · k=8 · salida ${OUT_DIR}"
"${PY}" -u -m evaluation.run_baseline \
  --models "${MODEL}" \
  --split test \
  --k 8 \
  --num-gpu "${NUM_GPU}" \
  --host "${OLLAMA_HOST}" \
  --out "${OUT_DIR}/baseline.json" \
  --traces-dir "${OUT_DIR}" \
  --resume \
  "$@"
