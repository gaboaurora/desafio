#!/usr/bin/env bash
#SBATCH --job-name=sft-test
#SBATCH --comment="Evaluacion SFT LoRA pass@8 en test (602 tareas, incluidas plantillas heldout)"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=outputs/slurm-sft-test-%j.out
#SBATCH --error=outputs/slurm-sft-test-%j.err

# Evaluación de un modelo HF + adaptador LoRA (por defecto SFT en test, k=8) en una H100 de Peteroa.
# Usa los mismos parámetros SLURM que run_test.sh y llama a evaluation/run_hf.py.
#
# En Peteroa, desde la raíz del repo:
#   bash run_sft_test.sh
#
# Para evaluar otro checkpoint o split sin duplicar script:
#   MODEL_NAME=dpo_lora ADAPTER_DIR=checkpoints/dpo_lora bash run_sft_test.sh
#   SPLIT=eval bash run_sft_test.sh

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  cd "$(dirname "$0")"
  echo "No hay asignación SLURM. Enviando el job a la cola (1 GPU H100)..."
  exec sbatch --chdir="$(pwd)" "$0" "$@"
fi

cd "${SLURM_SUBMIT_DIR}"
if [[ ! -f evaluation/run_hf.py ]]; then
  echo "El directorio de trabajo (${SLURM_SUBMIT_DIR}) no es la raíz del repo. Envíalo desde ahí." >&2
  exit 1
fi

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

MODEL_NAME="${MODEL_NAME:-sft_lora}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
ADAPTER_DIR="${ADAPTER_DIR:-checkpoints/sft_lora}"
SPLIT="${SPLIT:-test}"
K="${K:-8}"
OUT_DIR="${OUT_DIR:-outputs/${MODEL_NAME}_${SPLIT}}"
mkdir -p "${OUT_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

echo "Job ${SLURM_JOB_ID} · nodo $(hostname) · ${SLURM_NODELIST:-} · $(date -Is)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} · MODEL_NAME=${MODEL_NAME} · SPLIT=${SPLIT} · K=${K}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi no está en el nodo. Vuelve a enviar con CUDA_MODULE=<módulo cuda>." >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

if ! "${PY}" -c "import torch, transformers, peft, dotenv, tqdm" >/dev/null 2>&1; then
  echo "Faltan dependencias en ${PY}. Instala en Peteroa:" >&2
  echo "  ${PY} -m pip install -r requirements.txt torch transformers peft accelerate safetensors" >&2
  exit 1
fi

ADAPTER_ARGS=()
if [[ -n "${ADAPTER_DIR}" ]]; then
  if [[ ! -d "${ADAPTER_DIR}" ]]; then
    echo "No existe ADAPTER_DIR='${ADAPTER_DIR}'. Copia tu carpeta del adaptador LoRA al servidor." >&2
    exit 1
  fi
  ADAPTER_ARGS=(--adapter "${ADAPTER_DIR}")
fi

echo "Evaluación HF · modelo ${MODEL_NAME} · base ${BASE_MODEL} · adapter ${ADAPTER_DIR:-ninguno} · salida ${OUT_DIR}"
"${PY}" -u -m evaluation.run_hf \
  --name "${MODEL_NAME}" \
  --base-model "${BASE_MODEL}" \
  ${ADAPTER_ARGS[@]+"${ADAPTER_ARGS[@]}"} \
  --split "${SPLIT}" \
  --k "${K}" \
  --out "${OUT_DIR}/${MODEL_NAME}.json" \
  --traces-dir "${OUT_DIR}" \
  --resume \
  "$@"
