"""
Generación de trazas SFT.

Procedimiento:
  1. Un modelo abierto "profesor" (TEACHER_MODEL_NAME) resuelve las tareas de train.jsonl
     interactuando con SQLEnvironment (varias muestras por tarea).
  2. Se conservan solo los episodios con reward == 1.0.
  3. Se deduplican por (task_id, SQL normalizado) y se balancea por dificultad.
  4. Se exporta en formato chat (messages) a outputs/sft.jsonl.

Uso previsto:
    python -m training_data.generate_sft --model qwen2.5-coder:7b-instruct --samples 4
"""

from __future__ import annotations

import argparse
from typing import List

from harness.environment import Episode


def filter_correct(episodes: List[Episode]) -> List[Episode]:
    raise NotImplementedError


def to_chat_format(episode: Episode) -> dict:
    """{"messages": [{"role": "system"|"user"|"assistant"|"tool", "content": ...}, ...]}"""
    raise NotImplementedError


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--out", default="outputs/sft.jsonl")
    args = parser.parse_args(argv)
    raise NotImplementedError


if __name__ == "__main__":
    main()
