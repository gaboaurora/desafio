"""
Línea base: ejecuta varios modelos (vía Ollama u otro backend) sobre dataset/test.jsonl y
reporta pass@1 y pass@k por nivel de dificultad.

Uso previsto:
    python -m evaluation.run_baseline --models qwen2.5-coder:1.5b-instruct-q4_K_M qwen2.5-coder:7b-instruct \
        --split test --k 8 --out outputs/baseline.json

Guarda además todas las trazas (Episode) en outputs/traces_<modelo>.jsonl para análisis cualitativo
y para alimentar training_data/.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

from harness.environment import Episode, SQLEnvironment, Task


def load_tasks(split: str) -> List[Task]:
    raise NotImplementedError


def run_model_on_tasks(model_name: str, tasks: List[Task], k: int, env: SQLEnvironment) -> List[List[Episode]]:
    """Para cada tarea genera `k` episodios independientes (muestreo con temperatura > 0)."""
    raise NotImplementedError


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Línea base pass@1 / pass@k")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--out", default="outputs/baseline.json")
    args = parser.parse_args(argv)
    raise NotImplementedError


if __name__ == "__main__":
    main()
