"""
Generación de pares de preferencia DPO.

Para cada tarea con al menos un episodio correcto (reward 1.0) y uno incorrecto (reward 0.0)
del mismo prompt, se construye {"prompt", "chosen", "rejected"}.

Se priorizan `rejected` informativos: SQL que ejecuta pero responde mal, formato inválido,
respuesta múltiple, etc., etiquetando la categoría del error para análisis posterior.

Uso previsto:
    python -m training_data.generate_dpo --traces outputs/traces_*.jsonl --out outputs/dpo.jsonl
"""

from __future__ import annotations

import argparse
from typing import Dict, List

from harness.environment import Episode


def categorize_failure(episode: Episode) -> str:
    """invalid_format | sql_error | wrong_answer | multiple_answers | no_answer"""
    raise NotImplementedError


def build_pairs(episodes_by_task: Dict[str, List[Episode]], max_pairs_per_task: int = 2) -> List[dict]:
    raise NotImplementedError


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", nargs="+", required=True)
    parser.add_argument("--out", default="outputs/dpo.jsonl")
    args = parser.parse_args(argv)
    raise NotImplementedError


if __name__ == "__main__":
    main()
