"""
Generación de trazas SFT a partir de trazas existentes.

Procedimiento:
  1. Lee los episodios desde los archivos de trazas (por defecto:
     outputs/traces_qwen2.5-coder_1.5b-instruct-q4_K_M.jsonl).
  2. Conserva solo los episodios con reward == 1.0 y respuesta final válida.
  3. Deduplica por (task_id, SQL normalizado).
  4. Exporta en formato chat (messages) a outputs/sft.jsonl.

Uso previsto:
    python -m training_data.generate_sft
    python -m training_data.generate_sft --traces outputs/traces_qwen2.5-coder_1.5b-instruct-q4_K_M.jsonl --out outputs/sft.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from harness.environment import Episode
from harness.parser import clean_sql, parse_completion
from harness.reward import get_task

DEFAULT_TRACES_FILE = "outputs/traces_qwen2.5-coder_1.5b-instruct-q4_K_M.jsonl"

_PROTOCOL_ERROR_MARKERS = (
    "Etiquetas mal formadas",
    "No se detectó bloque sql",
)


def _has_protocol_error(turns: List[Dict[str, str]]) -> bool:
    """Detecta episodios con turnos assistant de formato roto o <result> inventado."""
    for turn in turns:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "assistant" and "<result>" in content.lower():
            return True
        if role == "user" and any(marker in content for marker in _PROTOCOL_ERROR_MARKERS):
            return True
    return False


def filter_correct(episodes: List[Episode]) -> List[Episode]:
    """Filtra episodios con reward == 1.0, sin errores de protocolo y deduplicados por (task_id, SQL)."""
    correct: List[Episode] = []
    seen_keys: Set[str] = set()
    for ep in episodes:
        if (ep.reward or 0.0) < 1.0:
            continue
        if not ep.final_answer:
            continue
        if _has_protocol_error(ep.turns):
            continue
        # Extraemos la consulta SQL del asistente para deduplicar
        assistant_text = "\n".join(
            t["content"] for t in ep.turns if t.get("role") == "assistant"
        )
        parsed = parse_completion(assistant_text)
        sql_signature = clean_sql(parsed.sql_queries[-1]) if parsed.sql_queries else ""

        # Clave única para deduplicar: (task_id, SQL normalizado)
        dedup_key = f"{ep.task_id}::{sql_signature}"
        if dedup_key not in seen_keys:
            seen_keys.add(dedup_key)
            correct.append(ep)
    return correct


def to_chat_format(episode: Episode, difficulty: Optional[str] = None) -> Dict[str, Any]:
    """
    Convierte el Episode al formato estándar:
    {"task_id": str, "difficulty": str, "messages": [{"role": "system"|"user"|"assistant", "content": str}, ...]}
    """
    if difficulty is None:
        task = get_task(episode.task_id)
        difficulty = task.difficulty if task is not None else "unknown"
    return {
        "task_id": episode.task_id,
        "difficulty": difficulty,
        "messages": [
            {"role": t.get("role", "user"), "content": t.get("content", "")}
            for t in episode.turns
        ],
    }


def load_episodes_from_traces(trace_files: List[str]) -> List[Episode]:
    """Carga episodios ya guardados en disco."""
    episodes: List[Episode] = []
    for tf in trace_files:
        path = Path(tf)
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de trazas: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    episodes.append(
                        Episode(
                            task_id=data.get("task_id", ""),
                            prompt=data.get("prompt", ""),
                            turns=data.get("turns", []),
                            final_answer=data.get("final_answer"),
                            reward=data.get("reward"),
                        )
                    )
    return episodes


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generación de dataset SFT exclusivamente a partir de archivos de trazas existentes."
    )
    parser.add_argument(
        "--traces",
        nargs="+",
        default=[DEFAULT_TRACES_FILE],
        help=f"Archivos traces_*.jsonl existentes (por defecto: {DEFAULT_TRACES_FILE})",
    )
    parser.add_argument(
        "--out",
        default="outputs/sft.jsonl",
        help="Ruta del archivo de salida (por defecto: outputs/sft.jsonl)",
    )
    args = parser.parse_args(argv)

    episodes = load_episodes_from_traces(args.traces)
    print(f"Cargadas {len(episodes)} trazas desde: {args.traces}")

    # Filtrar solo los episodios con éxito rotundo (reward == 1.0) y deduplicar
    correct_episodes = filter_correct(episodes)
    print(
        f"Episodios exitosos únicos (SFT): {len(correct_episodes)} de {len(episodes)} totales"
    )

    # Guardar en outputs/sft.jsonl
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    diff_counts: Counter[str] = Counter()
    with open(out_path, "w", encoding="utf-8") as f:
        for ep in correct_episodes:
            record = to_chat_format(ep)
            diff_counts[record["difficulty"]] += 1
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Por dificultad:", dict(diff_counts))
    print(f"¡Dataset SFT exportado exitosamente en: {out_path}!")


if __name__ == "__main__":
    main()
