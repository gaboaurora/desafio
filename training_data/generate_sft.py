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
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import argparse

from harness.environment import Episode, SQLEnvironment, Task
from harness.parser import clean_sql, parse_completion


def filter_correct(episodes: List[Episode]) -> List[Episode]:
    """Filtra y conserva únicamente los episodios con reward == 1.0 y respuesta válida."""
    correct: List[Episode] = []
    seen_keys: Set[str] = set()
    for ep in episodes:
        if (ep.reward or 0.0) < 1.0:
            continue
        if not ep.final_answer:
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


def to_chat_format(episode: Episode) -> Dict[str, Any]:
    """
    Convierte el Episode al formato estándar:
    {"task_id": str, "messages": [{"role": "system"|"user"|"assistant", "content": str}, ...]}
    """
    return {
        "task_id": episode.task_id,
        "messages": [
            {"role": t.get("role", "user"), "content": t.get("content", "")}
            for t in episode.turns
        ],
    }


def load_episodes_from_traces(trace_files: List[str]) -> List[Episode]:
    """Carga episodios ya guardados en disco por run_baseline o corridas previas."""
    episodes: List[Episode] = []
    for tf in trace_files:
        path = Path(tf)
        if not path.exists():
            continue
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
        description="Generación de trazas SFT a partir de trazas correctas"
    )
    parser.add_argument(
        "--traces", nargs="+", default=[], help="Archivos traces_*.jsonl ya existentes"
    )
    parser.add_argument("--model", default="qwen2.5-coder:1.5b-instruct-q4_K_M", help="Nombre del modelo")
    parser.add_argument(
        "--backend",
        default="ollama",
        choices=["ollama", "oracle"],
        help="Backend a usar",
    )
    parser.add_argument("--split", default="train", help="train|val|test")
    parser.add_argument("--limit", type=int, help="Límite de tareas a procesar")
    parser.add_argument(
        "--out", default="outputs/sft.jsonl", help="Ruta del del archivo de salida"
    )
    args = parser.parse_args(argv)

    episodes: List[Episode] = []

    if args.traces:
        episodes = load_episodes_from_traces(args.traces)
        print(f"Cargadas {len(episodes)} trazas de {args.traces}")

    else:
        from evaluation.run_baseline import (
            OllamaPolicy,
            load_tasks,
            oracle_policy_factory,
        )

        tasks = load_tasks(args.split)
        if args.limit:
            tasks = tasks[: args.limit]
        env = SQLEnvironment()
        print(
            f"Generando trazas para {len(tasks)} tareas con backend '{args.backend}'..."
        )
        for task in tasks:
            if args.backend == "oracle":
                policy = oracle_policy_factory(task, sample=0)
            else:
                model_name = args.model or "qwen2.5-coder:1.5b-instruct-q4_K_M"
                policy = OllamaPolicy(model=model_name, temperature=0.0)
            ep = env.run_episode(task, policy)
            episodes.append(ep)
    # Filtrar solo los episodios con éxito rotundo (reward == 1.0)
    correct_episodes = filter_correct(episodes)
    print(f"Episodios exitosos encontrados: {len(correct_episodes)} de {len(episodes)}")
    # Guardar en outputs/sft.jsonl
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ep in correct_episodes:
            record = to_chat_format(ep)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"¡Dataset SFT exportado exitosamente en: {out_path}!")


if __name__ == "__main__":
    main()
