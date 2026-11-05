from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


from harness.environment import Episode
from harness.parser import parse_completion
from harness.sql_executor import SQLExecutor


def pass_at_k(n: int, c: int, k: int) -> float:
    """Estimador insesgado de pass@k para una tarea con n muestras y c correctas."""
    assert n >= k >= 1, f"Se requiere n ({n}) >= k ({k}) >= 1"
    assert 0 <= c <= n, f"c ({c}) debe estar entre 0 y n ({n})"

    if n - c < k:
        return 1.0

    prob_all_fail = math.comb(n - c, k) / math.comb(n, k)
    return 1.0 - prob_all_fail


def aggregate_by_difficulty(
    results: List[Dict], k_values: List[int]
) -> Dict[str, Dict[str, float]]:
    """
    results: [{"task_id": str, "difficulty": str, "n": int, "c": int}, ...]
    Devuelve {difficulty: {"pass@1": ..., "pass@k": ...}, "all": {...}}.
    """
    if not results:
        return {"all": {f"pass@{k}": 0.0 for k in k_values}}

    # Agrupar tareas por dificultad
    by_diff: Dict[str, List[Dict]] = defaultdict(list)
    for item in results:
        by_diff[item["difficulty"]].append(item)

    aggregated: Dict[str, Dict[str, float]] = {}

    # Promedio de pass@k dentro de cada dificultad
    for difficulty, items in by_diff.items():
        aggregated[difficulty] = {
            f"pass@{k}": sum(pass_at_k(r["n"], r["c"], k) for r in items) / len(items)
            for k in k_values
        }

    # Promedio global sobre todas las tareas ("all")
    aggregated["all"] = {
        f"pass@{k}": sum(pass_at_k(r["n"], r["c"], k) for r in results) / len(results)
        for k in k_values
    }

    return aggregated


def tool_usage_stats(episodes: Iterable[Episode]) -> Dict[str, float]:
    """
    - valid_format_rate: proporción de episodios con etiquetas válidas y <answer> presente.
    - sql_emitted_rate: proporción de episodios donde el modelo emitió al menos un <sql>.
    - executable_sql_rate: proporción de consultas <sql> emitidas que ejecutan sin error en SQLite.
    - avg_turns: promedio de turnos del asistente por episodio.
    - reward_rate: promedio de reward (si está calculado en el episodio).
    """
    ep_list = list(episodes)
    if not ep_list:
        return {
            "valid_format_rate": 0.0,
            "sql_emitted_rate": 0.0,
            "executable_sql_rate": 0.0,
            "avg_turns": 0.0,
            "reward_rate": 0.0,
        }

    executor = SQLExecutor()
    valid_format_count = 0
    sql_emitted_episodes = 0
    total_sql_queries = 0
    executable_sql_queries = 0
    total_assistant_turns = 0
    rewards: List[float] = []

    for ep in ep_list:
        assistant_msgs = [t["content"] for t in ep.turns if t.get("role") == "assistant"]
        total_assistant_turns += len(assistant_msgs)

        # Unimos únicamente los turnos emitidos por el asistente para analizar su completion
        combined_assistant = "\n".join(assistant_msgs)
        parsed = parse_completion(combined_assistant)

        if parsed.is_valid and parsed.has_answer:
            valid_format_count += 1

        if parsed.sql_queries:
            sql_emitted_episodes += 1
            for q in parsed.sql_queries:
                total_sql_queries += 1
                if executor.execute(q).ok:
                    executable_sql_queries += 1

        if ep.reward is not None:
            rewards.append(float(ep.reward))

    n_eps = len(ep_list)
    return {
        "valid_format_rate": valid_format_count / n_eps,
        "sql_emitted_rate": sql_emitted_episodes / n_eps,
        "executable_sql_rate": (
            executable_sql_queries / total_sql_queries if total_sql_queries > 0 else 0.0
        ),
        "avg_turns": total_assistant_turns / n_eps,
        "reward_rate": sum(rewards) / len(rewards) if rewards else 0.0,
    }


if __name__ == "__main__":
    assert pass_at_k(n=10, c=7, k=1) == 0.7

    results = [
        {"task_id": "1", "difficulty": "easy", "n": 10, "c": 7},
        {"task_id": "2", "difficulty": "easy", "n": 10, "c": 5},
        {"task_id": "3", "difficulty": "medium", "n": 10, "c": 3},
        {"task_id": "4", "difficulty": "hard", "n": 10, "c": 1},
    ]
    k_values = [1, 5]
    aggregated = aggregate_by_difficulty(results, k_values)
    assert set(aggregated) == {"easy", "medium", "hard", "all"}

    episodes = [
        Episode(
            task_id="dummy-1",
            prompt="Pregunta dummy",
            turns=[
                {
                    "role": "assistant",
                    "content": "<sql>SELECT 1</sql>",
                },
                {
                    "role": "assistant",
                    "content": "<answer>1</answer>",
                },
            ],
            reward=1.0,
        ),
        Episode(
            task_id="dummy-2",
            prompt="Pregunta dummy",
            turns=[
                {
                    "role": "assistant",
                    "content": "<sql>UPDATE patients SET name = 'x'</sql>",
                },
                {
                    "role": "assistant",
                    "content": "<answer>sin resultado</answer>",
                },
            ],
            reward=0.0,
        ),
    ]
    usage = tool_usage_stats(episodes)
    assert usage["valid_format_rate"] == 1.0
    assert usage["sql_emitted_rate"] == 1.0
    assert usage["avg_turns"] == 2.0
    assert usage["reward_rate"] == 0.5

    print("pass_at_k:", pass_at_k(n=10, c=7, k=1))
    print("aggregate_by_difficulty:", aggregated)
    print("tool_usage_stats:", usage)
    print("Test metrics checks: OK")