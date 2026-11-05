"""
Métricas de evaluación.

pass@k se calcula con el estimador insesgado de Chen et al. (2021):
    pass@k = 1 - C(n - c, k) / C(n, k)
donde n = muestras generadas por tarea y c = muestras correctas.
"""

from __future__ import annotations

import math
from typing import Dict, List


def pass_at_k(n: int, c: int, k: int) -> float:
    """Estimador insesgado de pass@k para una tarea con n muestras y c correctas."""
    # Combinatoria no definida (denominador)
    assert n >= k, f"n ({n}) debe ser mayor o igual a k ({k})"

    # Combinatoria no definida (numerador)
    if n - c < k:
        return 1.0

    prob_fail = math.comb(n - c, k) / math.comb(n, k)
    return 1 - prob_fail


def aggregate_by_difficulty(results: List[Dict], k_values: List[int]) -> Dict[str, Dict[str, float]]:
    """
    results: [{"task_id", "difficulty", "n", "c"}, ...]
    Devuelve {difficulty: {"pass@1": ..., "pass@k": ...}, "all": {...}}.
    """
    aggregated = {}
    for result in results:
        for k in k_values:
            pass_k = pass_at_k(result["n"], result["c"], k)
            difficulty = result["difficulty"]
            if difficulty not in aggregated:
                aggregated[difficulty] = {}
            aggregated[difficulty][f"pass@{k}"] = pass_k

    # Agregar estadísticas para todas las dificultades
    if "all" not in aggregated:
        aggregated["all"] = {}
    for k in k_values:
        aggregated["all"][f"pass@{k}"] = 0.0

    for difficulty, metrics in aggregated.items():
        if difficulty != "all":
            for k in k_values:
                aggregated["all"][f"pass@{k}"] += metrics[f"pass@{k}"]

    for k in k_values:
        aggregated["all"][f"pass@{k}"] /= len(aggregated) - 1  # Restar 1 para no incluir "all"

    return aggregated


def tool_usage_stats(episodes) -> Dict[str, float]:
    """Tasa de SQL ejecutable, tasa de formato válido, promedio de turnos, etc."""
    
    raise NotImplementedError
