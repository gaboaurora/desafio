"""
Métricas de evaluación.

pass@k se calcula con el estimador insesgado de Chen et al. (2021):
    pass@k = 1 - C(n - c, k) / C(n, k)
donde n = muestras generadas por tarea y c = muestras correctas.
"""

from __future__ import annotations

from typing import Dict, List


def pass_at_k(n: int, c: int, k: int) -> float:
    """Estimador insesgado de pass@k para una tarea con n muestras y c correctas."""
    raise NotImplementedError


def aggregate_by_difficulty(results: List[Dict], k_values: List[int]) -> Dict[str, Dict[str, float]]:
    """
    results: [{"task_id", "difficulty", "n", "c"}, ...]
    Devuelve {difficulty: {"pass@1": ..., "pass@k": ...}, "all": {...}}.
    """
    raise NotImplementedError


def tool_usage_stats(episodes) -> Dict[str, float]:
    """Tasa de SQL ejecutable, tasa de formato válido, promedio de turnos, etc."""
    raise NotImplementedError
