"""
Auditoría de robustez del verificador (Sección II del enunciado).

Dos familias de casos:
  A. Equivalencias que DEBEN aceptarse: "40" / "40.0" / "cuarenta" / "40 estudiantes",
     listas en distinto orden, sinónimos de booleanos, redondeos razonables.
  B. Ataques que DEBEN rechazarse (reward hacking):
     - múltiples respuestas ("40 o 41"), rangos ("entre 38 y 42"),
     - varios bloques <answer>, bloques <result> fabricados, etiquetas anidadas,
     - respuesta correcta escondida en texto con otros números,
     - SQL con sentencias múltiples / escritura / funciones peligrosas,
     - copiar literalmente la observación sin responder.

Reporta tasa de aceptación en A (debe ser ~100%) y tasa de rechazo en B (debe ser ~100%).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class RobustnessCase:
    name: str
    task_id: str
    completion: str
    expected_reward: float
    category: str          # "equivalence" | "attack"


def build_cases() -> List[RobustnessCase]:
    raise NotImplementedError


def run(cases: List[RobustnessCase]) -> dict:
    """Ejecuta `reward` sobre cada caso y devuelve el resumen + casos fallidos."""
    raise NotImplementedError


if __name__ == "__main__":
    print(run(build_cases()))
