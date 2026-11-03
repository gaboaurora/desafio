"""
Verificador: compara la respuesta final del modelo con el ground truth.

Debe aceptar formas equivalentes (criterio iii, Robustez):
  - Números: "40", "40.0", "cuarenta", "40 estudiantes", "4,0" vs "4.0", tolerancia relativa
    configurable para promedios (p.ej. redondeo a 2 decimales).
  - Listas / conjuntos: orden irrelevante salvo que la pregunta pida ranking; separadores
    (coma, punto y coma, salto de línea, viñetas) normalizados; mayúsculas y acentos ignorados.
  - Booleanos: "sí"/"no", "true"/"false", "verdadero"/"falso".
  - Tablas pequeñas: comparación por conjunto de filas normalizadas.

Y debe rechazar intentos de engaño:
  - Varias respuestas candidatas ("40 o 41"), rangos, respuestas vacías o "ver resultado".
  - Respuestas que contienen el valor correcto rodeado de texto adicional con otros números.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class VerificationResult:
    is_correct: bool
    normalized_prediction: Any = None
    normalized_target: Any = None
    reason: str = ""


def normalize_number(text: str) -> Optional[float]:
    """Convierte texto a número (dígitos, palabras en español/inglés, separadores locales)."""
    raise NotImplementedError


def normalize_text(text: str) -> str:
    """Minúsculas, sin acentos, sin puntuación redundante, espacios colapsados."""
    raise NotImplementedError


def verify(prediction: str, target: Any, answer_type: str, tolerance: float = 1e-2) -> VerificationResult:
    """Compara `prediction` (texto del modelo) con `target` según `answer_type`."""
    raise NotImplementedError
