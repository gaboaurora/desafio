"""
Parser de la completion del modelo.

Extrae de forma estricta:
  - Consultas SQL dentro de <sql>...</sql>.
  - La respuesta final dentro de <answer>...</answer>.

Reglas anti reward-hacking (ver tests/test_parser.py):
  - Más de un bloque <answer> => completion inválida (no se premia "varias respuestas a la vez").
  - Bloques <result> escritos por el modelo se ignoran: la observación la produce el harness,
    nunca se confía en resultados "inventados" dentro de la completion.
  - Etiquetas mal cerradas o anidadas => inválida.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ParsedCompletion:
    sql_queries: List[str] = field(default_factory=list)
    final_answer: Optional[str] = None
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)


def parse_completion(completion: str) -> ParsedCompletion:
    """Analiza la completion completa y devuelve sus componentes y su validez."""
    raise NotImplementedError


def extract_sql(completion: str) -> List[str]:
    """Devuelve todas las consultas dentro de <sql>...</sql> en orden de aparición."""
    raise NotImplementedError


def extract_answer(completion: str) -> Optional[str]:
    """Devuelve el contenido de <answer>...</answer> si hay exactamente uno; None en otro caso."""
    raise NotImplementedError
