"""
Parser de la completion del modelo.

Protocolo de etiquetas (ver prompt_builder):
    <sql> ... </sql>        llamada a la herramienta (una consulta SELECT)
    <result> ... </result>  observación que escribe el HARNESS (nunca el modelo)
    <answer> ... </answer>  respuesta final (exactamente una)

El parser es deliberadamente estricto, porque sus decisiones son parte de la recompensa:

  - <sql> vacío              => inválida.
  - Más de un bloque <answer>  => inválida ("varias respuestas a la vez").
  - <answer> vacío             => inválida.
  - No hay bloque <answer>     => inválida.
  - No hay bloque <sql>        => inválida.
  - Etiquetas anidadas, sin cerrar o cerradas sin abrir => inválida.
  - Bloques <result> se extraen como NO confiables: el reward re-ejecuta el SQL del modelo y
    jamás usa un <result> presente en la completion (el modelo podría fabricarlo).
  - Texto fuera de etiquetas (razonamiento, "la respuesta es 40") se ignora: solo cuenta lo que
    está dentro de <answer>.

Tolerancias que no afectan la seguridad: nombres de etiqueta en cualquier capitalización,
espacios alrededor del contenido, cercas markdown (```sql ... ```) y un ';' final dentro de <sql>.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

TAGS = ("sql", "answer", "result")
_TAG_RE = re.compile(r"<\s*(/?)\s*(sql|answer|result)\s*>", re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*\n?|\n?\s*```\s*$")


@dataclass
class ParsedCompletion:
    sql_queries: List[str] = field(default_factory=list)      # en orden de aparición, limpias
    final_answer: Optional[str] = None                        # contenido del único <answer>
    result_blocks: List[str] = field(default_factory=list)    # observaciones presentes (NO confiables)
    events: List[Tuple[str, str]] = field(default_factory=list)  # [(kind, content), ...] en orden
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)

    @property
    def has_answer(self) -> bool:
        return self.final_answer is not None

    def invalidate(self, error: str) -> None:
        self.is_valid = False
        if error not in self.errors:
            self.errors.append(error)


def clean_sql(text: str) -> str:
    """Quita cercas markdown y un ';' final. No altera el SQL en sí."""
    s = _FENCE_RE.sub("", text).strip()
    if s.endswith(";"):
        s = s[:-1].rstrip()
    return s


def _tokenize(completion: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    Recorre las etiquetas en orden y devuelve los bloques (kind, content) y los errores
    estructurales detectados (anidamiento, cierre sin apertura, apertura sin cierre).
    """
    events: List[Tuple[str, str]] = []
    errors: List[str] = []
    open_kind: Optional[str] = None
    open_pos = 0
    for m in _TAG_RE.finditer(completion):
        closing, kind = bool(m.group(1)), m.group(2).lower()
        if not closing:
            if open_kind is not None:
                errors.append(f"nested_tag:{kind}_inside_{open_kind}")
                return events, errors
            open_kind, open_pos = kind, m.end()
        else:
            if open_kind is None:
                errors.append(f"closing_without_opening:{kind}")
                return events, errors
            if kind != open_kind:
                errors.append(f"mismatched_closing:{kind}_closes_{open_kind}")
                return events, errors
            events.append((kind, completion[open_pos:m.start()]))
            open_kind = None
    if open_kind is not None:
        errors.append(f"unclosed_tag:{open_kind}")
    return events, errors


def parse_completion(completion: str) -> ParsedCompletion:
    """Analiza la completion completa y devuelve sus componentes y su validez."""
    parsed = ParsedCompletion()
    if not isinstance(completion, str) or not completion.strip():
        parsed.invalidate("empty_completion")
        return parsed

    events, structural_errors = _tokenize(completion)
    parsed.events = events
    for err in structural_errors:
        parsed.invalidate(err)

    answers: List[str] = []
    for kind, content in events:
        if kind == "sql":
            sql = clean_sql(content)
            if not sql:
                parsed.invalidate("empty_sql_block")
            else:
                parsed.sql_queries.append(sql)
        elif kind == "answer":
            answers.append(content.strip())
        else:
            parsed.result_blocks.append(content.strip())

    if len(parsed.sql_queries) == 0:
        parsed.invalidate("no_sql_block")

    if len(answers) > 1:
        parsed.invalidate("multiple_answer_blocks")
    elif len(answers) == 1:
        if not answers[0]:
            parsed.invalidate("empty_answer")
        else:
            parsed.final_answer = answers[0]
    else:
        parsed.invalidate("no_answer_block")
    return parsed


def extract_sql(completion: str) -> List[str]:
    """Consultas dentro de <sql>...</sql> en orden de aparición (vacío si la completion es inválida)."""
    parsed = parse_completion(completion)
    return parsed.sql_queries if parsed.is_valid else []


def extract_answer(completion: str) -> Optional[str]:
    """Contenido de <answer>...</answer> si hay exactamente uno y la completion es válida; None en otro caso."""
    parsed = parse_completion(completion)
    return parsed.final_answer if parsed.is_valid else None
