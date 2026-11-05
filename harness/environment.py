"""
Entorno de interacción entre el modelo y la base de datos.

Ciclo de pocos turnos (recomendación del enunciado):

    1. El harness entrega el prompt (esquema + pregunta).
    2. El modelo emite una llamada a la herramienta:  <sql> SELECT ... </sql>
    3. El harness ejecuta la consulta y devuelve la observación: <result> ... </result>
    4. El modelo emite la respuesta final:  <answer> ... </answer>

`MAX_TOOL_TURNS` (por defecto 2) acota cuántas llamadas SQL se permiten por episodio.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Task:
    """Una tarea del dataset (una línea de dataset/*.jsonl)."""

    id: str                         # p.ej. "hosp_0421"
    database: str                   # archivo relativo a db/, p.ej. "hospital.sqlite"
    difficulty: str                 # easy | medium | hard | expert
    template_id: str                # plantilla de la que se instanció (control de fugas)
    features: List[str]             # construcciones SQL: join, group_by, having, subquery, ...
    question: str                   # pregunta en lenguaje natural (español)
    gold_sql: str                   # consulta canónica; expected_result = ejecutar gold_sql
    answer_type: str                # number | text | boolean | list | table
    ordered: bool                   # si el orden de list/table importa (rankings)
    result_columns: List[str]       # nombres de las columnas de la respuesta
    expected_result: Any            # valor tipado según answer_type
    template_heldout: bool = False  # True si la plantilla solo aparece en test

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Task":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class Episode:
    """Traza completa de una interacción, útil para SFT/DPO y auditoría."""

    task_id: str
    prompt: str
    turns: List[Dict[str, str]] = field(default_factory=list)   # [{"role": ..., "content": ...}]
    final_answer: Optional[str] = None
    reward: Optional[float] = None


class SQLEnvironment:
    """Orquesta la interacción modelo <-> herramienta SQL para una tarea."""

    def __init__(self, max_tool_turns: Optional[int] = None) -> None:
        self.max_tool_turns = max_tool_turns or int(os.getenv("MAX_TOOL_TURNS", "2"))

    def reset(self, task: Task) -> str:
        """Inicia un episodio para `task` y devuelve el prompt inicial."""
        raise NotImplementedError

    def step(self, completion: str) -> Tuple[str, bool]:
        """
        Procesa una completion del modelo.

        Returns:
            (observation, done): si la completion contiene <sql>, `observation` es el
            resultado de la consulta y `done=False`; si contiene <answer>, `done=True`.
        """
        raise NotImplementedError

    def run_episode(self, task: Task, policy: Callable[[List[Dict[str, str]]], str]) -> Episode:
        """Ejecuta el ciclo completo usando `policy(messages) -> completion`."""
        raise NotImplementedError
