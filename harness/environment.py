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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from generator.schema_generator import build_schema, to_prompt_description
from harness.parser import parse_completion
from harness.prompt_builder import (
    build_prompt,
    build_system_prompt,
    build_tool_observation,
    build_user_prompt,
)
from harness.sql_executor import SQLExecutor

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
    completion: str = ""
    turns: List[Dict[str, str]] = field(default_factory=list)   # [{"role": ..., "content": ...}]
    final_answer: Optional[str] = None
    reward: Optional[float] = None


class SQLEnvironment:
    """Orquesta la interacción modelo <-> herramienta SQL para una tarea."""

    def __init__(
        self,
        max_tool_turns: Optional[int] = None,
        schema_description: Optional[str] = None,
        db_dir: str = "db",
    ) -> None:
        self.max_tool_turns = max_tool_turns or int(os.getenv("MAX_TOOL_TURNS", "2"))
        self.db_dir = Path(db_dir)
        self.schema_description = schema_description or to_prompt_description(build_schema())

        self.current_task: Optional[Task] = None
        self.tool_turns_used: int = 0
        self.messages: List[Dict[str, str]] = []
        self.last_final_answer: Optional[str] = None

    def reset(self, task: Task) -> str:
        """Inicia un episodio para `task` y devuelve el prompt inicial."""
        self.current_task = task
        self.tool_turns_used = 0
        self.last_final_answer = None

        system_content = build_system_prompt(self.schema_description)
        user_content = build_user_prompt(task.id, task.question)

        self.messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        return build_prompt(task.id, task.question, self.schema_description)

    def step(self, completion: str) -> Tuple[str, bool]:
        """
        Procesa una completion del modelo.

        Returns:
            (observation, done): si la completion contiene <sql>, `observation` es el
            resultado de la consulta y `done=False`; si contiene <answer>, `done=True`.
        """
        if self.current_task is None:
            raise RuntimeError("Debe llamar a reset(task) antes de step().")

        self.messages.append({"role": "assistant", "content": completion})
        parsed = parse_completion(completion)

        # 1. Si emitió una respuesta final, el episodio termina
        if parsed.has_answer:
            self.last_final_answer = parsed.final_answer
            return ("", True)

        # 2. Si emitió consulta SQL y aún tiene turnos permitidos
        if parsed.sql_queries:
            self.tool_turns_used += 1
            query = parsed.sql_queries[-1]

            db_path = self.db_dir / self.current_task.database
            executor = SQLExecutor(db_path=str(db_path))
            exec_result = executor.execute(query)
            formatted = executor.format_result(exec_result)

            observation = build_tool_observation(formatted, error=exec_result.error)
            self.messages.append({"role": "user", "content": observation})

            done = self.tool_turns_used >= self.max_tool_turns
            return (observation, done)

        # 3. Si no emitió ni SQL ni answer (o formato inválido)
        done = self.tool_turns_used >= self.max_tool_turns
        obs = build_tool_observation("", error="No se detectó <sql> ni <answer> válido.")
        self.messages.append({"role": "user", "content": obs})
        return (obs, done)

    def run_episode(self, task: Task, policy: Callable[[List[Dict[str, str]]], str]) -> Episode:
        """Ejecuta el ciclo completo usando `policy(messages) -> completion`."""
        init_prompt = self.reset(task)
        done = False

        while not done:
            completion = policy(self.messages)
            _, done = self.step(completion)

        return Episode(
            task_id=task.id,
            prompt=init_prompt,
            turns=list(self.messages),
            final_answer=self.last_final_answer,
            completion=completion,
        )


if __name__ == "__main__":
    env = SQLEnvironment(max_tool_turns=2)
    task = Task(
        id="hosp_0421",
        database="hospital.sqlite",
        difficulty="easy",
        template_id="template_01",
        features=["join"],
        question="¿Cuál es el nombre del departamento del piso 1?",
        gold_sql="SELECT name FROM departments WHERE floor = 1 LIMIT 1;",
        answer_type="text",
        ordered=False,
        result_columns=["name"],
        expected_result="Urgencias",
    )

    prompt = env.reset(task)
    print("Prompt inicial generado con éxito!")

    print(prompt)

    def mock_policy(msgs: List[Dict[str, str]]) -> str:
        last = msgs[-1]["content"]
        if "<result>" in last:
            return "<answer>Urgencias</answer>"
        return "<sql>SELECT name FROM departments WHERE floor = 1 LIMIT 1;</sql>"

    ep = env.run_episode(task, mock_policy)
    print(f"Episodio ejecutado. Final answer: {ep.final_answer}")
    print(f"Total de turnos: {len(ep.turns)}")
    print(f"Completion: {ep.completion}")
