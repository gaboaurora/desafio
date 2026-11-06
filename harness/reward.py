"""
Función de recompensa con la interfaz común del curso:

    reward(prompt, completion) -> float

Flujo:
  1. `extract_task_id(prompt)` recupera la tarea y su ground truth desde el registro
     (dataset/*.jsonl cargado en memoria).
  2. `parse_completion(completion)` extrae SQL y respuesta final; completions inválidas => 0.0.
  3. Si la completion incluye la(s) consulta(s) <sql>, el harness las RE-EJECUTA y verifica que la
     respuesta final sea consistente con el resultado real (evita que el modelo invente <result>).
  4. `verify(...)` compara la respuesta final con el ground truth.

Recompensa binaria por defecto (1.0 / 0.0). Opcionalmente, `shaped_reward` entrega crédito parcial
(p.ej. 0.2 por SQL ejecutable, 0.5 por resultado correcto sin respuesta final bien formateada) para
experimentos del proyecto de curso; la versión final reportada debe ser la binaria.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from harness.environment import Task
from harness.parser import parse_completion
from harness.prompt_builder import extract_task_id
from harness.sql_executor import SQLResult, SQLExecutor
from harness.verifier import _ANSWER_ALIASES, _parse_bool, verify

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
DB_DIR = ROOT / "db"

_TASK_REGISTRY: Dict[str, Task] = {}


def load_task_registry(*jsonl_paths: str) -> int:
    """Carga tareas desde archivos JSONL al registro global. Devuelve cuántas tareas hay."""
    for path in jsonl_paths:
        p = Path(path)
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                task = Task.from_dict(json.loads(line))
                _TASK_REGISTRY[task.id] = task
    return len(_TASK_REGISTRY)


def _ensure_registry_loaded() -> None:
    if not _TASK_REGISTRY:
        default_paths = [str(DATASET_DIR / f"{split}.jsonl") for split in ("train", "eval", "test")]
        load_task_registry(*default_paths)


def get_task(task_id: str) -> Optional[Task]:
    if not task_id:
        return None
    if task_id not in _TASK_REGISTRY:
        _ensure_registry_loaded()
    return _TASK_REGISTRY.get(task_id)


def reward(prompt: str, completion: str) -> float:
    """Recompensa binaria: 1.0 si la respuesta final es correcta y la completion es válida."""
    if not isinstance(prompt, str) or not isinstance(completion, str):
        return 0.0
    task_id = extract_task_id(prompt)
    if not task_id:
        return 0.0
    task = get_task(task_id)
    if task is None:
        return 0.0

    parsed = parse_completion(completion)
    if not parsed.is_valid or not parsed.has_answer or not parsed.sql_queries:
        return 0.0

    # 1. Verificar que la respuesta final coincida con el ground truth de la tarea
    ver = verify(
        parsed.final_answer,
        task.expected_result,
        task.answer_type,
        ordered=task.ordered,
    )
    if not ver.is_correct:
        return 0.0

    # 2. Re-ejecutar la última consulta SQL del modelo y verificar coherencia con su respuesta
    executor = SQLExecutor(DB_DIR / task.database)
    real_value = executor.execute(parsed.sql_queries[-1])
    if not real_value.ok:
        return 0.0

    if not _is_coherent(parsed.final_answer, real_value, task):
        return 0.0

    return 1.0


def process_type_sql(answer_type: str, sql_result: SQLResult) -> Any:
    """Extrae el valor tipado principal desde el resultado SQL según `answer_type`."""
    kind = _ANSWER_ALIASES.get(answer_type, answer_type)
    rows = sql_result.rows
    if kind in ("number", "text"):
        if not rows or not rows[0]:
            return None
        return rows[0][0]
    if kind == "boolean":
        if not rows or not rows[0]:
            return False
        val = rows[0][0]
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        if isinstance(val, str):
            return _parse_bool(val)
        return None
    if kind == "list":
        if not rows:
            return []
        return [r[0] for r in rows if r]
    if kind == "table":
        if not rows:
            return []
        return [list(r) for r in rows]
    return None


def _is_coherent(final_answer: str, sql_result: SQLResult, task: Task) -> bool:
    """Comprueba que `final_answer` se desprenda del resultado real de la consulta SQL ejecutada."""
    kind = _ANSWER_ALIASES.get(task.answer_type, task.answer_type)
    rows = sql_result.rows

    clean_value = process_type_sql(kind, sql_result)
    if clean_value is not None:
        if verify(final_answer, clean_value, kind, ordered=task.ordered).is_correct:
            return True

    # Si el modelo seleccionó columnas adicionales (p.ej. SELECT id, nombre ... LIMIT 1),
    # aceptamos si alguna columna del resultado coincide con la respuesta.
    if kind in ("number", "text") and len(rows) == 1 and len(rows[0]) > 1:
        return any(verify(final_answer, cell, kind).is_correct for cell in rows[0] if cell is not None)

    if kind == "list" and rows and len(rows[0]) > 1:
        num_cols = len(rows[0])
        for col_idx in range(1, num_cols):
            col_vals = [r[col_idx] for r in rows if len(r) > col_idx]
            if verify(final_answer, col_vals, kind, ordered=task.ordered).is_correct:
                return True

    return False


def shaped_reward(prompt: str, completion: str) -> float:
    """Variante con crédito parcial (solo para experimentos; no es la métrica oficial)."""
    if not isinstance(prompt, str) or not isinstance(completion, str):
        return 0.0
    task_id = extract_task_id(prompt)
    task = get_task(task_id) if task_id else None
    if task is None:
        return 0.0

    parsed = parse_completion(completion)
    if not parsed.is_valid or not parsed.sql_queries:
        return 0.0

    executor = SQLExecutor(DB_DIR / task.database)
    real_value = executor.execute(parsed.sql_queries[-1])
    if not real_value.ok:
        return 0.1

    score = 0.2
    clean_value = process_type_sql(task.answer_type, real_value)
    if clean_value is not None and repr(clean_value) == repr(task.expected_result):
        score = 0.5

    if parsed.has_answer and reward(prompt, completion) == 1.0:
        return 1.0
    return score
