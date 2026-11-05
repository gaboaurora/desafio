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

from typing import Dict, Optional

from harness.environment import Task

_TASK_REGISTRY: Dict[str, Task] = {}


def load_task_registry(*jsonl_paths: str) -> int:
    """Carga tareas desde archivos JSONL al registro global. Devuelve cuántas tareas hay."""
    raise NotImplementedError


def get_task(task_id: str) -> Optional[Task]:
    return _TASK_REGISTRY.get(task_id)


def reward(prompt: str, completion: str) -> float:
    """Recompensa binaria: 1.0 si la respuesta final es correcta y la completion es válida."""
    raise NotImplementedError


def shaped_reward(prompt: str, completion: str) -> float:
    """Variante con crédito parcial (solo para experimentos; no es la métrica oficial)."""
    raise NotImplementedError
