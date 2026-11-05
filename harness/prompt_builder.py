"""
Construcción del prompt para el modelo.

El prompt incluye:
  - Un identificador de tarea (`[TASK_ID: ...]`) que permite a `reward(prompt, completion)`
    recuperar el ground truth sin parámetros adicionales (interfaz común del curso).
  - El esquema de la base de datos (DDL compacto + descripción de columnas).
  - Las reglas de formato: una consulta dentro de <sql>...</sql>, respuesta final dentro de
    <answer>...</answer>, exactamente una respuesta.
  - La pregunta en lenguaje natural.
"""

from __future__ import annotations

from typing import Optional

TASK_ID_TAG = "[TASK_ID: {task_id}]"


def build_system_prompt(schema_description: str) -> str:
    """Instrucciones generales + esquema. Se mantiene estable entre tareas del mismo esquema."""
    raise NotImplementedError


def build_user_prompt(task_id: str, question: str) -> str:
    """Pregunta concreta precedida por la etiqueta de identificación de la tarea."""
    raise NotImplementedError


def build_prompt(task_id: str, question: str, schema_description: str) -> str:
    """Prompt completo en texto plano (system + user) para la interfaz `reward(prompt, completion)`."""
    raise NotImplementedError


def build_tool_observation(result_text: str, error: Optional[str] = None) -> str:
    """Envuelve el resultado (o error) de la herramienta SQL en <result>...</result>."""
    raise NotImplementedError


def extract_task_id(prompt: str) -> Optional[str]:
    """Recupera el `task_id` embebido en el prompt; None si no está presente."""
    raise NotImplementedError
