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
    sql_instructions = """Instrucciones de formato para la consulta SQL:
          - La consulta debe estar entre las etiquetas <sql>...</sql>.
          - Solo se permite una consulta por vez.
          - La respuesta final debe estar entre las etiquetas <answer>...</answer>."""
    return f"{sql_instructions}\n\n{schema_description}"


def build_user_prompt(task_id: str, question: str) -> str:
    """Pregunta concreta precedida por la etiqueta de identificación de la tarea."""
    return f"{TASK_ID_TAG.format(task_id=task_id)}\n\n{question}"


def build_prompt(task_id: str, question: str, schema_description: str) -> str:
    """Prompt completo en texto plano (system + user) para la interfaz `reward(prompt, completion)`."""
    system_prompt = build_system_prompt(schema_description)
    user_prompt = build_user_prompt(task_id, question)
    return f"{system_prompt}\n\n{user_prompt}"


def build_tool_observation(result_text: str, error: Optional[str] = None) -> str:
    """Envuelve el resultado (o error) de la herramienta SQL en <result>...</result>."""
    if error:
        return f"<result><error>{error}</error></result>"
    else:
        return f"<result>{result_text}</result>"


def extract_task_id(prompt: str) -> Optional[str]:
    """Recupera el `task_id` embebido en el prompt; None si no está presente."""
    if TASK_ID_TAG in prompt:
        start = prompt.index(TASK_ID_TAG) + len("[TASK_ID: ")
        end = prompt.index("]", start)
        return prompt[start:end].strip()
    return None