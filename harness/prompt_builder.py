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

import re





TASK_ID_TAG = "[TASK_ID: {task_id}]"
TASK_ID_PATTERN = re.compile(r"\[TASK_ID:\s*(.*?)\s*\]")

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
    match = TASK_ID_PATTERN.search(prompt)
    return match.group(1) if match else None


# Testear archivo

if __name__ == "__main__":
    # Ejemplo de construcción de prompt
    schema_desc = "Tabla users(id INT, name TEXT, age INT)"
    task_id = "hosp_0001"
    question = "¿Cuál es la edad promedio de los usuarios?"
    prompt = build_prompt(task_id, question, schema_desc)
    print("Prompt completo:\n", prompt)

    # Ejemplo de extracción de task_id
    extracted_id = extract_task_id(prompt)
    print("Task ID extraído:", extracted_id)

    # Ejemplo de construcción de observación de herramienta
    result_text = "Resultado de la consulta SQL"
    observation = build_tool_observation(result_text)
    print("Observación de herramienta:\n", observation)

    error_message = "Error en la consulta SQL"
    error_observation = build_tool_observation(result_text="", error=error_message)
    print("Observación de herramienta con error:\n", error_observation)