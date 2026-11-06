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
    instructions = (
        "Eres un asistente experto en bases de datos relacionales SQLite para un entorno hospitalario.\n"
        "Tu objetivo es responder con precisión a la pregunta del usuario consultando la base de datos.\n\n"
        "### Protocolo de Interacción y Formato:\n"
        "1. Razonamiento (<think> ... </think>):\n"
        "   - Antes de realizar una consulta o de emitir tu respuesta final, analiza el problema paso a paso dentro de <think> ... </think>.\n"
        "   - Identifica las tablas involucradas, condiciones WHERE, JOINs, agregaciones o cálculos de fechas (p. ej., julianday).\n"
        "   - Al recibir el resultado de la base de datos (<result>), evalúa si los datos responden la pregunta o si debes ajustar tu estrategia.\n\n"
        "2. Consulta SQL (<sql> ... </sql>):\n"
        "   - Cuando necesites información, emite una única consulta SELECT entre etiquetas <sql> ... </sql>.\n"
        "   - La base de datos es SQLite (solo lectura). No uses comandos destructivos ni múltiples sentencias separadas por ';'.\n"
        "   - El sistema ejecutará tu consulta y te responderá con <result> ... </result>.\n\n"
        "3. Respuesta Final (<answer> ... </answer>):\n"
        "   - Una vez que tengas los datos necesarios para responder, emite exactamente una respuesta dentro de <answer> ... </answer>.\n"
        "   - Sé directo y conciso: incluye solo el dato pedido (número, texto, booleano, lista o tabla) según corresponda.\n"
        "   - No inventes resultados: tu respuesta final debe estar respaldada por los datos obtenidos."
    )
    return f"{instructions}\n\n### Esquema de la Base de Datos:\n{schema_description}"


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