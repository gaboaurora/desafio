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

import re
from typing import Optional

TASK_ID_TAG = "[TASK_ID: {task_id}]"
TASK_ID_PATTERN = re.compile(r"\[TASK_ID:\s*(.*?)\s*\]")


def build_system_prompt(schema_description: str) -> str:
    """Instrucciones generales + esquema. Se mantiene estable entre tareas del mismo esquema."""
    instructions = (
        "Eres un agente experto en bases de datos SQLite para un entorno hospitalario.\n"
        "Tu objetivo es responder con exactitud a la pregunta del usuario consultando la base de datos.\n\n"
        "### PROTOCOLO DE INTERACCIÓN POR TURNO:\n"
        "- Si necesitas consultar la base de datos (o corregir una consulta anterior): emite un bloque "
        "<sql>SELECT ...</sql> y detén tu respuesta inmediatamente para esperar el bloque <result>...</result> del sistema.\n"
        "- Si ya recibiste un <result>...</result> válido con los datos que responden la pregunta: emite un único bloque "
        "<answer>...</answer> con el dato final y termina.\n"
        "- Opcionalmente puedes razonar brevemente antes dentro de <think>...</think>.\n"
        "- PROHIBIDO incluir <sql> y <answer> en el mismo turno.\n"
        "- PROHIBIDO inventar bloques <result>...</result>.\n\n"
        "### REGLAS DE SQLITE Y CORRECCIÓN DE ERRORES:\n"
        "1. <sql>...</sql>:\n"
        "   - Escribe UNA ÚNICA sentencia SELECT válida para SQLite (solo lectura), sin bloques markdown.\n"
        "   - Fechas en SQLite: NO existen YEAR() ni MONTH(). Usa `strftime('%Y', columna) = '2025'`, "
        "`strftime('%Y-%m', columna) = '2025-05'`, o comparaciones ISO como `hire_date >= '2016-01-01'` "
        "('a partir del año 2016' o 'desde 2016' significa únicamente `>= '2016-01-01'` sin fecha límite superior).\n"
        "   - JOINs y alias: usa siempre alias distintos para cada tabla (ej. `departments dp JOIN doctors d ON dp.department_id = d.department_id`; "
        "NUNCA uses el mismo alias `d` para dos tablas).\n"
        "   - Si recibes `<result>ERROR: ...</result>`, NO emitas `<answer>`. Corrige el error de SQL y emite un nuevo `<sql>...</sql>`.\n\n"
        "2. <answer>...</answer>:\n"
        "   - Emite `<answer>` SOLO cuando tengas el resultado de la consulta.\n"
        "   - Incluye únicamente el dato solicitado, sin explicaciones adicionales ni encabezados de columna:\n"
        "     * Preguntas de Sí/No (¿Existe...?, ¿Hay...?, ¿Es...?, Responde sí o no): si hay filas o el conteo es > 0 responde `Sí` (`<answer>Sí</answer>`); si es 0 o `(sin filas)` responde `No` (`<answer>No</answer>`).\n"
        "     * Número o texto (una celda): solo el valor (ej. `<answer>18</answer>` o `<answer>Talca</answer>`).\n"
        "     * Lista (una columna, varias filas): copia TODAS las filas del resultado (una por línea o separadas por coma), sin omitir ninguna.\n"
        "     * Tabla (dos o más columnas): copia TODAS las filas del resultado (sin encabezados), una fila por línea separando las columnas con ` | `.\n\n"
        "### EJEMPLO 1 (Valor único):\n"
        "Turno 1 (Asistente):\n"
        "<think>Contaré los pacientes de Talca con Fonasa en la tabla patients.</think>\n"
        "<sql>SELECT COUNT(*) FROM patients WHERE city = 'Talca' AND insurance = 'Fonasa'</sql>\n\n"
        "Turno del Sistema:\n"
        "<result>COUNT(*)\n18</result>\n\n"
        "Turno 2 (Asistente):\n"
        "<answer>18</answer>\n\n"
        "### EJEMPLO 2 (Lista de varios elementos):\n"
        "Turno 1 (Asistente):\n"
        "<sql>SELECT full_name FROM doctors WHERE specialty = 'Dermatología' ORDER BY full_name</sql>\n\n"
        "Turno del Sistema:\n"
        "<result>full_name\nAna Gómez\nCarlos Ruiz\nElena Díaz</result>\n\n"
        "Turno 2 (Asistente):\n"
        "<answer>Ana Gómez\nCarlos Ruiz\nElena Díaz</answer>\n\n"
        "### EJEMPLO 3 (Corrección de error SQL y respuesta de tabla):\n"
        "Turno 1 (Asistente):\n"
        "<sql>SELECT d.name, COUNT(*) FROM departments d JOIN doctors d ON d.department_id = d.department_id GROUP BY d.name</sql>\n\n"
        "Turno del Sistema:\n"
        "<result>ERROR: ambiguous column name: d.department_id</result>\n\n"
        "Turno 2 (Asistente):\n"
        "<think>Usé el mismo alias d en ambas tablas. Usaré dp para departments y d para doctors.</think>\n"
        "<sql>SELECT dp.name, COUNT(*) FROM departments dp JOIN doctors d ON dp.department_id = d.department_id GROUP BY dp.name</sql>\n\n"
        "Turno del Sistema:\n"
        "<result>name | COUNT(*)\nCardiología | 6\nPediatría | 5</result>\n\n"
        "Turno 3 (Asistente):\n"
        "<answer>Cardiología | 6\nPediatría | 5</answer>"
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
        return f"<result>ERROR: {error}</result>"
    return f"<result>{result_text}</result>"


def extract_task_id(prompt: str) -> Optional[str]:
    """Recupera el `task_id` embebido en el prompt; None si no está presente."""
    match = TASK_ID_PATTERN.search(prompt)
    return match.group(1) if match else None


if __name__ == "__main__":
    schema_desc = "Tabla users(id INT, name TEXT, age INT)"
    task_id = "hosp_0001"
    question = "¿Cuál es la edad promedio de los usuarios?"
    prompt = build_prompt(task_id, question, schema_desc)
    print("Prompt completo:\n", prompt)
    print("Task ID extraído:", extract_task_id(prompt))
    print("Observación de herramienta:\n", build_tool_observation("age\n42"))
    print("Observación de herramienta con error:\n", build_tool_observation("", error="Error en la consulta SQL"))
