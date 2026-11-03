"""
Módulo 1: Harness y Verificador
Responsabilidad:
  - Conexión a PostgreSQL (separando usuario `LLM_DB_USER` solo-lectura de `POSTGRES_ADMIN_USER`).
  - Formato de prompt, ejecución de la herramienta SQL (pocos turnos) y parser de la respuesta.
  - Función obligatoria del curso: `reward(prompt, completion) -> float`.
"""

import os
from typing import Any
from dotenv import load_dotenv

load_dotenv()


class SQLHarness:
    """Esqueleto del entorno de interacción y ejecución segura de consultas SQL."""

    def __init__(self) -> None:
        self.db_host = os.getenv("POSTGRES_HOST", "localhost")
        self.db_port = int(os.getenv("POSTGRES_PORT", "5432"))
        self.db_name = os.getenv("POSTGRES_DB", "sql_reasoning_db")
        self.llm_user = os.getenv("LLM_DB_USER", "llm_readonly")
        self.llm_password = os.getenv("LLM_DB_PASSWORD", "")

    def format_prompt(self, question: str) -> str:
        """Construye el prompt con el esquema (Students, Courses, Enrollments) y la pregunta."""
        raise NotImplementedError

    def execute_sql_tool(self, sql_query: str) -> str:
        """Ejecuta una consulta SELECT en PostgreSQL usando el rol `llm_readonly` y devuelve la observación."""
        raise NotImplementedError

    def parse_completion(self, completion: str) -> Any:
        """Extrae la consulta SQL y/o la respuesta final emitida por el modelo."""
        raise NotImplementedError


def reward(prompt: str, completion: str) -> float:
    """
    Debe verificar que la respuesta sea equivalente a la solución correcta y
    resistir intentos de engaño (reward hacking).
    """
    raise NotImplementedError
