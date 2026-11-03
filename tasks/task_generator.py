"""
Módulo 2: Generación de Tareas y Particiones
Responsabilidad:
  - Poblar las tablas (`students`, `courses`, `enrollments`) usando `POSTGRES_ADMIN_USER`.
  - Generar preguntas por nivel de dificultad (`easy`, `medium`, `hard`).
  - Calcular la respuesta correcta por construcción (ejecutando el SQL canónico en la BD).
  - Exportar particiones `train`, `eval` y `test`.
"""

import os
from typing import Any, Dict, List
from dotenv import load_dotenv

load_dotenv()


class TaskGenerator:
    """Esqueleto para poblar la base de datos y construir el dataset verificable por código."""

    def __init__(self) -> None:
        self.db_host = os.getenv("POSTGRES_HOST", "localhost")
        self.db_port = int(os.getenv("POSTGRES_PORT", "5432"))
        self.db_name = os.getenv("POSTGRES_DB", "sql_reasoning_db")
        self.admin_user = os.getenv("POSTGRES_ADMIN_USER", "harness_admin")
        self.admin_password = os.getenv("POSTGRES_ADMIN_PASSWORD", "")

    def seed_database(self, seed: int = 2026) -> None:
        """Inserta registros sintéticos en `students`, `courses` y `enrollments`."""
        raise NotImplementedError

    def compute_ground_truth(self, reference_sql: str) -> Any:
        """Ejecuta la consulta canónica en PostgreSQL para obtener la respuesta verificable por construcción."""
        raise NotImplementedError

    def build_splits(self) -> Dict[str, List[Dict[str, Any]]]:
        """Genera y retorna las particiones `train`, `eval` y `test` balanceadas por dificultad."""
        raise NotImplementedError
