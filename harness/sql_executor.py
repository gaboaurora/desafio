"""
Ejecución segura de consultas SQL emitidas por el modelo (SQLite).

Defensas:
  - Solo se acepta una única sentencia SELECT / WITH (sin ';' intermedios, sin DDL/DML, sin PRAGMA
    ni ATTACH). Validación estática antes de tocar la base de datos.
  - Conexión en modo solo lectura (`file:...?mode=ro`), de modo que cualquier escritura falla a
    nivel de motor aunque pase la validación estática.
  - Timeout por consulta mediante `Connection.set_progress_handler`, que aborta consultas largas
    (p.ej. productos cartesianos) tras `LLM_STATEMENT_TIMEOUT_MS`.
  - Límite de filas devueltas (`LLM_MAX_ROWS_RETURNED`) para acotar el largo de la observación.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence

from dotenv import load_dotenv

load_dotenv()

DEFAULT_DB_PATH = "db/hospital.sqlite"


@dataclass
class SQLResult:
    columns: List[str] = field(default_factory=list)
    rows: List[Sequence[Any]] = field(default_factory=list)
    truncated: bool = False
    error: Optional[str] = None
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


class SQLExecutor:
    """Cliente de solo lectura hacia la base de datos SQLite del entorno."""

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        max_rows: Optional[int] = None,
        timeout_ms: Optional[int] = None,
    ) -> None:
        self.db_path = Path(db_path or os.getenv("DB_PATH", DEFAULT_DB_PATH))
        self.max_rows = max_rows or int(os.getenv("LLM_MAX_ROWS_RETURNED", "50"))
        self.timeout_ms = timeout_ms or int(os.getenv("LLM_STATEMENT_TIMEOUT_MS", "3000"))

    @staticmethod
    def is_safe_select(sql: str) -> bool:
        """Valida estáticamente que `sql` sea una sola sentencia de lectura."""
        raise NotImplementedError

    def execute(self, sql: str) -> SQLResult:
        """Ejecuta la consulta en modo solo lectura y devuelve columnas/filas (truncadas) o el error."""
        raise NotImplementedError

    @staticmethod
    def format_result(result: SQLResult) -> str:
        """Serializa el resultado como texto compacto para la observación del modelo."""
        raise NotImplementedError
