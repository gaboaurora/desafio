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
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence
from urllib.parse import quote

from dotenv import load_dotenv

load_dotenv()

DEFAULT_DB_PATH = "db/hospital.sqlite"
MAX_SQL_CHARS = 8_000

# Palabras que nunca pueden aparecer como token fuera de literales.
_FORBIDDEN = {
    "insert", "update", "delete", "drop", "alter", "create", "attach", "detach",
    "pragma", "vacuum", "reindex", "begin", "commit", "rollback", "savepoint",
    "grant", "revoke", "copy", "truncate", "merge", "call", "exec", "execute",
    "load_extension", "writefile", "readfile", "edit", "fts3_tokenizer",
    "pg_sleep", "sleep",
}
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


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
        self.max_rows = max_rows if max_rows is not None else int(os.getenv("LLM_MAX_ROWS_RETURNED", "50"))
        self.timeout_ms = timeout_ms if timeout_ms is not None else int(os.getenv("LLM_STATEMENT_TIMEOUT_MS", "3000"))

    @staticmethod
    def is_safe_select(sql: str) -> bool:
        """Valida estáticamente que `sql` sea una sola sentencia de lectura."""
        if not isinstance(sql, str):
            return False
        if "\x00" in sql or len(sql) > MAX_SQL_CHARS:
            return False
        masked = _mask_literals(sql)
        if masked is None:
            return False
        masked = masked.strip()
        if masked.endswith(";"):
            masked = masked[:-1].rstrip()
        if not masked or ";" in masked:
            return False
        tokens = [t.lower() for t in _TOKEN_RE.findall(masked)]
        if not tokens or tokens[0] not in ("select", "with"):
            return False
        if any(t in _FORBIDDEN for t in tokens):
            return False
        # REPLACE INTO es escritura; replace(...) es la función de texto y se permite.
        for i, tok in enumerate(tokens[:-1]):
            if tok == "replace" and tokens[i + 1] == "into":
                return False
        return True

    def execute(self, sql: str) -> SQLResult:
        """Ejecuta la consulta en modo solo lectura y devuelve columnas/filas (truncadas) o el error."""
        started = time.perf_counter()
        if not self.is_safe_select(sql):
            return SQLResult(error="consulta rechazada: solo se permite un SELECT de lectura", elapsed_ms=_elapsed(started))
        if not self.db_path.exists():
            return SQLResult(error="base de datos no encontrada", elapsed_ms=_elapsed(started))

        uri = "file:" + quote(self.db_path.resolve().as_posix(), safe="/:") + "?mode=ro"
        try:
            conn = sqlite3.connect(uri, uri=True)
        except sqlite3.Error as exc:
            return SQLResult(error=f"no se pudo abrir la base: {exc}", elapsed_ms=_elapsed(started))

        def _on_progress() -> int:
            if self.timeout_ms > 0 and (time.perf_counter() - started) * 1000.0 > self.timeout_ms:
                return 1
            return 0

        try:
            conn.set_progress_handler(_on_progress, 1000)
            cur = conn.execute(sql)
            columns = [d[0] for d in cur.description] if cur.description else []
            limit = max(0, self.max_rows)
            fetched = cur.fetchmany(limit + 1)
            truncated = len(fetched) > limit
            return SQLResult(
                columns=columns,
                rows=fetched[:limit],
                truncated=truncated,
                elapsed_ms=_elapsed(started),
            )
        except sqlite3.Error as exc:
            message = str(exc).lower()
            if "interrupt" in message:
                text = "consulta interrumpida por tiempo máximo"
            else:
                text = str(exc).splitlines()[0][:300]
            return SQLResult(error=text, elapsed_ms=_elapsed(started))
        finally:
            conn.close()

    @staticmethod
    def format_result(result: SQLResult) -> str:
        """Serializa el resultado como texto compacto para la observación del modelo."""
        if result.error:
            return f"ERROR: {result.error}"

        def cell(value: Any) -> str:
            if value is None:
                return "NULL"
            if isinstance(value, float):
                return str(int(value)) if value.is_integer() else f"{value:.4f}".rstrip("0").rstrip(".")
            return str(value)

        lines: List[str] = []
        if result.columns:
            lines.append(" | ".join(result.columns))
        lines.extend(" | ".join(cell(v) for v in row) for row in result.rows)
        if not result.rows:
            lines.append("(sin filas)")
        if result.truncated:
            lines.append(f"(resultado truncado a {len(result.rows)} filas)")
        return "\n".join(lines)


def _elapsed(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def _mask_literals(sql: str) -> Optional[str]:
    """Reemplaza literales y comentarios por espacios. None si quedan sin cerrar."""
    out: List[str] = []
    i, n = 0, len(sql)
    while i < n:
        c = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""
        if c == "-" and nxt == "-":
            end = sql.find("\n", i)
            if end < 0:
                break
            out.append(" ")
            i = end
            continue
        if c == "/" and nxt == "*":
            end = sql.find("*/", i + 2)
            if end < 0:
                return None
            out.append(" ")
            i = end + 2
            continue
        if c in ("'", '"'):
            i += 1
            while i < n:
                if sql[i] == c:
                    if i + 1 < n and sql[i + 1] == c:
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            else:
                return None
            out.append(" '' ")
            continue
        out.append(c)
        i += 1
    return "".join(out)
