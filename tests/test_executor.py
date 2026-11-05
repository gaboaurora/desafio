"""Tests del ejecutor SQL (validación estática + ejecución en BD de prueba)."""

import pytest

from harness.sql_executor import SQLExecutor


def test_module_importable():
    assert SQLExecutor


@pytest.mark.skip(reason="pendiente")
@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE students",
        "SELECT 1; DELETE FROM students",
        "INSERT INTO students VALUES (1)",
        "SELECT pg_sleep(100)",
        "COPY students TO '/tmp/x'",
    ],
)
def test_rejects_unsafe_statements(sql):
    assert not SQLExecutor.is_safe_select(sql)


@pytest.mark.skip(reason="pendiente")
def test_accepts_select_and_cte():
    pass


@pytest.mark.skip(reason="pendiente: requiere BD levantada")
def test_truncates_rows_and_respects_timeout():
    pass
