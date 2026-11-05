"""Tests del ejecutor SQL (validación estática + ejecución en BD de prueba)."""

import sqlite3

import pytest

from harness.sql_executor import SQLExecutor

SAFE_SELECT = "SELECT COUNT(*) AS n FROM patients WHERE city = 'Talca'"
SAFE_CTE = "WITH t AS (SELECT 1 AS n) SELECT n FROM t"


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE students",
        "SELECT 1; DELETE FROM students",
        "INSERT INTO students VALUES (1)",
        "SELECT pg_sleep(100)",
        "COPY students TO '/tmp/x'",
        "UPDATE patients SET age = 1",
        "ATTACH DATABASE 'otra.sqlite' AS extra",
        "PRAGMA table_info(patients)",
        "SELECT load_extension('x')",
        "SELECT 1; SELECT 2",
        "/* sin cerrar",
        "",
        "SELECT 'texto'; DROP TABLE patients",
    ],
)
def test_rejects_unsafe_statements(sql):
    assert not SQLExecutor.is_safe_select(sql)


def test_accepts_select_and_cte():
    assert SQLExecutor.is_safe_select(SAFE_SELECT)
    assert SQLExecutor.is_safe_select(SAFE_CTE)
    assert SQLExecutor.is_safe_select("SELECT replace('a', 'a', 'b')")
    assert SQLExecutor.is_safe_select("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c LIMIT 3) SELECT x FROM c")
    assert SQLExecutor.is_safe_select("SELECT ';' AS punto")
    assert SQLExecutor.is_safe_select("SELECT 1;  -- comentario final\n")


def test_literals_do_not_hide_a_second_statement():
    assert not SQLExecutor.is_safe_select("SELECT 'ok' ; DELETE FROM patients")


def test_execute_select_and_format(tmp_path):
    db = tmp_path / "mini.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE patients (city TEXT, age INTEGER)")
    conn.executemany("INSERT INTO patients VALUES (?, ?)", [("Talca", 30), ("Talca", 40), ("Rancagua", 20)])
    conn.commit()
    conn.close()

    result = SQLExecutor(db).execute(SAFE_SELECT)
    assert result.ok and result.columns == ["n"] and result.rows == [(2,)]
    assert SQLExecutor.format_result(result) == "n\n2"


def test_execute_rejects_writes_without_touching_the_database(tmp_path):
    db = tmp_path / "mini.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE patients (city TEXT)")
    conn.execute("INSERT INTO patients VALUES ('Talca')")
    conn.commit()
    conn.close()

    result = SQLExecutor(db).execute("DELETE FROM patients")
    assert not result.ok
    assert "rechazada" in result.error
    check = sqlite3.connect(db)
    assert check.execute("SELECT COUNT(*) FROM patients").fetchone()[0] == 1
    check.close()


def test_truncates_rows(tmp_path):
    db = tmp_path / "wide.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE t (i INTEGER)")
    conn.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(30)])
    conn.commit()
    conn.close()

    result = SQLExecutor(db, max_rows=5).execute("SELECT i FROM t ORDER BY i")
    assert result.ok and result.truncated
    assert result.rows == [(0,), (1,), (2,), (3,), (4,)]
    text = SQLExecutor.format_result(result)
    assert "truncado a 5 filas" in text


def test_respects_timeout(tmp_path):
    db = tmp_path / "slow.sqlite"
    sqlite3.connect(db).close()
    sql = (
        "WITH RECURSIVE c(x) AS ("
        " SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 100000000"
        ") SELECT COUNT(*) FROM c AS a, c AS b"
    )
    result = SQLExecutor(db, timeout_ms=200).execute(sql)
    assert not result.ok
    assert "tiempo" in result.error
    assert result.elapsed_ms < 5_000


def test_missing_database_returns_error(tmp_path):
    result = SQLExecutor(tmp_path / "no-existe.sqlite").execute("SELECT 1")
    assert not result.ok and "no encontrada" in result.error


def test_sql_error_is_reported(tmp_path):
    db = tmp_path / "mini.sqlite"
    sqlite3.connect(db).close()
    result = SQLExecutor(db).execute("SELECT * FROM tabla_que_no_existe")
    assert not result.ok
    assert "tabla_que_no_existe" in result.error


def test_hospital_database_is_readable():
    from pathlib import Path

    db = Path(__file__).resolve().parent.parent / "db" / "hospital.sqlite"
    if not db.exists():
        pytest.skip("db/hospital.sqlite no generado")
    result = SQLExecutor(db).execute("SELECT COUNT(*) FROM departments")
    assert result.ok and result.rows == [(10,)]
