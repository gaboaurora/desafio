"""Tests del pipeline de generación (schema, datos, consultas, preguntas)."""

import sqlite3
from pathlib import Path

import pytest

from generator import data_generator, query_generator, question_generator, schema_generator
from generator.data_generator import DataGenerator
from generator.query_generator import MAX_ROWS, TEMPLATES, QueryGenerator, _coerce
from generator.question_generator import QUESTIONS, QuestionGenerator
from generator.schema_generator import build_schema, to_ddl, to_prompt_description


def test_modules_importable():
    assert schema_generator and data_generator and query_generator and question_generator


def test_schema_ddl_is_valid_sqlite():
    """El DDL generado debe poder ejecutarse en SQLite y crear todas las tablas."""
    schema = build_schema()
    conn = sqlite3.connect(":memory:")
    conn.executescript(to_ddl(schema))
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == {t.name for t in schema.tables}


def test_schema_has_required_relational_features():
    """Relación recursiva, N:M y al menos 6 tablas."""
    schema = build_schema()
    fks = schema.foreign_keys()
    assert len(schema.tables) >= 6
    assert any(src == dst for src, _, dst, _ in fks), "falta relación recursiva"
    prescriptions_fks = [fk for fk in fks if fk[0] == "prescriptions"]
    assert len(prescriptions_fks) == 2, "prescriptions debe ser tabla puente N:M"


def test_foreign_keys_reference_existing_columns():
    schema = build_schema()
    for _, _, dst_table, dst_col in schema.foreign_keys():
        assert schema.table(dst_table).column(dst_col).primary_key


def test_check_constraints_are_enforced():
    conn = sqlite3.connect(":memory:")
    conn.executescript(to_ddl(build_schema()))
    conn.execute("INSERT INTO departments VALUES (1, 'Cardiología', 3, 500000000)")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO departments VALUES (2, 'Inexistente', 3, 500000000)")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO departments VALUES (3, 'Pediatría', 99, 500000000)")


def test_prompt_description_mentions_every_table():
    schema = build_schema()
    desc = to_prompt_description(schema)
    for t in schema.tables:
        assert f"{t.name}(" in desc
    assert "supervisor_id" in desc and "doctors(doctor_id)" in desc


@pytest.fixture(scope="module")
def generated():
    gen = DataGenerator(seed=123)
    data = gen.generate()
    conn = gen.build_database(data, ":memory:")
    yield gen, data, conn
    conn.close()


def test_data_generation_is_deterministic():
    """Misma semilla => mismos datos; distinta semilla => datos distintos."""
    a = DataGenerator(seed=7).generate()
    b = DataGenerator(seed=7).generate()
    c = DataGenerator(seed=8).generate()
    assert a == b
    assert a["patients"] != c["patients"]


def test_row_counts_match_sizes(generated):
    gen, data, conn = generated
    counts = DataGenerator.summary(conn, gen.schema)
    assert counts["departments"] == 10
    assert counts["patients"] == gen.sizes.patients
    assert counts["appointments"] == gen.sizes.appointments
    assert counts["admissions"] == gen.sizes.admissions
    assert 50 <= counts["doctors"] <= 70
    assert counts["prescriptions"] > counts["appointments"] * 0.6


def test_foreign_keys_and_checks_hold(generated):
    _, _, conn = generated
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_domain_correlations(generated):
    """Restricciones semánticas que hacen no triviales las preguntas con varias condiciones."""
    _, _, conn = generated
    q = conn.execute
    assert q("SELECT MAX(p.age) FROM appointments a JOIN doctors d USING(doctor_id) "
             "JOIN patients p USING(patient_id) WHERE d.specialty='Pediatría'").fetchone()[0] <= 15
    assert q("SELECT COUNT(*) FROM appointments a JOIN doctors d USING(doctor_id) "
             "JOIN patients p USING(patient_id) WHERE d.specialty='Ginecología' AND p.sex='M'").fetchone()[0] == 0
    assert q("SELECT COUNT(*) FROM appointments WHERE status='completed' AND diagnosis IS NULL").fetchone()[0] == 0
    assert q("SELECT COUNT(*) FROM appointments WHERE status<>'completed' AND diagnosis IS NOT NULL").fetchone()[0] == 0
    assert q("SELECT COUNT(*) FROM prescriptions pr JOIN appointments a USING(appointment_id) "
             "WHERE a.status<>'completed'").fetchone()[0] == 0
    assert q("SELECT COUNT(*) FROM doctors WHERE supervisor_id IS NULL").fetchone()[0] == 10
    assert q("SELECT COUNT(*) FROM appointments a JOIN doctors d USING(doctor_id) "
             "WHERE a.appointment_date < d.hire_date").fetchone()[0] == 0
    # El médico supervisor pertenece al mismo departamento.
    assert q("SELECT COUNT(*) FROM doctors d JOIN doctors s ON d.supervisor_id=s.doctor_id "
             "WHERE d.department_id<>s.department_id").fetchone()[0] == 0


def test_non_degenerate_answers(generated):
    """Preguntas típicas deben tener respuestas no vacías y sin empates en los extremos."""
    _, _, conn = generated
    q = conn.execute
    assert q("SELECT COUNT(*) FROM doctors d JOIN doctors s ON d.supervisor_id=s.doctor_id "
             "WHERE d.salary > s.salary").fetchone()[0] >= 3
    assert q("SELECT COUNT(*) FROM medications m WHERE NOT EXISTS "
             "(SELECT 1 FROM prescriptions WHERE medication_id=m.medication_id)").fetchone()[0] >= 1
    assert q("SELECT COUNT(*) FROM doctors d WHERE NOT EXISTS "
             "(SELECT 1 FROM appointments WHERE doctor_id=d.doctor_id)").fetchone()[0] >= 1
    assert q("SELECT COUNT(*) FROM admissions WHERE discharge_date IS NULL").fetchone()[0] >= 10
    assert q("SELECT COUNT(*) FROM (SELECT salary FROM doctors GROUP BY salary HAVING COUNT(*)>1)").fetchone()[0] == 0
    assert q("SELECT COUNT(*) FROM (SELECT unit_price FROM medications GROUP BY 1 HAVING COUNT(*)>1)").fetchone()[0] == 0
    assert q("SELECT COUNT(*) FROM (SELECT full_name FROM patients GROUP BY 1 HAVING COUNT(*)>1)").fetchone()[0] == 0


def test_committed_database_matches_generator():
    """db/hospital.sqlite debe ser exactamente lo que produce el generador con la semilla por defecto."""
    db_path = DB_PATH
    if not db_path.exists():
        pytest.skip("db/hospital.sqlite no generado")
    gen = DataGenerator(seed=2026)
    data = gen.generate()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        for table in gen.schema.tables:
            cols = [c.name for c in table.columns]
            db_rows = conn.execute(f"SELECT {', '.join(cols)} FROM {table.name} ORDER BY {cols[0]}").fetchall()
            gen_rows = [tuple(r[c] for c in cols) for r in data[table.name]]
            assert db_rows == gen_rows, f"desfase en {table.name}"
    finally:
        conn.close()


# ----------------------------------------------------------------------------------------------
# Consultas, preguntas y particiones
# ----------------------------------------------------------------------------------------------

DB_PATH = Path(__file__).resolve().parent.parent / "db" / "hospital.sqlite"
DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

requires_db = pytest.mark.skipif(not DB_PATH.exists(), reason="db/hospital.sqlite no generado")


def _recompute(conn: sqlite3.Connection, task: dict):
    rows = conn.execute(task["gold_sql"]).fetchall()
    at = task["answer_type"]
    if at == "number":
        return _coerce(rows[0][0])
    if at == "text":
        return str(rows[0][0])
    if at == "boolean":
        return bool(rows[0][0])
    if at == "list":
        return [_coerce(r[0]) for r in rows]
    return [[_coerce(v) for v in r] for r in rows]


@pytest.fixture(scope="module")
def small_dataset():
    qg = QueryGenerator(db_path=DB_PATH, seed=11)
    queries = qg.generate({"easy": 40, "medium": 40, "hard": 40, "expert": 40})
    gen = QuestionGenerator(seed=11)
    tasks = gen.build_tasks(queries)
    return qg, tasks, gen.split(tasks)


@requires_db
def test_every_template_has_questions_and_instantiates():
    assert set(QUESTIONS) == {t.id for t in TEMPLATES}
    qg = QueryGenerator(db_path=DB_PATH, seed=5)
    for t in TEMPLATES:
        q = qg.try_instantiate(t, attempts=200)
        assert q is not None, f"plantilla sin instancias válidas: {t.id}"
        for variant in QUESTIONS[t.id]:
            text = variant.format(**q.slots)
            assert "{" not in text and "}" not in text


@requires_db
def test_generated_queries_are_well_posed(small_dataset):
    qg, tasks, _ = small_dataset
    assert len(tasks) == 160
    for t in tasks:
        d = t.to_dict()
        assert _recompute(qg.conn, d) == t.expected_result
        if t.answer_type == "number":
            assert t.expected_result != 0
        if t.answer_type in ("list", "table"):
            assert 1 <= len(t.expected_result) <= MAX_ROWS
    assert len({t.gold_sql for t in tasks}) == len(tasks)
    assert len({t.question for t in tasks}) == len(tasks)


@requires_db
def test_splits_are_disjoint_and_heldout_templates_only_in_test(small_dataset):
    _, tasks, splits = small_dataset
    ids = [t.id for s in splits.values() for t in s]
    assert len(ids) == len(set(ids)) == len(tasks)
    heldout = {t.template_id for t in splits["test"] if t.template_heldout}
    assert heldout
    for name in ("train", "eval"):
        assert not any(t.template_id in heldout for t in splits[name])
    assert any(not t.template_heldout for t in splits["test"]), "test debe incluir plantillas vistas"
    assert all(t.difficulty in {"easy", "medium", "hard", "expert"} for t in tasks)


@requires_db
@pytest.mark.skipif(not (DATASET_DIR / "train.jsonl").exists(), reason="dataset no generado")
def test_committed_dataset_is_consistent_with_database():
    """Cada expected_result del dataset versionado se obtiene al ejecutar su gold_sql en la BD."""
    import json

    conn = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    all_tasks = []
    for split in ("train", "eval", "test"):
        with (DATASET_DIR / f"{split}.jsonl").open(encoding="utf-8") as f:
            all_tasks += [json.loads(line) for line in f if line.strip()]
    assert len(all_tasks) >= 2000
    assert len({t["id"] for t in all_tasks}) == len(all_tasks)
    required = {"id", "database", "difficulty", "template_id", "features", "question", "gold_sql",
                "answer_type", "ordered", "result_columns", "expected_result", "template_heldout"}
    for t in all_tasks:
        assert required <= set(t)
        assert _recompute(conn, t) == t["expected_result"], t["id"]
    conn.close()
