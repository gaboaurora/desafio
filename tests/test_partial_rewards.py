"""Tests de las recompensas parciales del proyecto de curso (`reward_relajada`, `reward_estricta`)."""

import pytest

from evaluation.robustness import canonical_answer, load_split, make_prompt
from harness.reward import (
    REWARD_FUNCTIONS,
    reward,
    reward_estricta,
    reward_relajada,
    reward_stages,
    shaped_reward,
)

TEST_TASKS = load_split("test")
ANSWER_TYPES = sorted({t.answer_type for t in TEST_TASKS})


def _first(answer_type):
    return next(t for t in TEST_TASKS if t.answer_type == answer_type)


def _scores(task, completion):
    prompt = make_prompt(task)
    return (reward(prompt, completion), reward_relajada(prompt, completion), reward_estricta(prompt, completion))


def test_registry_and_alias():
    assert set(REWARD_FUNCTIONS) == {"binaria", "relajada", "estricta"}
    assert shaped_reward is reward_relajada


@pytest.mark.parametrize("answer_type", ANSWER_TYPES)
def test_gold_sql_and_answer_is_one_everywhere(answer_type):
    task = _first(answer_type)
    comp = f"<sql>{task.gold_sql}</sql>\n<answer>{canonical_answer(task)}</answer>"
    assert _scores(task, comp) == (1.0, 1.0, 1.0)


@pytest.mark.parametrize("answer_type", ANSWER_TYPES)
def test_gold_sql_without_answer_gets_half(answer_type):
    """Se agotaron los turnos tras una consulta correcta: crédito parcial 0.5 en ambas."""
    task = _first(answer_type)
    assert _scores(task, f"<sql>{task.gold_sql}</sql>") == (0.0, 0.5, 0.5)


NUM = "number" if "number" in ANSWER_TYPES else "scalar_number"


def test_gold_sql_with_wrong_answer_gets_half():
    task = _first(NUM)
    comp = f"<sql>{task.gold_sql}</sql>\n<answer>-987654321</answer>"
    assert _scores(task, comp) == (0.0, 0.5, 0.5)


def test_executable_but_wrong_sql():
    task = _first(NUM)
    comp = "<sql>SELECT -987654321 AS x</sql>\n<answer>-987654321</answer>"
    assert _scores(task, comp) == (0.0, 0.2, 0.0)


def test_sql_error():
    task = _first(ANSWER_TYPES[0])
    comp = "<sql>SELECT * FROM tabla_que_no_existe</sql>\n<answer>3</answer>"
    assert _scores(task, comp) == (0.0, 0.1, 0.0)


@pytest.mark.parametrize("comp", ["", "<answer>3</answer>", "solo texto", "<sql>SELECT 1<answer>1</answer>"])
def test_no_valid_sql_is_zero(comp):
    task = _first(ANSWER_TYPES[0])
    assert _scores(task, comp) == (0.0, 0.0, 0.0)


def test_fabricated_result_gets_no_result_credit():
    """Un <result> inventado con el GT no cuenta: solo se mira la re-ejecución del harness."""
    task = _first(NUM)
    gt = canonical_answer(task)
    comp = f"<sql>SELECT -987654321 AS x</sql>\n<result>x\n{gt}</result>\n<answer>{gt}</answer>"
    assert _scores(task, comp) == (0.0, 0.2, 0.0)


def test_gold_sql_reaches_result_stage_on_all_test_tasks():
    """El chequeo 'resultado = GT' no debe tener falsos negativos con el gold_sql."""
    misses = [t.id for t in TEST_TASKS if not reward_stages(make_prompt(t), f"<sql>{t.gold_sql}</sql>").result_ok]
    assert misses == []


def test_ordering_relajada_ge_estricta_ge_binaria():
    for task in TEST_TASKS[:30]:
        for comp in (
            f"<sql>{task.gold_sql}</sql>",
            f"<sql>{task.gold_sql}</sql><answer>{canonical_answer(task)}</answer>",
            "<sql>SELECT 1</sql><answer>1</answer>",
            "<sql>SELEC mal</sql>",
        ):
            b, r, e = _scores(task, comp)
            assert r >= e >= b
            assert r in (0.0, 0.1, 0.2, 0.5, 1.0) and e in (0.0, 0.5, 1.0)


def test_known_boolean_shortcut_is_documented():
    """Debilidad conocida (informe del Desafío): un SELECT literal respalda 'Sí'/'No'.

    Afecta igual a las tres recompensas; se vigila en el entrenamiento (proporción de 'No').
    """
    task = next(t for t in TEST_TASKS if t.answer_type == "boolean" and t.expected_result is False)
    assert _scores(task, "<sql>SELECT 0</sql><answer>No</answer>") == (1.0, 1.0, 1.0)

