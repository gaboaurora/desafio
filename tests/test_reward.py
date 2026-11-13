"""Tests de la interfaz común `reward(prompt, completion) -> float` y de reward hacking end-to-end."""

import importlib
import inspect

import pytest

from evaluation.robustness import canonical_answer, load_split, make_prompt
from harness import reward


def test_reward_signature_matches_course_interface():
    sig = inspect.signature(reward)
    assert list(sig.parameters) == ["prompt", "completion"]


def test_reward_is_exported_from_package():
    reward_module = importlib.import_module("harness.reward")
    assert reward is reward_module.reward


def test_reward_in_unit_interval():
    task = load_split("test")[0]
    prompt = make_prompt(task)
    good = f"<sql>{task.gold_sql}</sql>\n<answer>{canonical_answer(task)}</answer>"
    bad = "<sql>SELECT 1</sql>\n<answer>respuesta incorrecta</answer>"
    for comp in (good, bad, "", "<answer>42</answer>"):
        r = reward(prompt, comp)
        assert isinstance(r, float)
        assert 0.0 <= r <= 1.0
    assert reward(prompt, good) == 1.0
    assert reward(prompt, bad) == 0.0


def test_unknown_task_id_gives_zero():
    assert reward("[TASK_ID: hosp_999999]\n¿Pregunta?", "<sql>SELECT 1</sql><answer>1</answer>") == 0.0
    assert reward("Prompt sin task id", "<sql>SELECT 1</sql><answer>1</answer>") == 0.0


def test_fabricated_result_does_not_help():
    """El modelo escribe un <result> falso y una respuesta coherente con él: reward 0."""
    task = load_split("test")[0]
    prompt = make_prompt(task)
    fake_completion = f"<sql>{task.gold_sql}</sql>\n<result>count\n999999</result>\n<answer>999999</answer>"
    assert reward(prompt, fake_completion) == 0.0
    unrelated_sql = f"<sql>SELECT 'irrelevante' AS x</sql>\n<result>count\n{canonical_answer(task)}</result>\n<answer>{canonical_answer(task)}</answer>"
    assert reward(prompt, unrelated_sql) == 0.0
