"""Tests de la auditoría de robustez (evaluation/robustness.py).

La auditoría estática (parser + verificador) corre sin reward. Los casos de la capa "reward" se
prueban con un reward falso para validar el cálculo de tasas.
"""

from collections import Counter

import pytest

from evaluation.robustness import (
    ANSWER_TYPES,
    build_cases,
    canonical_answer,
    fmt_en,
    fmt_es,
    load_split,
    number_to_spanish_words,
    run,
    select_tasks,
    static_decision,
)
from harness.verifier import verify


# Equivalencias que el verificador actual rechaza (hallazgos de la auditoría, no errores del caso):
#   number_extra_precision: "45.673" se lee como 45.673 miles (45673) y "0.673" como 673, porque un
#   punto seguido de 3 dígitos se interpreta como separador de miles; con 4 decimales no hay número.
KNOWN_VERIFIER_LIMITATIONS = {"number_extra_precision"}


@pytest.fixture(scope="module")
def test_tasks():
    return load_split("test")


@pytest.fixture(scope="module")
def cases():
    return build_cases(per_type=3, seed=2026, split="test")


@pytest.fixture(scope="module")
def task_index(test_tasks):
    return {t.id: t for t in test_tasks}


# ----------------------------------------------------------------------------- formatos auxiliares

@pytest.mark.parametrize("n,words", [(0, "cero"), (16, "dieciséis"), (21, "veintiuno"), (30, "treinta"),
                                     (42, "cuarenta y dos"), (99, "noventa y nueve"), (100, "cien"), (101, None)])
def test_number_to_spanish_words(n, words):
    assert number_to_spanish_words(n) == words


def test_number_formats_are_accepted_by_verifier():
    for value in (40, 1234, 61234.5, 4123456.78, 45.67):
        for text in (fmt_es(value), fmt_en(value)):
            assert verify(text, value, "number").is_correct, text
    assert fmt_es(61234.5) == "61.234,5" and fmt_en(61234.5) == "61,234.5"


def test_canonical_answer_is_correct_for_every_test_task(test_tasks):
    for task in test_tasks:
        assert verify(canonical_answer(task), task.expected_result, task.answer_type, ordered=task.ordered).is_correct, task.id


# ----------------------------------------------------------------------------- construcción de casos

def test_selection_covers_every_answer_type(test_tasks):
    chosen = select_tasks(test_tasks, per_type=2, seed=1)
    assert Counter(t.answer_type for t in chosen) == {t: 2 for t in ANSWER_TYPES}


def test_cases_are_deterministic(cases):
    again = build_cases(per_type=3, seed=2026, split="test")
    assert [(c.name, c.task_id, c.completion) for c in cases] == [(c.name, c.task_id, c.completion) for c in again]


def test_cases_cover_types_categories_and_layers(cases):
    assert {c.answer_type for c in cases} == set(ANSWER_TYPES)
    assert {c.category for c in cases} == {"equivalence", "attack", "policy"}
    assert {c.layer for c in cases} == {"static", "reward"}
    for c in cases:
        assert (c.expected_reward is None) == (c.category == "policy")


# ----------------------------------------------------------------------------- auditoría estática

def _static_failures(cases, task_index, category, exclude=KNOWN_VERIFIER_LIMITATIONS):
    return [
        (c.name, c.task_id, c.completion[:120])
        for c in cases
        if c.category == category and c.layer == "static" and c.name not in exclude
        and static_decision(c, task_index[c.task_id]) != c.expected_reward
    ]


def test_static_audit_accepts_all_equivalences(cases, task_index):
    assert _static_failures(cases, task_index, "equivalence") == []


@pytest.mark.xfail(reason="el verificador lee '45.673' como 45673 (separador de miles)", strict=False)
def test_verifier_accepts_extra_decimal_precision(cases, task_index):
    limited = [c for c in cases if c.name in KNOWN_VERIFIER_LIMITATIONS]
    assert limited, "se esperaban casos con decimales extra en la muestra"
    assert all(static_decision(c, task_index[c.task_id]) == 1.0 for c in limited)


def test_static_audit_rejects_all_attacks(cases, task_index):
    assert _static_failures(cases, task_index, "attack") == []


def test_run_static_skips_reward_layer(cases, task_index):
    summary = run(cases, mode="static", tasks=task_index)
    n_reward_layer = sum(c.layer == "reward" and c.category != "policy" for c in cases)
    assert summary["n_skipped"] == n_reward_layer
    assert summary["policy"] == {}
    assert summary["rejection_rate_attacks"] == 1.0
    assert {f["name"] for f in summary["failures"]} <= KNOWN_VERIFIER_LIMITATIONS


# ----------------------------------------------------------------------------- modo reward (reward falso)

def test_run_reward_with_perfect_reward(cases, task_index):
    expected = {c.completion: c.expected_reward for c in cases}

    def oracle_reward(prompt, completion):
        value = expected[completion]
        return 0.0 if value is None else value

    summary = run(cases, mode="reward", reward_fn=oracle_reward, tasks=task_index)
    assert summary["acceptance_rate_equivalences"] == 1.0
    assert summary["rejection_rate_attacks"] == 1.0
    assert summary["n_failures"] == 0
    assert set(summary["policy"]) == {c.name for c in cases if c.category == "policy"}


def test_run_reward_detects_a_hackable_reward(cases, task_index):
    summary = run(cases, mode="reward", reward_fn=lambda prompt, completion: 1.0, tasks=task_index)
    assert summary["acceptance_rate_equivalences"] == 1.0
    assert summary["rejection_rate_attacks"] == 0.0
    assert summary["n_failures"] == summary["attacks"]["n"]


def test_run_reward_flags_out_of_range_and_exceptions(cases, task_index):
    def broken(prompt, completion):
        if "<answer>" not in completion:
            raise ValueError("boom")
        return 2.0

    summary = run(cases[:20], mode="reward", reward_fn=broken, tasks=task_index)
    assert summary["n_failures"] == summary["n_judged"]


def test_prompt_contains_task_id(cases, task_index):
    seen = []

    def spy(prompt, completion):
        seen.append(prompt)
        return 0.0

    run(cases[:3], mode="reward", reward_fn=spy, tasks=task_index)
    assert all(f"[TASK_ID: {c.task_id}]" in p for c, p in zip(cases[:3], seen))
