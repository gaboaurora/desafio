"""Tests de la interfaz común `reward(prompt, completion) -> float` y de reward hacking end-to-end."""

import importlib
import inspect

import pytest

from harness import reward


def test_reward_signature_matches_course_interface():
    sig = inspect.signature(reward)
    assert list(sig.parameters) == ["prompt", "completion"]


def test_reward_is_exported_from_package():
    reward_module = importlib.import_module("harness.reward")
    assert reward is reward_module.reward


@pytest.mark.skip(reason="pendiente")
def test_reward_in_unit_interval():
    pass


@pytest.mark.skip(reason="pendiente")
def test_unknown_task_id_gives_zero():
    pass


@pytest.mark.skip(reason="pendiente")
def test_fabricated_result_does_not_help():
    """El modelo escribe un <result> falso y una respuesta coherente con él: reward 0."""
