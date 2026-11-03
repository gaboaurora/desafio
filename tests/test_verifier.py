"""Tests del verificador: equivalencias que debe aceptar y engaños que debe rechazar."""

import pytest

from harness import verifier


def test_module_importable():
    assert callable(verifier.verify)


@pytest.mark.skip(reason="pendiente")
@pytest.mark.parametrize("pred", ["40", "40.0", "cuarenta", "40 estudiantes", "Son 40."])
def test_accepts_equivalent_numbers(pred):
    assert verifier.verify(pred, 40, "scalar_number").is_correct


@pytest.mark.skip(reason="pendiente")
@pytest.mark.parametrize("pred", ["40 o 41", "entre 38 y 42", "", "ver resultado", "40, 41, 42"])
def test_rejects_hedged_answers(pred):
    assert not verifier.verify(pred, 40, "scalar_number").is_correct


@pytest.mark.skip(reason="pendiente")
def test_list_order_irrelevant_for_sets():
    pass


@pytest.mark.skip(reason="pendiente")
def test_list_order_matters_for_rankings():
    pass
