"""Tests del verificador: equivalencias que debe aceptar y engaños que debe rechazar."""

import pytest

from harness.verifier import normalize_number, normalize_text, verify


@pytest.mark.parametrize("pred", ["40", "40.0", "cuarenta", "40 estudiantes", "Son 40.", "  40  "])
def test_accepts_equivalent_numbers(pred):
    assert verify(pred, 40, "number").is_correct
    assert verify(pred, 40, "scalar_number").is_correct


@pytest.mark.parametrize("pred", ["40 o 41", "entre 38 y 42", "", "ver resultado", "40, 41, 42", "cuarenta y cinco mil o 40"])
def test_rejects_hedged_answers(pred):
    assert not verify(pred, 40, "number").is_correct


def test_number_formats_and_rounding():
    assert normalize_number("4,0") == 4.0
    assert normalize_number("61.234,5") == 61234.5
    assert normalize_number("1,234.50") == 1234.5
    assert normalize_number("40.000") == 40000
    assert verify("61.234,50", 61234.5, "number").is_correct
    assert verify("61234.504", 61234.5, "number").is_correct
    assert not verify("61.000", 61234.5, "number").is_correct
    assert not verify("61235", 61234.5, "number").is_correct
    assert not verify("40.4", 40, "number").is_correct
    assert verify("cuarenta y dos", 42, "number").is_correct


def test_text_ignores_case_accents_and_filler():
    assert normalize_text("María José") == "maria jose"
    assert verify("ANDRÉS CONTRERAS MOLINA", "Andrés Contreras Molina", "text").is_correct
    assert verify("El médico es Andrés Contreras Molina.", "Andrés Contreras Molina", "text").is_correct
    assert not verify("Andrés Contreras Molina y Pedro Soto", "Andrés Contreras Molina", "text").is_correct
    assert verify("2024-03", "2024-03", "text").is_correct


@pytest.mark.parametrize("pred", ["sí", "Si", "verdadero", "true", "1", "Es correcto"])
def test_accepts_boolean_yes(pred):
    assert verify(pred, True, "boolean").is_correct
    assert not verify(pred, False, "boolean").is_correct


@pytest.mark.parametrize("pred", ["no", "falso", "false", "0"])
def test_accepts_boolean_no(pred):
    assert verify(pred, False, "boolean").is_correct


@pytest.mark.parametrize("pred", ["sí o no", "no sé", "ver resultado", ""])
def test_rejects_ambiguous_booleans(pred):
    assert not verify(pred, True, "boolean").is_correct
    assert not verify(pred, False, "boolean").is_correct


def test_list_order_irrelevant_for_sets():
    target = ["María", "Ana", "Pedro"]
    result = verify("- Pedro\n- Ana\n- María", target, "list", ordered=False)
    assert result.is_correct
    assert verify("Ana, Pedro", target, "list").is_correct is False


def test_list_order_matters_for_rankings():
    target = ["Ana", "Pedro", "María"]
    assert not verify("Pedro, Ana, María", target, "list", ordered=True).is_correct
    assert verify("1. Ana\n2. Pedro\n3. María", target, "list", ordered=True).is_correct


def test_table_ignores_order_and_header_unless_ranked():
    target = [["Cardiología", 61234.5], ["Neurología", 1000]]
    prediction = "departamento | promedio\nNeurología | 1.000\nCardiología | 61.234,50"
    assert verify(prediction, target, "table", ordered=False).is_correct
    assert not verify(
        "Neurología | 1000\nCardiología | 61234.5",
        target,
        "table",
        ordered=True,
    ).is_correct
    ranked = [["Neurología", 1000], ["Cardiología", 61234.5]]
    assert verify("Neurología | 1000\nCardiología | 61234.50", ranked, "table", ordered=True).is_correct


def test_rejects_very_long_answer():
    assert not verify("40 " * 2_000, 40, "number").is_correct
