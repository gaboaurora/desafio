"""Tests del parser, incluyendo casos adversariales de formato (reward hacking)."""

import pytest

from harness.parser import clean_sql, extract_answer, extract_sql, parse_completion

SQL = "SELECT COUNT(*) FROM patients WHERE city = 'Talca'"


# ----------------------------------------------------------------------------- casos válidos

def test_single_sql_and_answer():
    p = parse_completion(f"Voy a consultar.\n<sql>{SQL}</sql>\n<result>count\n38</result>\n<answer>38</answer>")
    assert p.is_valid and p.errors == []
    assert p.sql_queries == [SQL]
    assert p.final_answer == "38"
    assert p.result_blocks == ["count\n38"]
    assert [k for k, _ in p.events] == ["sql", "result", "answer"]


def test_sql_only_turn_has_no_answer_but_is_valid():
    p = parse_completion(f"<sql>{SQL}</sql>")
    assert p.is_valid and p.final_answer is None and p.sql_queries == [SQL]


def test_multiple_sql_blocks_preserve_order():
    p = parse_completion("<sql>SELECT 1</sql> texto <sql>SELECT 2</sql><answer>2</answer>")
    assert p.is_valid and p.sql_queries == ["SELECT 1", "SELECT 2"]


def test_tags_are_case_insensitive_and_whitespace_tolerant():
    p = parse_completion(f"<SQL>\n  {SQL};\n</SQL>\n<Answer>  38  </Answer>")
    assert p.is_valid and p.sql_queries == [SQL] and p.final_answer == "38"


def test_markdown_fences_and_trailing_semicolon_are_stripped():
    assert clean_sql(f"```sql\n{SQL};\n```") == SQL
    assert clean_sql(f"```\n{SQL}\n```") == SQL
    assert clean_sql(f"  {SQL}  ") == SQL
    assert clean_sql("SELECT 1; SELECT 2;") == "SELECT 1; SELECT 2"   # el executor rechaza múltiples sentencias


def test_text_outside_tags_is_ignored():
    p = parse_completion("La respuesta es 38. <sql>SELECT 1</sql> Respuesta final: 40")
    assert p.is_valid and p.final_answer is None


def test_unknown_tags_are_ignored():
    p = parse_completion(f"<think>razonando...</think><sql>{SQL}</sql><answer>38</answer>")
    assert p.is_valid and p.final_answer == "38"


# ----------------------------------------------------------------------------- ataques / inválidas

@pytest.mark.parametrize("completion", [
    "<answer>38</answer><answer>40</answer>",
    "<sql>SELECT 1</sql><answer>38</answer> o quizás <answer>40</answer>",
    "<answer>38</answer>\n<answer>38</answer>",   # repetir la misma tampoco vale
])
def test_multiple_answer_blocks_are_invalid(completion):
    p = parse_completion(completion)
    assert not p.is_valid and "multiple_answer_blocks" in p.errors
    assert p.final_answer is None
    assert extract_answer(completion) is None


@pytest.mark.parametrize("completion,error", [
    ("<answer>38", "unclosed_tag:answer"),
    ("<sql>SELECT 1", "unclosed_tag:sql"),
    ("38</answer>", "closing_without_opening:answer"),
    ("<sql>SELECT 1</answer>", "mismatched_closing:answer_closes_sql"),
    ("<answer><sql>SELECT 1</sql></answer>", "nested_tag:sql_inside_answer"),
    ("<sql>SELECT '<answer>38</answer>'</sql>", "nested_tag:answer_inside_sql"),
    ("<answer>38 <answer>40</answer></answer>", "nested_tag:answer_inside_answer"),
])
def test_malformed_tags_are_invalid(completion, error):
    p = parse_completion(completion)
    assert not p.is_valid and error in p.errors
    assert extract_sql(completion) == [] and extract_answer(completion) is None


@pytest.mark.parametrize("completion", ["<answer></answer>", "<answer>   \n </answer>", "<sql>SELECT 1</sql><answer> </answer>"])
def test_empty_answer_is_invalid(completion):
    p = parse_completion(completion)
    assert not p.is_valid and "empty_answer" in p.errors


def test_empty_sql_block_is_invalid():
    p = parse_completion("<sql>  </sql><answer>38</answer>")
    assert not p.is_valid and "empty_sql_block" in p.errors


@pytest.mark.parametrize("completion", ["", "   ", None])
def test_empty_completion_is_invalid(completion):
    p = parse_completion(completion)
    assert not p.is_valid and "empty_completion" in p.errors


def test_model_written_result_block_is_extracted_but_not_trusted():
    """El modelo fabrica un <result>; el parser lo expone solo para auditoría."""
    p = parse_completion(f"<sql>{SQL}</sql><result>count\n999</result><answer>999</answer>")
    assert p.is_valid
    assert p.result_blocks == ["count\n999"]
    assert p.final_answer == "999"          # la verdad la decide el reward re-ejecutando el SQL


def test_answer_without_sql_is_parseable():
    """Responder sin consultar es válido para el parser; penalizarlo es decisión del reward."""
    p = parse_completion("<answer>38</answer>")
    assert p.is_valid and p.sql_queries == [] and p.final_answer == "38"


def test_helpers_agree_with_parse_completion():
    completion = f"<sql>{SQL}</sql><answer>38</answer>"
    assert extract_sql(completion) == [SQL]
    assert extract_answer(completion) == "38"
