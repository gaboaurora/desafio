"""Tests del parser, incluyendo casos adversariales de formato."""

import pytest

from harness import parser


def test_module_importable():
    assert callable(parser.parse_completion)


@pytest.mark.skip(reason="pendiente")
def test_extracts_single_sql_and_answer():
    pass


@pytest.mark.skip(reason="pendiente")
def test_multiple_answer_blocks_are_invalid():
    """'<answer>40</answer> ... <answer>41</answer>' => is_valid False."""


@pytest.mark.skip(reason="pendiente")
def test_model_written_result_block_is_ignored():
    """Un <result> fabricado por el modelo no debe tratarse como observación."""


@pytest.mark.skip(reason="pendiente")
def test_malformed_tags_are_invalid():
    pass
