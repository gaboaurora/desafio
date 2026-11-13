"""Pares DPO: zona aprendible, cortes sql/answer y prioridad de rechazos."""

import json

import pytest

from evaluation.robustness import canonical_answer, load_split
from harness.environment import Episode
from training_data.generate_dpo import (
    build_pairs,
    categorize_failure,
    pairs_from_episodes,
)


@pytest.fixture(scope="module")
def task():
    for candidate in load_split("train"):
        if candidate.answer_type == "number" and "COUNT(*)" in candidate.gold_sql:
            return candidate
    raise AssertionError("no hay un conteo easy en train")


def _episode(task, steps, reward, answer, prompt=None) -> Episode:
    turns = [
        {"role": "system", "content": "sistema"},
        {"role": "user", "content": f"[TASK_ID: {task.id}]\n\n{task.question}"},
        *steps,
    ]
    return Episode(
        task_id=task.id,
        prompt=prompt or f"[TASK_ID: {task.id}]\n\n{task.question}",
        turns=turns,
        final_answer=answer,
        reward=reward,
    )


def _success(task, sql=None, answer=None, result="<result>count\n17</result>"):
    return _episode(
        task,
        [
            {"role": "assistant", "content": f"<sql>{sql or task.gold_sql}</sql>"},
            {"role": "user", "content": result},
            {"role": "assistant", "content": f"<answer>{answer or canonical_answer(task)}</answer>"},
        ],
        reward=1.0,
        answer=answer or canonical_answer(task),
    )


def _failure(task, assistant, result=None, later=None):
    steps = [{"role": "assistant", "content": assistant}]
    if result is not None:
        steps.append({"role": "user", "content": result})
    if later is not None:
        steps.append({"role": "assistant", "content": later})
    return _episode(task, steps, reward=0.0, answer=None)


def test_categorize_failure_ranks_the_five_labels(task):
    assert categorize_failure(_failure(task, "<sql>SELECT 1</sql>\n<answer>999999</answer>")) == "wrong_answer"
    assert categorize_failure(_failure(task, "<sql>SELECT FROM patients</sql>")) == "sql_error"
    assert categorize_failure(_failure(task, "<answer>1</answer><answer>2</answer>")) == "multiple_answers"
    assert categorize_failure(_failure(task, "<sql>SELECT 1")) == "invalid_format"
    assert categorize_failure(_failure(task, "no tengo la respuesta")) == "no_answer"


def test_sql_pair_uses_the_shared_prefix_and_only_the_assistant_turn(task):
    good = _success(task)
    bad = _failure(task, "<sql>SELECT 1</sql>", result="<result>1\n1</result>", later="<answer>1</answer>")
    pairs = build_pairs({task.id: [good, bad]}, max_pairs_per_task=2)
    sql_pairs = [pair for pair in pairs if pair["turn"] == "sql"]
    assert len(sql_pairs) == 1
    pair = sql_pairs[0]
    assert pair["difficulty"] == task.difficulty
    assert pair["failure_category"] == "wrong_answer"
    assert [message["role"] for message in pair["prompt"]] == ["system", "user"]
    assert task.gold_sql in pair["chosen"]
    assert pair["rejected"] == "<sql>SELECT 1</sql>"
    assert "<result>" not in pair["chosen"]
    assert "<result>" not in pair["rejected"]
    assert "_similarity" not in pair


def test_answer_pair_requires_the_same_result_text(task):
    result = "<result>count\n17</result>"
    good = _success(task, result=result)
    same_result = _failure(
        task,
        f"<sql>{task.gold_sql}</sql>",
        result=result,
        later="<answer>0</answer>",
    )
    other_result = _failure(
        task,
        "<sql>SELECT 1</sql>",
        result="<result>1\n1</result>",
        later="<answer>1</answer>",
    )
    pairs = build_pairs({task.id: [good, same_result]}, max_pairs_per_task=2)
    assert [pair["turn"] for pair in pairs] == ["answer"]
    assert pairs[0]["chosen"].strip() == f"<answer>{canonical_answer(task)}</answer>"
    assert "<answer>0</answer>" in pairs[0]["rejected"]
    assert pairs[0]["prompt"][-1]["content"] == result

    mismatched = build_pairs({task.id: [good, other_result]}, max_pairs_per_task=2)
    assert all(pair["turn"] == "sql" for pair in mismatched)


def test_identical_assistant_text_is_not_a_pair(task):
    good = _success(task)
    echoed = _failure(task, f"  <sql>{task.gold_sql}</sql>  ", result="<result>otro</result>")
    assert build_pairs({task.id: [good, echoed]}) == []


def test_only_successes_produce_no_pairs(task):
    assert build_pairs({task.id: [_success(task), _success(task)]}) == []


def test_two_slots_prefer_distinct_categories_and_the_closer_sql(task):
    good = _success(task, sql="SELECT COUNT(*) FROM patients WHERE city = 'Santiago'")
    close = _failure(
        task,
        "<sql>SELECT COUNT(*) FROM patients WHERE city = 'Temuco'</sql>",
        result="<result>count\n3</result>",
        later="<answer>999999</answer>",
    )
    far = _failure(task, "<sql>SELECT 1</sql>", result="<result>1\n1</result>", later="<answer>999999</answer>")
    broken = _failure(task, "<sql>SELECT FROM patients</sql>")
    silent = _failure(task, "no tengo la respuesta")
    pairs = build_pairs({task.id: [good, close, far, broken, silent]}, max_pairs_per_task=2)
    assert len(pairs) == 2
    by_category = {pair["failure_category"]: pair["rejected"] for pair in pairs}
    assert set(by_category) == {"wrong_answer", "sql_error"}
    assert "Temuco" in by_category["wrong_answer"]
    assert "SELECT FROM" in by_category["sql_error"]


def test_a_correct_completion_marked_zero_is_not_rejected(task):
    alt = task.gold_sql.replace("COUNT(*)", "(COUNT(*))", 1)
    good = _success(task)
    disguised = _episode(
        task,
        [
            {"role": "assistant", "content": f"<sql>{alt}</sql>"},
            {"role": "user", "content": "<result>count\n1</result>"},
            {"role": "assistant", "content": f"<answer>{canonical_answer(task)}</answer>"},
        ],
        reward=0.0,
        answer=canonical_answer(task),
    )
    assert build_pairs({task.id: [good, disguised]}) == []


def test_pairs_keep_only_the_allowed_task_ids(task):
    other = next(candidate for candidate in load_split("train") if candidate.id != task.id)
    good = _success(task)
    bad = _failure(task, "<sql>SELECT 1</sql>", later="<answer>999999</answer>")
    foreign = _success(other)
    foreign_bad = _failure(other, "<sql>SELECT 1</sql>")
    pairs = pairs_from_episodes([good, bad, foreign, foreign_bad], allowed_ids={task.id})
    assert pairs
    assert {pair["task_id"] for pair in pairs} == {task.id}


def test_main_writes_jsonl(task, tmp_path, monkeypatch):
    from training_data import generate_dpo

    traces = tmp_path / "traces.jsonl"
    tasks = tmp_path / "tasks.jsonl"
    out = tmp_path / "dpo.jsonl"
    good = _success(task)
    bad = _failure(task, "<sql>SELECT FROM patients</sql>")
    with traces.open("w", encoding="utf-8") as handle:
        for episode in (good, bad):
            handle.write(json.dumps({
                "task_id": episode.task_id,
                "prompt": episode.prompt,
                "turns": episode.turns,
                "final_answer": episode.final_answer,
                "reward": episode.reward,
            }, ensure_ascii=False) + "\n")
    tasks.write_text(json.dumps({"id": task.id}) + "\n", encoding="utf-8")

    generate_dpo.main([
        "--traces", str(traces),
        "--tasks", str(tasks),
        "--out", str(out),
    ])
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["task_id"] == task.id
    assert record["failure_category"] == "sql_error"
    assert record["turn"] == "sql"
