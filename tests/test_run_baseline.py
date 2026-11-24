"""Tests de la línea base (evaluation/run_baseline.py) con un entorno mínimo de prueba.

`FakeEnv` imita el ciclo de SQLEnvironment (prompt -> <sql> -> <result> -> <answer>) con el parser,
el ejecutor y el verificador reales, para probar la línea base sin depender de environment.py ni de
reward.py, que se implementan aparte.
"""

import json

import pytest

from evaluation import run_baseline as rb
from harness.environment import Episode
from harness.parser import parse_completion
from harness.sql_executor import SQLExecutor
from harness.verifier import verify

ROOT = rb.ROOT


class FakeEnv:
    def __init__(self, max_tool_turns=None):
        self.max_tool_turns = max_tool_turns or 2      # mismo default que SQLEnvironment
        self.executor = SQLExecutor(ROOT / "db" / "hospital.sqlite")

    def run_episode(self, task, policy):
        prompt = f"[TASK_ID: {task.id}]\n\n{task.question}"
        messages = [{"role": "user", "content": prompt}]
        turns = list(messages)
        answer = None
        for _ in range(self.max_tool_turns + 1):
            completion = policy(messages)
            turns.append({"role": "assistant", "content": completion})
            messages.append({"role": "assistant", "content": completion})
            parsed = parse_completion(completion)
            if parsed.has_answer:
                answer = parsed.final_answer
                break
            if parsed.sql_queries:
                obs = "<result>" + SQLExecutor.format_result(self.executor.execute(parsed.sql_queries[-1])) + "</result>"
                turns.append({"role": "tool", "content": obs})
                messages.append({"role": "tool", "content": obs})
        ok = answer is not None and verify(answer, task.expected_result, task.answer_type, ordered=task.ordered).is_correct
        return Episode(task_id=task.id, prompt=prompt, turns=turns, final_answer=answer, reward=1.0 if ok else 0.0)


def scripted_factory(correct_samples):
    """Política que responde bien solo en las muestras indicadas por tarea: {task_id: {0, 2}}."""
    oracle = rb.oracle_policy_factory

    def factory(task, sample):
        if sample in correct_samples.get(task.id, set()):
            return oracle(task, sample)
        return lambda messages: "<answer>no sé</answer>"

    return factory


@pytest.fixture(scope="module")
def tasks():
    return rb.subsample(rb.load_tasks("test"), per_level=3, limit=None, seed=0)


# ----------------------------------------------------------------------------- selección de tareas

def test_subsample_is_stratified_and_deterministic():
    all_tasks = rb.load_tasks("test")
    a = rb.subsample(all_tasks, per_level=4, limit=None, seed=7)
    b = rb.subsample(all_tasks, per_level=4, limit=None, seed=7)
    assert [t.id for t in a] == [t.id for t in b]
    assert sorted({t.difficulty for t in a}) == sorted(rb.LEVELS)
    assert all(sum(t.difficulty == level for t in a) == 4 for level in rb.LEVELS)
    assert len(rb.subsample(all_tasks, per_level=None, limit=5, seed=0)) == 5


# ----------------------------------------------------------------------------- ejecución

def test_oracle_backend_solves_everything(tasks):
    episodes = rb.run_model_on_tasks("oracle", tasks, 2, FakeEnv(), policy_factory=rb.oracle_policy_factory, progress=False)
    summary = rb.summarize_model(tasks, episodes, k=2)
    assert summary["overall"]["pass@1"] == 1.0
    assert summary["calibration"]["all"]["always_solved"] == 1.0
    assert summary["tool_usage"]["executable_sql_rate"] == 1.0


def test_pass_at_k_and_calibration_from_scripted_policy(tasks):
    t0, t1, t2 = tasks[:3]
    script = {t0.id: {0, 1, 2, 3}, t1.id: {0}}           # t0 siempre, t1 1/4, t2 nunca
    subset = [t0, t1, t2]
    episodes = rb.run_model_on_tasks("x", subset, 4, FakeEnv(), policy_factory=scripted_factory(script), progress=False)
    summary = rb.summarize_model(subset, episodes, k=4)
    per_task = {r["task_id"]: r["c"] for r in summary["per_task"]}
    assert per_task == {t0.id: 4, t1.id: 1, t2.id: 0}
    assert summary["overall"]["pass@1"] == pytest.approx((1 + 0.25 + 0) / 3, abs=1e-4)
    assert summary["overall"]["pass@4"] == pytest.approx((1 + 1 + 0) / 3, abs=1e-4)
    cal = summary["calibration"]["all"]
    assert (cal["never_solved"], cal["learnable"], cal["always_solved"]) == pytest.approx((1 / 3, 1 / 3, 1 / 3), abs=1e-4)
    assert summary["k_values"] == [1, 2, 4]


def test_resume_does_not_repeat_episodes(tasks, tmp_path):
    subset = tasks[:2]
    calls = []

    def counting_factory(task, sample):
        calls.append((task.id, sample))
        return rb.oracle_policy_factory(task, sample)

    path = tmp_path / "traces.jsonl"
    with path.open("w", encoding="utf-8") as f:
        def write(task, sample, episode, elapsed):
            f.write(json.dumps(rb.trace_record("m", "oracle", task, sample, episode, elapsed), ensure_ascii=False) + "\n")

        rb.run_model_on_tasks("m", subset, 2, FakeEnv(), policy_factory=counting_factory, on_episode=write, progress=False)
    assert len(calls) == 4

    done = rb.load_traces(path)
    assert set(done) == {(t.id, s) for t in subset for s in range(2)}
    calls.clear()
    episodes = rb.run_model_on_tasks("m", subset, 3, FakeEnv(), policy_factory=counting_factory, done=done, progress=False)
    assert calls == [(t.id, 2) for t in subset]        # solo la muestra nueva
    assert all(len(eps) == 3 for eps in episodes)


def test_trace_record_has_metadata_for_training_data(tasks):
    task = tasks[0]
    episode = FakeEnv().run_episode(task, rb.oracle_policy_factory(task, 0))
    rec = rb.trace_record("qwen:1.5b", "ollama", task, 0, episode, 1.23)
    for key in ("task_id", "prompt", "turns", "final_answer", "reward", "model", "sample", "difficulty",
                "template_id", "answer_type", "template_heldout", "elapsed_s"):
        assert key in rec
    assert rb.safe_name("qwen2.5-coder:1.5b-instruct-q4_K_M") == "qwen2.5-coder_1.5b-instruct-q4_K_M"


# ----------------------------------------------------------------------------- utilidades

def test_close_sql_tag_restores_stop_sequence():
    assert rb._close_sql_tag("Voy a consultar <sql>SELECT 1") == "Voy a consultar <sql>SELECT 1</sql>"
    assert rb._close_sql_tag("<sql>SELECT 1</sql>") == "<sql>SELECT 1</sql>"
    assert rb._close_sql_tag("<answer>3</answer>") == "<answer>3</answer>"


def test_episode_completion_excludes_prompt_but_keeps_observations():
    ep = Episode(task_id="t", prompt="p", turns=[
        {"role": "system", "content": "reglas"},
        {"role": "user", "content": "[TASK_ID: t] pregunta"},
        {"role": "assistant", "content": "<sql>SELECT 1</sql>"},
        {"role": "user", "content": "<result>1</result>"},
        {"role": "assistant", "content": "<answer>1</answer>"},
    ])
    assert rb.episode_completion(ep) == "<sql>SELECT 1</sql>\n<result>1</result>\n<answer>1</answer>"


def test_ollama_policy_requests_full_gpu_offload():
    pytest.importorskip("ollama")
    policy = rb.OllamaPolicy("dummy", num_gpu=-1, seed=0)
    assert policy.options["num_gpu"] == -1
    seeded = policy.with_seed(4)
    assert seeded.options["num_gpu"] == -1
    assert seeded.options["seed"] == 4
    assert rb.OllamaPolicy("dummy", num_gpu=0).options["num_gpu"] == 0


def test_k_values_never_exceed_samples():
    assert rb.k_values_for(8, 8) == [1, 2, 4, 8]
    assert rb.k_values_for(1, 8) == [1]
    assert rb.k_values_for(5, 5) == [1, 2, 4, 5]


def test_summary_csv(tasks, tmp_path):
    episodes = rb.run_model_on_tasks("oracle", tasks[:4], 1, FakeEnv(), policy_factory=rb.oracle_policy_factory, progress=False)
    report = {"models": {"oracle": rb.summarize_model(tasks[:4], episodes, k=1)}}
    path = tmp_path / "summary.csv"
    rb.write_summary_csv(path, report)
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("model,grouping,group")
    assert "difficulty" in text and "answer_type" in text and "block" in text


# ----------------------------------------------------------------------------- CLI de punta a punta

def test_main_end_to_end_with_oracle(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rb, "SQLEnvironment", FakeEnv)
    out = tmp_path / "baseline.json"
    common = ["--backend", "oracle", "--models", "oracle", "--k", "2", "--per-level", "2",
              "--traces-dir", str(tmp_path), "--out", str(out)]
    rb.main(common)

    report = json.loads(out.read_text(encoding="utf-8"))
    summary = report["models"]["oracle"]
    assert report["config"]["n_tasks"] == 8
    assert summary["overall"]["pass@1"] == 1.0 and summary["overall"]["pass@2"] == 1.0
    assert list(summary["by_difficulty"]) == ["easy", "medium", "hard", "expert", "all"]
    assert (tmp_path / "baseline_summary.csv").exists()
    traces = (tmp_path / "traces_oracle.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(traces) == 16

    # --resume con más muestras: solo agrega las nuevas.
    rb.main(common[:5] + ["3"] + common[6:] + ["--resume"])
    assert len((tmp_path / "traces_oracle.jsonl").read_text(encoding="utf-8").splitlines()) == 24

    # --summarize-only no ejecuta nada y reproduce el resumen.
    monkeypatch.setattr(rb, "SQLEnvironment", None)
    rb.main(common[:5] + ["3"] + common[6:] + ["--summarize-only"])
    again = json.loads(out.read_text(encoding="utf-8"))["models"]["oracle"]
    assert again["samples_per_task"] == 3 and again["overall"]["pass@1"] == 1.0
    assert "oracle" in capsys.readouterr().out
