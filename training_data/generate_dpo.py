"""
Generación de pares de preferencia DPO a partir de trazas del modelo.

Para cada tarea de train con al menos un episodio correcto (reward 1.0) y uno
incorrecto (reward 0.0) se construyen como máximo dos pares. Cada par comparte
el prefijo y contrasta solo el mensaje del asistente:

  - turn "sql": el prompt es el system y la pregunta; chosen/rejected son el
    primer mensaje del asistente.
  - turn "answer": solo si ambos episodios recibieron el mismo <result>. El
    prompt llega hasta esa observación; chosen/rejected son el <answer>.

Los rechazos se ordenan de más a menos informativos: wrong_answer, sql_error,
multiple_answers, invalid_format, no_answer. Dentro de una categoría se prefiere
el SQL más parecido al del episodio correcto. Si hay categorías distintas, los
dos cupos se usan en categorías distintas.

Uso previsto:
    python -m training_data.generate_dpo
    python -m training_data.generate_dpo --traces outputs/traces_qwen2.5-coder_1.5b-instruct-q4_K_M.jsonl --out outputs/dpo.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from harness.environment import Episode
from harness.parser import clean_sql, parse_completion
from harness.reward import _is_coherent, get_task
from harness.sql_executor import SQLExecutor
from harness.verifier import verify

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACES_FILE = "outputs/traces_qwen2.5-coder_1.5b-instruct-q4_K_M.jsonl"
DEFAULT_TASKS_FILE = "dataset/train.jsonl"
DEFAULT_OUT = "outputs/dpo.jsonl"

FAILURE_CATEGORIES = (
    "wrong_answer",
    "sql_error",
    "multiple_answers",
    "invalid_format",
    "no_answer",
)
CATEGORY_RANK = {name: i for i, name in enumerate(FAILURE_CATEGORIES)}
_STRUCTURAL_PREFIXES = ("nested_tag", "closing_without_opening", "mismatched_closing", "unclosed_tag")
_STRUCTURAL_EXACT = {"empty_sql_block", "empty_answer", "empty_completion"}

_executor: Optional[SQLExecutor] = None


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _get_executor() -> SQLExecutor:
    global _executor
    if _executor is None:
        _executor = SQLExecutor(ROOT / "db" / "hospital.sqlite")
    return _executor


def _assistant_messages(episode: Episode) -> List[str]:
    return [t.get("content", "") for t in episode.turns if t.get("role") == "assistant"]


def _message(role: str, content: str) -> Dict[str, str]:
    return {"role": role, "content": content}


def episode_sql(episode: Episode) -> str:
    parsed = parse_completion("\n".join(_assistant_messages(episode)))
    if not parsed.sql_queries:
        return ""
    return clean_sql(parsed.sql_queries[-1])


def sql_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def _is_structural(error: str) -> bool:
    return error in _STRUCTURAL_EXACT or error.startswith(_STRUCTURAL_PREFIXES)


def _inspect_failure(episode: Episode) -> Optional[str]:
    """Categoría del rechazo, o None si el SQL y la respuesta ya son correctos."""
    parsed = parse_completion("\n".join(_assistant_messages(episode)))
    errors = set(parsed.errors)

    if parsed.sql_queries:
        result = _get_executor().execute(parsed.sql_queries[-1])
        if not result.ok:
            return "sql_error"
        if parsed.final_answer:
            task = get_task(episode.task_id)
            if task is None:
                return "wrong_answer"
            matches = verify(
                parsed.final_answer,
                task.expected_result,
                task.answer_type,
                ordered=task.ordered,
            ).is_correct
            coherent = _is_coherent(parsed.final_answer, result, task)
            if matches and coherent:
                return None
            return "wrong_answer"

    if "multiple_answer_blocks" in errors:
        return "multiple_answers"
    if any(_is_structural(err) for err in errors):
        return "invalid_format"
    return "no_answer"


def categorize_failure(episode: Episode) -> str:
    """invalid_format | sql_error | wrong_answer | multiple_answers | no_answer.

    Si un episodio cae en varias categorías, devuelve la más informativa.
    """
    category = _inspect_failure(episode)
    if category is None:
        raise ValueError(f"{episode.task_id} no es un rechazo: la respuesta ya es correcta")
    return category


def _sql_decision(episode: Episode) -> Optional[Tuple[List[Dict[str, str]], str]]:
    prefix: List[Dict[str, str]] = []
    for turn in episode.turns:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "assistant":
            if not content.strip():
                return None
            return prefix, content
        prefix.append(_message(role, content))
    return None


def _answer_decision(episode: Episode) -> Optional[Tuple[List[Dict[str, str]], str, str]]:
    """(prompt hasta el <result>, mensaje con <answer>, texto normalizado del result)."""
    prefix: List[Dict[str, str]] = []
    pending_result: Optional[str] = None
    pending_prefix: Optional[List[Dict[str, str]]] = None
    found: Optional[Tuple[List[Dict[str, str]], str, str]] = None
    for turn in episode.turns:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "assistant":
            if pending_result is not None and "<answer" in content.lower():
                found = (pending_prefix or [], content, pending_result)
                pending_result = None
            continue
        prefix.append(_message(role, content))
        if "<result" in content.lower():
            pending_result = normalize_text(content)
            pending_prefix = list(prefix)
    return found


def _candidate(
    task_id: str,
    difficulty: str,
    turn: str,
    category: str,
    prompt: Sequence[Dict[str, str]],
    chosen: str,
    rejected: str,
    similarity: float,
) -> Dict[str, object]:
    return {
        "task_id": task_id,
        "difficulty": difficulty,
        "turn": turn,
        "failure_category": category,
        "prompt": [dict(message) for message in prompt],
        "chosen": chosen,
        "rejected": rejected,
        "_similarity": similarity,
    }


def _select(candidates: List[Dict[str, object]], max_pairs: int) -> List[Dict[str, object]]:
    """Hasta `max_pairs`, una categoría distinta primero y el SQL más cercano dentro de ella."""
    ordered = sorted(
        candidates,
        key=lambda pair: (
            CATEGORY_RANK[str(pair["failure_category"])],
            -float(pair["_similarity"]),
            0 if pair["turn"] == "sql" else 1,
        ),
    )
    selected: List[Dict[str, object]] = []
    seen_categories: Set[str] = set()
    same_category: List[Dict[str, object]] = []
    for pair in ordered:
        category = str(pair["failure_category"])
        if category in seen_categories:
            same_category.append(pair)
            continue
        selected.append(pair)
        seen_categories.add(category)
        if len(selected) >= max_pairs:
            return selected
    for pair in same_category:
        selected.append(pair)
        if len(selected) >= max_pairs:
            break
    return selected


def _pairs_for_task(
    task_id: str,
    episodes: Sequence[Episode],
    max_pairs: int,
) -> List[Dict[str, object]]:
    successes = [ep for ep in episodes if (ep.reward or 0.0) >= 1.0 and ep.final_answer]
    failures = [ep for ep in episodes if (ep.reward or 0.0) < 1.0]
    if not successes or not failures:
        return []

    task = get_task(task_id)
    difficulty = task.difficulty if task is not None else "unknown"
    categories = {id(ep): _inspect_failure(ep) for ep in failures}
    good_sql = {id(ep): episode_sql(ep) for ep in successes}
    bad_sql = {id(ep): episode_sql(ep) for ep in failures}

    candidates: List[Dict[str, object]] = []
    seen: Set[Tuple[str, str, str]] = set()

    def add(turn: str, prompt: Sequence[Dict[str, str]], chosen: str, rejected: str, failure: Episode, success: Episode) -> None:
        category = categories[id(failure)]
        if category is None:
            return
        key = (turn, normalize_text(chosen), normalize_text(rejected))
        if not key[1] or not key[2] or key[1] == key[2] or key in seen:
            return
        seen.add(key)
        candidates.append(
            _candidate(
                task_id,
                difficulty,
                turn,
                category,
                prompt,
                chosen,
                rejected,
                sql_similarity(good_sql[id(success)], bad_sql[id(failure)]),
            )
        )

    for success in successes:
        sql_choice = _sql_decision(success)
        answer_choice = _answer_decision(success)
        for failure in failures:
            if sql_choice is not None:
                rejected_sql = _sql_decision(failure)
                if rejected_sql is not None:
                    add("sql", sql_choice[0], sql_choice[1], rejected_sql[1], failure, success)
            if answer_choice is not None:
                rejected_answer = _answer_decision(failure)
                if rejected_answer is not None and rejected_answer[2] == answer_choice[2]:
                    add("answer", answer_choice[0], answer_choice[1], rejected_answer[1], failure, success)

    chosen_pairs = _select(candidates, max_pairs)
    for pair in chosen_pairs:
        pair.pop("_similarity", None)
    return chosen_pairs


def build_pairs(
    episodes_by_task: Dict[str, List[Episode]],
    max_pairs_per_task: int = 2,
) -> List[dict]:
    pairs: List[dict] = []
    for task_id in sorted(episodes_by_task):
        pairs.extend(_pairs_for_task(task_id, episodes_by_task[task_id], max_pairs_per_task))
    return pairs


def load_task_ids(path: str | Path) -> Set[str]:
    ids: Set[str] = set()
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = ROOT / file_path
    with file_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.add(json.loads(line)["id"])
    return ids


def load_episodes_from_traces(trace_files: Iterable[str]) -> List[Episode]:
    episodes: List[Episode] = []
    fields = set(Episode.__dataclass_fields__)
    for trace in trace_files:
        path = Path(trace)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de trazas: {path}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                episodes.append(Episode(**{key: data[key] for key in fields if key in data}))
    return episodes


def pairs_from_episodes(
    episodes: Sequence[Episode],
    allowed_ids: Set[str],
    max_pairs_per_task: int = 2,
) -> List[dict]:
    grouped: Dict[str, List[Episode]] = defaultdict(list)
    for episode in episodes:
        if episode.task_id in allowed_ids:
            grouped[episode.task_id].append(episode)
    return build_pairs(grouped, max_pairs_per_task)


def _resolve(path: str) -> Path:
    file_path = Path(path)
    return file_path if file_path.is_absolute() else ROOT / file_path


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Genera pares DPO desde trazas del modelo")
    parser.add_argument("--traces", nargs="+", default=[DEFAULT_TRACES_FILE])
    parser.add_argument(
        "--tasks",
        default=DEFAULT_TASKS_FILE,
        help="JSONL de tareas permitidas (por defecto dataset/train.jsonl)",
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--max-pairs-per-task", type=int, default=2)
    args = parser.parse_args(argv)

    if args.max_pairs_per_task < 1:
        parser.error("--max-pairs-per-task debe ser >= 1")

    episodes = load_episodes_from_traces(args.traces)
    allowed = load_task_ids(args.tasks)
    in_split = [ep for ep in episodes if ep.task_id in allowed]
    learnable = 0
    by_task: Dict[str, List[Episode]] = defaultdict(list)
    for episode in in_split:
        by_task[episode.task_id].append(episode)
    for group in by_task.values():
        has_success = any((ep.reward or 0.0) >= 1.0 and ep.final_answer for ep in group)
        has_failure = any((ep.reward or 0.0) < 1.0 for ep in group)
        if has_success and has_failure:
            learnable += 1

    pairs = build_pairs(by_task, args.max_pairs_per_task)
    print(f"Cargadas {len(episodes)} trazas desde: {args.traces}")
    print(f"Tareas de {args.tasks} con trazas: {len(by_task)}")
    print(f"Tareas en zona aprendible: {learnable}")
    print(f"Pares DPO: {len(pairs)}")
    if pairs:
        print("Por corte:", dict(Counter(pair["turn"] for pair in pairs)))
        print("Por rechazo:", dict(Counter(pair["failure_category"] for pair in pairs)))

    out_path = _resolve(args.out)
    if not pairs:
        outside = len({ep.task_id for ep in episodes} - allowed)
        print(
            "No se escribieron pares. Las trazas no tienen tareas aprendibles dentro de "
            f"{args.tasks}."
            + (f" Hay {outside} tareas de las trazas fuera de ese archivo." if outside else "")
        )
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Pares DPO exportados en: {out_path}")


if __name__ == "__main__":
    main()
