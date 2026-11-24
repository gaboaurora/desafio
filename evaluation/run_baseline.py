"""
Línea base: ejecuta varios modelos (vía Ollama u otro backend) sobre dataset/test.jsonl y
reporta pass@1 y pass@k por nivel de dificultad.

Uso previsto:
    python -m evaluation.run_baseline --models qwen2.5-coder:1.5b-instruct-q4_K_M qwen2.5-coder:7b-instruct \
        --split test --k 8 --out outputs/baseline.json

Guarda además todas las trazas (Episode) en outputs/traces_<modelo>.jsonl para análisis cualitativo
y para alimentar training_data/.

Qué reporta por modelo (outputs/baseline.json y outputs/baseline_summary.csv):
  - pass@1 y pass@k (estimador insesgado de evaluation.metrics) por dificultad, por tipo de
    respuesta y por bloque del test (plantillas vistas en train vs. reservadas).
  - Calibración (punto 4 del enunciado): fracción de tareas que el modelo nunca resuelve (c = 0),
    siempre resuelve (c = n) o resuelve a veces (0 < c < n). Solo estas últimas dan señal a
    RLVR/GRPO/PPO: es la "zona aprendible" del modelo objetivo.
  - Uso de la herramienta (evaluation.metrics.tool_usage_stats): formato válido, SQL emitido,
    SQL ejecutable, turnos promedio.

Backends:
  - ollama : modelos abiertos servidos localmente (OLLAMA_HOST). Muestreo con temperatura > 0 y una
             semilla distinta por muestra, para que las k muestras de una tarea sean independientes.
             Por defecto envía las capas a CUDA (`num_gpu=-1`: todas las que quepan en VRAM).
  - oracle : responde con el gold_sql y la respuesta esperada. No es un modelo: sirve para validar
             el harness de punta a punta (su pass@1 debe ser 1.0).

Ejemplos:
    python -m evaluation.run_baseline --backend oracle --models oracle --k 1 --per-level 10
    python -m evaluation.run_baseline --models qwen2.5-coder:1.5b-instruct-q4_K_M --k 8 --per-level 25
    python -m evaluation.run_baseline --models qwen2.5-coder:1.5b-instruct-q4_K_M --k 8 --resume
    python -m evaluation.run_baseline --models qwen2.5-coder:1.5b-instruct-q4_K_M --summarize-only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from harness.environment import Episode, SQLEnvironment, Task
from harness.reward import get_task, load_task_registry, reward

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
LEVELS = ("easy", "medium", "hard", "expert")

Messages = List[Dict[str, str]]
Policy = Callable[[Messages], str]
PolicyFactory = Callable[[Task, int], Policy]


# --------------------------------------------------------------------------------------------------
# Tareas
# --------------------------------------------------------------------------------------------------

def load_tasks(split: str) -> List[Task]:
    path = DATASET_DIR / f"{split}.jsonl"
    with path.open(encoding="utf-8") as f:
        return [Task.from_dict(json.loads(line)) for line in f if line.strip()]


def subsample(tasks: Sequence[Task], per_level: Optional[int], limit: Optional[int], seed: int) -> List[Task]:
    """Submuestra estratificada por dificultad (determinista) para corridas más cortas."""
    selected = list(tasks)
    if per_level is not None:
        rng = random.Random(seed)
        by_level: Dict[str, List[Task]] = defaultdict(list)
        for t in tasks:
            by_level[t.difficulty].append(t)
        selected = []
        for level in sorted(by_level, key=lambda l: LEVELS.index(l) if l in LEVELS else len(LEVELS)):
            group = sorted(by_level[level], key=lambda t: t.id)
            selected += rng.sample(group, min(per_level, len(group)))
        selected.sort(key=lambda t: t.id)
    if limit is not None:
        selected = selected[:limit]
    return selected


# --------------------------------------------------------------------------------------------------
# Políticas (backends)
# --------------------------------------------------------------------------------------------------

class OllamaPolicy:
    """`policy(messages) -> completion` con un modelo servido por Ollama."""

    def __init__(
        self,
        model: str,
        host: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
        seed: Optional[int] = None,
        stop_after_sql: bool = True,
        retries: int = 3,
        num_gpu: Optional[int] = -1,
    ) -> None:
        import ollama  # import diferido: el backend oracle y los tests no lo necesitan

        self._ollama = ollama
        self.client = ollama.Client(host=host or os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        self.model = model
        self.stop_after_sql = stop_after_sql
        self.retries = retries
        self.options: Dict[str, Any] = {"temperature": temperature, "top_p": top_p, "num_predict": max_tokens}
        if seed is not None:
            self.options["seed"] = seed
        # -1: Ollama envía a la GPU todas las capas que quepan (CUDA/Metal). 0 fuerza CPU.
        if num_gpu is not None:
            self.options["num_gpu"] = num_gpu
        if stop_after_sql:
            # Al cerrar </sql> el modelo debe esperar la observación del harness, no inventarla.
            self.options["stop"] = ["</sql>", "</SQL>"]

    def with_seed(self, seed: int) -> "OllamaPolicy":
        clone = object.__new__(OllamaPolicy)
        clone.__dict__.update(self.__dict__)
        clone.options = {**self.options, "seed": seed}
        return clone

    def __call__(self, messages: Messages) -> str:
        payload = [{"role": _ollama_role(m.get("role", "user")), "content": m.get("content", "")} for m in messages]
        last_error: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                response = self.client.chat(model=self.model, messages=payload, options=self.options)
                return _close_sql_tag(_message_content(response)) if self.stop_after_sql else _message_content(response)
            except self._ollama.ResponseError as exc:
                if getattr(exc, "status_code", None) == 404:
                    raise RuntimeError(f"Ollama no tiene el modelo '{self.model}'. Ejecuta: ollama pull {self.model}") from exc
                last_error = exc
            except Exception as exc:  # conexión caída, timeout del servidor, etc.
                last_error = exc
            time.sleep(2 ** attempt)
        raise RuntimeError(f"Ollama falló {self.retries} veces con {self.model}: {last_error}") from last_error


def _ollama_role(role: str) -> str:
    return role if role in ("system", "user", "assistant", "tool") else "user"


def _message_content(response: Any) -> str:
    message = response["message"] if isinstance(response, dict) else response.message
    return (message["content"] if isinstance(message, dict) else message.content) or ""


def _close_sql_tag(text: str) -> str:
    """La parada en '</sql>' no incluye la etiqueta: se repone si quedó un <sql> abierto."""
    cleaned = re.sub(r"<\s*think\s*>.*?<\s*/\s*think\s*>", "", text, flags=re.IGNORECASE | re.DOTALL)
    opened = [m.start() for m in re.finditer(r"<\s*sql\s*>", cleaned, re.IGNORECASE)]
    closed = [m.start() for m in re.finditer(r"<\s*/\s*sql\s*>", cleaned, re.IGNORECASE)]
    ans_tags = [m.start() for m in re.finditer(r"<\s*/?\s*answer\s*>", cleaned, re.IGNORECASE)]
    if opened and (not closed or closed[-1] < opened[-1]):
        if not ans_tags or ans_tags[-1] < opened[-1]:
            return text + "</sql>"
    return text


def oracle_policy_factory(task: Task, sample: int) -> Policy:
    """Primero consulta con el gold_sql y luego responde lo esperado (control del harness)."""
    from evaluation.robustness import canonical_answer

    state = {"turn": 0}

    def policy(messages: Messages) -> str:
        state["turn"] += 1
        if state["turn"] == 1:
            return f"<sql>{task.gold_sql}</sql>"
        return f"<answer>{canonical_answer(task)}</answer>"

    return policy


def ollama_policy_factory(model: str, base_seed: int, **kwargs: Any) -> PolicyFactory:
    base = OllamaPolicy(model, **kwargs)

    def factory(task: Task, sample: int) -> Policy:
        # Una semilla por muestra: las k muestras de una tarea son distintas pero reproducibles.
        return base.with_seed(base_seed + sample)

    return factory


# --------------------------------------------------------------------------------------------------
# Ejecución
# --------------------------------------------------------------------------------------------------

def episode_completion(episode: Episode) -> str:
    """Todo lo que ocurre después del prompt (turnos del modelo y observaciones del harness)."""
    return "\n".join(t["content"] for t in episode.turns if t.get("role") not in ("system", "user") or _is_observation(t))


def _is_observation(turn: Dict[str, str]) -> bool:
    return turn.get("role") == "user" and turn.get("content", "").lstrip().lower().startswith("<result")


def _ensure_reward(episode: Episode) -> Episode:
    """Si el entorno no calculó el reward, se calcula con la interfaz común."""
    if episode.reward is None:
        if get_task(episode.task_id) is None:
            load_task_registry(*(str(DATASET_DIR / f"{s}.jsonl") for s in ("train", "eval", "test")))
        episode.reward = float(reward(episode.prompt, episode_completion(episode)))
    return episode


def run_model_on_tasks(
    model_name: str,
    tasks: List[Task],
    k: int,
    env: SQLEnvironment,
    *,
    policy_factory: Optional[PolicyFactory] = None,
    done: Optional[Dict[Tuple[str, int], Episode]] = None,
    on_episode: Optional[Callable[[Task, int, Episode, float], None]] = None,
    progress: bool = True,
) -> List[List[Episode]]:
    """Para cada tarea genera `k` episodios independientes (muestreo con temperatura > 0).

    `done` contiene episodios ya calculados ((task_id, muestra) -> Episode) y no se repiten.
    `on_episode` se llama tras cada episodio nuevo (p.ej. para escribir la traza en disco).
    """
    if policy_factory is None:
        policy_factory = ollama_policy_factory(model_name, base_seed=int(os.getenv("RANDOM_SEED", "2026")))
    done = done or {}
    pending = sum(1 for t in tasks for s in range(k) if (t.id, s) not in done)
    bar = _progress_bar(pending, model_name) if progress and pending else None

    all_episodes: List[List[Episode]] = []
    for task in tasks:
        episodes: List[Episode] = []
        for sample in range(k):
            if (task.id, sample) in done:
                episodes.append(done[(task.id, sample)])
                continue
            started = time.perf_counter()
            episode = _ensure_reward(env.run_episode(task, policy_factory(task, sample)))
            elapsed = time.perf_counter() - started
            episodes.append(episode)
            if on_episode:
                on_episode(task, sample, episode, elapsed)
            if bar:
                bar.update(1)
        all_episodes.append(episodes)
    if bar:
        bar.close()
    return all_episodes


def _progress_bar(total: int, desc: str):
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    return tqdm(total=total, desc=desc, unit="ep")


# --------------------------------------------------------------------------------------------------
# Trazas
# --------------------------------------------------------------------------------------------------

def safe_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def traces_path(traces_dir: Path, model: str) -> Path:
    return traces_dir / f"traces_{safe_name(model)}.jsonl"


def trace_record(model: str, backend: str, task: Task, sample: int, episode: Episode, elapsed: float) -> Dict[str, Any]:
    record = asdict(episode)
    record.update({
        "model": model,
        "backend": backend,
        "sample": sample,
        "difficulty": task.difficulty,
        "template_id": task.template_id,
        "answer_type": task.answer_type,
        "template_heldout": task.template_heldout,
        "elapsed_s": round(elapsed, 3),
    })
    return record


def load_traces(path: Path) -> Dict[Tuple[str, int], Episode]:
    """Episodios ya guardados en una traza, indexados por (task_id, muestra)."""
    done: Dict[Tuple[str, int], Episode] = {}
    if not path.exists():
        return done
    fields = set(Episode.__dataclass_fields__)
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            done[(rec["task_id"], int(rec["sample"]))] = Episode(**{k: rec[k] for k in fields if k in rec})
    return done


# --------------------------------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------------------------------

def is_correct(episode: Episode) -> bool:
    return episode.reward is not None and episode.reward >= 0.5


def task_results(tasks: Sequence[Task], episodes: Sequence[Sequence[Episode]]) -> List[Dict[str, Any]]:
    """[{task_id, difficulty, answer_type, block, n, c}] en el formato de aggregate_by_difficulty."""
    out = []
    for task, eps in zip(tasks, episodes):
        out.append({
            "task_id": task.id,
            "difficulty": task.difficulty,
            "answer_type": task.answer_type,
            "block": "heldout" if task.template_heldout else "seen",
            "n": len(eps),
            "c": sum(is_correct(e) for e in eps),
        })
    return out


def aggregate_by(results: List[Dict[str, Any]], key: str, k_values: List[int]) -> Dict[str, Dict[str, float]]:
    """aggregate_by_difficulty sobre otra columna (answer_type, block, ...), con nº de tareas."""
    from evaluation.metrics import aggregate_by_difficulty

    relabeled = [{**r, "difficulty": r[key]} for r in results]
    aggregated = aggregate_by_difficulty(relabeled, k_values)
    counts: Dict[str, int] = defaultdict(int)
    for r in relabeled:
        counts[r["difficulty"]] += 1
    counts["all"] = len(relabeled)
    return {g: {**{m: round(v, 4) for m, v in vals.items()}, "n_tasks": counts[g]} for g, vals in aggregated.items()}


def calibration(results: List[Dict[str, Any]], key: str = "difficulty") -> Dict[str, Dict[str, float]]:
    """Fracción de tareas nunca resueltas, en la zona aprendible (0 < c < n) y siempre resueltas."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        groups[r[key]].append(r)
    groups["all"] = list(results)
    out = {}
    for g, rs in groups.items():
        n = len(rs)
        never = sum(r["c"] == 0 for r in rs)
        always = sum(r["c"] == r["n"] for r in rs)
        out[g] = {
            "n_tasks": n,
            "never_solved": round(never / n, 4) if n else 0.0,
            "learnable": round((n - never - always) / n, 4) if n else 0.0,
            "always_solved": round(always / n, 4) if n else 0.0,
        }
    return out


def k_values_for(n: int, k: int) -> List[int]:
    candidates = {1, k} | {v for v in (2, 4, 8, 16) if v < k}
    return sorted(v for v in candidates if 1 <= v <= n)


def summarize_model(tasks: Sequence[Task], episodes: Sequence[Sequence[Episode]], k: int) -> Dict[str, Any]:
    from evaluation.metrics import tool_usage_stats

    results = task_results(tasks, episodes)
    n_min = min((r["n"] for r in results), default=0)
    if n_min == 0:
        return {"n_tasks": 0}
    ks = k_values_for(n_min, k)
    flat = [e for eps in episodes for e in eps]
    return {
        "n_tasks": len(results),
        "samples_per_task": n_min,
        "k_values": ks,
        "overall": aggregate_by(results, "difficulty", ks)["all"],
        "by_difficulty": _ordered_levels(aggregate_by(results, "difficulty", ks)),
        "by_answer_type": aggregate_by(results, "answer_type", ks),
        "by_block": aggregate_by(results, "block", ks),
        "calibration": _ordered_levels(calibration(results)),
        "tool_usage": {m: round(v, 4) for m, v in tool_usage_stats(flat).items()},
        "per_task": results,
    }


def _ordered_levels(d: Dict[str, Any]) -> Dict[str, Any]:
    order = [l for l in LEVELS if l in d] + sorted(l for l in d if l not in LEVELS and l != "all") + (["all"] if "all" in d else [])
    return {l: d[l] for l in order}


# --------------------------------------------------------------------------------------------------
# Reportes
# --------------------------------------------------------------------------------------------------

def write_summary_csv(path: Path, report: Dict[str, Any]) -> None:
    """Una fila por (modelo, agrupación, grupo): pass@k y calibración, lista para el informe."""
    rows: List[Dict[str, Any]] = []
    for model, summary in report["models"].items():
        if not summary.get("n_tasks"):
            continue
        for grouping in ("by_difficulty", "by_answer_type", "by_block"):
            for group, vals in summary[grouping].items():
                row = {"model": model, "grouping": grouping.replace("by_", ""), "group": group, **vals}
                if grouping == "by_difficulty":
                    row.update({f"calib_{m}": v for m, v in summary["calibration"].get(group, {}).items() if m != "n_tasks"})
                rows.append(row)
    if not rows:
        return
    columns: List[str] = []
    for row in rows:
        columns += [c for c in row if c not in columns]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def format_table(report: Dict[str, Any]) -> str:
    lines = []
    for model, s in report["models"].items():
        if not s.get("n_tasks"):
            lines.append(f"{model}: sin episodios")
            continue
        ks = s["k_values"]
        lines.append(f"\n{model}  ({s['n_tasks']} tareas × {s['samples_per_task']} muestras)")
        header = f"  {'grupo':<10}" + "".join(f"{'pass@' + str(k):>9}" for k in ks) + f"{'nunca':>8}{'zona':>8}{'siempre':>9}"
        lines.append(header)
        for group, vals in s["by_difficulty"].items():
            cal = s["calibration"].get(group, {})
            lines.append(
                f"  {group:<10}" + "".join(f"{vals[f'pass@{k}']:>9.3f}" for k in ks)
                + f"{cal.get('never_solved', 0):>8.2f}{cal.get('learnable', 0):>8.2f}{cal.get('always_solved', 0):>9.2f}"
            )
        for block, vals in s["by_block"].items():
            if block != "all":
                lines.append(f"  {block:<10}" + "".join(f"{vals[f'pass@{k}']:>9.3f}" for k in ks))
        usage = s["tool_usage"]
        lines.append(
            f"  formato válido {usage['valid_format_rate']:.2f} · SQL emitido {usage['sql_emitted_rate']:.2f}"
            f" · SQL ejecutable {usage['executable_sql_rate']:.2f} · turnos {usage['avg_turns']:.2f}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Línea base pass@1 / pass@k")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--k", type=int, default=8, help="muestras por tarea (n) y mayor k reportado")
    parser.add_argument("--out", default="outputs/baseline.json")
    parser.add_argument("--backend", choices=["ollama", "oracle"], default="ollama")
    parser.add_argument("--traces-dir", default="outputs")
    parser.add_argument("--per-level", type=int, default=None, help="submuestra: tareas por dificultad")
    parser.add_argument("--limit", type=int, default=None, help="máximo de tareas (después de --per-level)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=int(os.getenv("RANDOM_SEED", "2026")))
    parser.add_argument("--max-tool-turns", type=int, default=None)
    parser.add_argument("--host", default=None, help="OLLAMA_HOST (por defecto el del .env)")
    parser.add_argument(
        "--num-gpu",
        type=int,
        default=int(os.getenv("OLLAMA_NUM_GPU", "-1")),
        help="capas a enviar a la GPU (-1: todas las que quepan en CUDA, 0: solo CPU)",
    )
    parser.add_argument("--no-stop-after-sql", action="store_true", help="no cortar la generación en </sql>")
    parser.add_argument("--resume", action="store_true", help="reutiliza episodios ya guardados en las trazas")
    parser.add_argument("--summarize-only", action="store_true", help="solo recalcula métricas desde las trazas")
    args = parser.parse_args(argv)

    if args.k < 1:
        parser.error("--k debe ser >= 1")
    tasks = subsample(load_tasks(args.split), args.per_level, args.limit, args.seed)
    traces_dir = _resolve(args.traces_dir)
    traces_dir.mkdir(parents=True, exist_ok=True)
    env = None if args.summarize_only else SQLEnvironment(max_tool_turns=args.max_tool_turns)
    print(f"{len(tasks)} tareas de {args.split} · {args.k} muestras por tarea · backend {args.backend}")
    if args.backend == "ollama":
        if args.num_gpu == 0:
            print("Inferencia en CPU (num_gpu=0).")
        else:
            print(f"Offload a GPU: num_gpu={args.num_gpu} (-1 = todas las capas que quepan en CUDA).")

    report: Dict[str, Any] = {
        "config": {**vars(args), "n_tasks": len(tasks), "task_ids": [t.id for t in tasks]},
        "models": {},
    }
    for model in args.models:
        path = traces_path(traces_dir, model)
        done = load_traces(path) if (args.resume or args.summarize_only) else {}
        if args.summarize_only:
            missing = [(t.id, s) for t in tasks for s in range(args.k) if (t.id, s) not in done]
            if missing:
                print(f"{model}: faltan {len(missing)} episodios en {path}; se resumen solo las tareas completas.")
            complete = [t for t in tasks if all((t.id, s) in done for s in range(args.k))]
            episodes = [[done[(t.id, s)] for s in range(args.k)] for t in complete]
            report["models"][model] = summarize_model(complete, episodes, args.k)
            continue

        if not args.resume and path.exists():
            path.unlink()   # corrida nueva: no mezclar con trazas antiguas
        factory = _make_factory(args, model)
        with path.open("a", encoding="utf-8") as trace_file:
            def write(task: Task, sample: int, episode: Episode, elapsed: float) -> None:
                trace_file.write(json.dumps(trace_record(model, args.backend, task, sample, episode, elapsed), ensure_ascii=False) + "\n")
                trace_file.flush()

            try:
                episodes = run_model_on_tasks(model, tasks, args.k, env, policy_factory=factory, done=done, on_episode=write)
            except NotImplementedError as exc:
                raise SystemExit(
                    "El harness aún no está completo (SQLEnvironment.run_episode o reward lanzan "
                    f"NotImplementedError: {exc}). La línea base se puede correr cuando estén implementados."
                ) from exc
        report["models"][model] = summarize_model(tasks, episodes, args.k)
        print(f"Trazas de {model} en {path}")

    out = _resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = out.with_name(out.stem + "_summary.csv")
    write_summary_csv(csv_path, report)
    print(format_table(report))
    print(f"\nResumen en {out} y {csv_path}")


def _make_factory(args: argparse.Namespace, model: str) -> PolicyFactory:
    if args.backend == "oracle":
        return oracle_policy_factory
    return ollama_policy_factory(
        model,
        base_seed=args.seed,
        host=args.host,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_after_sql=not args.no_stop_after_sql,
        num_gpu=args.num_gpu,
    )


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


if __name__ == "__main__":
    main()
