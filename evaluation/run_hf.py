"""
Evaluación de modelos HuggingFace (+ adaptadores LoRA de SFT / DPO / GRPO) sobre el harness SQL.

Reutiliza las funciones de ejecución, trazas, reanudación y métricas de `evaluation.run_baseline`
sin duplicar lógica ni modificar la línea base.

Uso:
    python -m evaluation.run_hf --name sft_lora --adapter checkpoints/sft_lora \
        --split test --k 8 --out outputs/sft_lora_test/sft_lora.json \
        --traces-dir outputs/sft_lora_test --resume
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.run_baseline import (
    Messages,
    Policy,
    PolicyFactory,
    _close_sql_tag,
    _resolve,
    format_table,
    load_tasks,
    load_traces,
    run_model_on_tasks,
    subsample,
    summarize_model,
    trace_record,
    traces_path,
    write_summary_csv,
)
from harness.environment import Episode, SQLEnvironment, Task


def resolve_adapter(adapter: str) -> Path:
    """Resuelve la carpeta del adaptador LoRA (si apunta al directorio padre, toma el último `final_*`)."""
    p = _resolve(adapter)
    if (p / "adapter_config.json").is_file():
        return p
    finals = sorted(d for d in p.glob("final_*") if (d / "adapter_config.json").is_file())
    if finals:
        return finals[-1]
    raise FileNotFoundError(
        f"No se encontró adapter_config.json en '{p}' ni en subcarpetas 'final_*'."
    )


class HFPolicy:
    """`policy(messages) -> completion` con Transformers + adaptador LoRA fusionado opcional."""

    def __init__(
        self,
        base_model: str,
        adapter: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
        stop_after_sql: bool = True,
    ) -> None:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            StoppingCriteria,
            StoppingCriteriaList,
        )

        self._torch = torch
        self._StoppingCriteriaList = StoppingCriteriaList
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop_after_sql = stop_after_sql

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        dtype = (
            torch.bfloat16 if use_bf16
            else (torch.float16 if torch.cuda.is_available() else torch.float32)
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if adapter:
            from peft import PeftModel

            adapter_dir = resolve_adapter(adapter)
            print(f"Cargando adaptador LoRA desde {adapter_dir}")
            model = PeftModel.from_pretrained(model, str(adapter_dir))
            model = model.merge_and_unload()
        model.eval()
        self.model = model

        # Detección rápida de cierre </sql>: solo decodifica la cola cuando aparece un token con '>'.
        gt_ids = {tid for tok, tid in self.tokenizer.get_vocab().items() if ">" in tok}
        tok_ref = self.tokenizer

        class _StopOnSqlClose(StoppingCriteria):
            def __init__(self, prompt_len: int) -> None:
                self.prompt_len = prompt_len

            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                if input_ids.shape[1] <= self.prompt_len:
                    return False
                if int(input_ids[0, -1]) not in gt_ids:
                    return False
                start = max(self.prompt_len, int(input_ids.shape[1]) - 6)
                tail = tok_ref.decode(input_ids[0, start:], skip_special_tokens=True)
                return "</sql>" in tail.lower()

        self._stop_cls = _StopOnSqlClose

    def __call__(self, messages: Messages) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = int(inputs["input_ids"].shape[1])
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p
        if self.stop_after_sql:
            gen_kwargs["stopping_criteria"] = self._StoppingCriteriaList(
                [self._stop_cls(prompt_len)]
            )

        with self._torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        text = self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        return _close_sql_tag(text) if self.stop_after_sql else text


def hf_policy_factory(base_model: str, base_seed: int, **kwargs: Any) -> PolicyFactory:
    policy = HFPolicy(base_model, **kwargs)

    def factory(task: Task, sample: int) -> Policy:
        seed = base_seed + sample
        policy._torch.manual_seed(seed)
        if policy._torch.cuda.is_available():
            policy._torch.cuda.manual_seed_all(seed)
        return policy

    return factory


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluación pass@1 / pass@k con HuggingFace + LoRA")
    parser.add_argument("--name", required=True, help="nombre del modelo en reportes y trazas (ej. sft_lora)")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--adapter", default=None, help="carpeta del adaptador LoRA (opcional)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--out", default=None)
    parser.add_argument("--traces-dir", default=None)
    parser.add_argument("--per-level", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=int(os.getenv("RANDOM_SEED", "2026")))
    parser.add_argument("--max-tool-turns", type=int, default=None)
    parser.add_argument("--no-stop-after-sql", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)

    if args.k < 1:
        parser.error("--k debe ser >= 1")

    default_dir = f"outputs/{args.name}_{args.split}"
    out = _resolve(args.out or f"{default_dir}/{args.name}.json")
    traces_dir = _resolve(args.traces_dir or default_dir)
    traces_dir.mkdir(parents=True, exist_ok=True)

    tasks = subsample(load_tasks(args.split), args.per_level, args.limit, args.seed)
    env = None if args.summarize_only else SQLEnvironment(max_tool_turns=args.max_tool_turns)
    print(f"{len(tasks)} tareas de {args.split} · {args.k} muestras por tarea · {args.name} (HF)")

    path = traces_path(traces_dir, args.name)
    done = load_traces(path) if (args.resume or args.summarize_only) else {}

    if args.summarize_only:
        complete = [t for t in tasks if all((t.id, s) in done for s in range(args.k))]
        episodes = [[done[(t.id, s)] for s in range(args.k)] for t in complete]
        summary = summarize_model(complete, episodes, args.k)
    else:
        if not args.resume and path.exists():
            path.unlink()
        factory = hf_policy_factory(
            args.base_model,
            base_seed=args.seed,
            adapter=args.adapter,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            stop_after_sql=not args.no_stop_after_sql,
        )
        with path.open("a", encoding="utf-8") as trace_file:
            def write(task: Task, sample: int, episode: Episode, elapsed: float) -> None:
                rec = trace_record(args.name, "hf", task, sample, episode, elapsed)
                trace_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                trace_file.flush()

            episodes = run_model_on_tasks(
                args.name, tasks, args.k, env, policy_factory=factory, done=done, on_episode=write
            )
        summary = summarize_model(tasks, episodes, args.k)
        print(f"Trazas de {args.name} en {path}")

    report: Dict[str, Any] = {
        "config": {**vars(args), "n_tasks": len(tasks), "task_ids": [t.id for t in tasks]},
        "models": {args.name: summary},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = out.with_name(out.stem + "_summary.csv")
    write_summary_csv(csv_path, report)
    print(format_table(report))
    print(f"\nResumen en {out} y {csv_path}")


if __name__ == "__main__":
    main()
