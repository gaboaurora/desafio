"""Orquestador y monitor resiliente para ejecutar evaluaciones en Google Colab vía CLI.

Levanta o reutiliza una sesión GPU en Google Colab, clona el repositorio de forma
segura mediante un script ``GIT_ASKPASS`` efímero, mantiene el kernel de Jupyter en
estado activo (``BUSY``) para evitar cierres por inactividad y respalda las trazas
localmente de manera atómica cada cierto intervalo.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from typing import Dict, List, Optional, Sequence
from urllib.parse import urlparse, urlunparse

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = ROOT / "outputs"
DEFAULT_DATASET_DIR = ROOT / "dataset"
DEFAULT_MODEL = "qwen2.5-coder:1.5b-instruct-q4_K_M"
DEFAULT_GPUS = ["G4", "L4", "T4"]


def _load_local_env(env_path: Path) -> None:
    """Carga ``.env`` con python-dotenv si está instalado, o fallback stdlib."""
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
        return
    except ImportError:
        pass

    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_local_env(ROOT / ".env")


def resolve_colab_bin() -> str:
    """Resuelve la ruta al binario de ``google-colab-cli`` sin rutas absolutas fijas."""
    env_bin = os.getenv("COLAB_CLI_BIN", "").strip()
    if env_bin:
        resolved = shutil.which(env_bin) or (env_bin if Path(env_bin).is_file() else None)
        if resolved:
            return resolved
        raise FileNotFoundError(
            f"COLAB_CLI_BIN apunta a una ruta inexistente: {env_bin!r}"
        )

    on_path = shutil.which("colab")
    if on_path:
        return on_path

    exe_name = "colab.exe" if os.name == "nt" else "colab"
    local_bin = Path.home() / ".local" / "bin" / exe_name
    if local_bin.is_file():
        return str(local_bin)

    raise FileNotFoundError(
        "No se encontró el binario de Colab CLI ('colab'). Instálalo con "
        "'uv tool install google-colab-cli' o define COLAB_CLI_BIN."
    )


def safe_model_name(model: str) -> str:
    """Normaliza el nombre del modelo para coincidir con el formato de trazas."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def compute_total_episodes(
    split: str,
    k: int,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
) -> int:
    """Calcula el total de episodios (tareas del split × muestras k)."""
    if k <= 0:
        raise ValueError(f"k debe ser > 0 (recibido: {k})")

    dataset_path = dataset_dir / f"{split}.jsonl"
    if not dataset_path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de dataset: {dataset_path}")

    with dataset_path.open(encoding="utf-8") as f:
        n_tasks = sum(1 for line in f if line.strip())

    if n_tasks == 0:
        raise ValueError(f"El archivo de dataset está vacío: {dataset_path}")

    return n_tasks * k


def _normalize_repo_url(raw_url: str) -> str:
    """Normaliza una URL SSH o HTTPS de Git a HTTPS sin credenciales embebidas."""
    url = raw_url.strip()
    if not url:
        raise ValueError("La URL del repositorio no puede estar vacía.")

    # Convertir formato scp (git@github.com:owner/repo.git) a HTTPS limpio.
    scp_match = re.match(r"^[^@]+@([^:]+):(.+)$", url)
    if scp_match and "://" not in url:
        host, path = scp_match.groups()
        clean_path = "/" + path.lstrip("/")
        if not clean_path.endswith(".git"):
            clean_path += ".git"
        return f"https://{host}{clean_path}"

    parsed = urlparse(url)
    scheme = "https" if parsed.scheme in ("http", "https", "ssh", "") else parsed.scheme
    host = parsed.hostname or ""
    if not host:
        raise ValueError(f"URL de repositorio inválida: {raw_url!r}")

    port = f":{parsed.port}" if parsed.port and parsed.scheme != "ssh" else ""
    clean_path = parsed.path or ""
    if clean_path and not clean_path.endswith(".git"):
        clean_path += ".git"

    # Reconstruir omitiendo cualquier userinfo (usuario/token) previo.
    return urlunparse((scheme, f"{host}{port}", clean_path, "", "", ""))


def detect_repo_url() -> str:
    """Obtiene la URL HTTPS limpia del repositorio desde env o ``git remote``."""
    env_url = os.getenv("GITHUB_REPO_URL", "").strip()
    if env_url:
        return _normalize_repo_url(env_url)

    res = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if res.returncode != 0 or not res.stdout.strip():
        raise RuntimeError(
            "No se pudo detectar remote.origin.url. Define GITHUB_REPO_URL o usa --repo-url."
        )
    return _normalize_repo_url(res.stdout.strip())


def detect_git_branch() -> str:
    """Obtiene la rama actual desde env o ``git rev-parse``, con fallback ``dev``."""
    env_branch = os.getenv("GITHUB_BRANCH", "").strip()
    if env_branch:
        return env_branch

    res = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    branch = res.stdout.strip() if res.returncode == 0 else ""
    if not branch or branch == "HEAD":
        return "dev"
    return branch


def resolve_github_token() -> str:
    """Obtiene un token de GitHub desde variables de entorno o ``gh auth token``."""
    for var in ("GH_TOKEN", "GITHUB_TOKEN"):
        token = os.getenv(var, "").strip()
        if token:
            return token

    res = subprocess.run(
        ["gh", "auth", "token"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if res.returncode == 0 and res.stdout.strip():
        return res.stdout.strip()

    raise RuntimeError(
        "No se encontró token de GitHub. Define GH_TOKEN/GITHUB_TOKEN o ejecuta 'gh auth login'."
    )


def redact_secrets(text: str, secrets: Sequence[str]) -> str:
    """Enmascara tokens conocidos y credenciales en URLs antes de imprimir o guardar logs."""
    result = text
    for secret in sorted({s for s in secrets if s}, key=len, reverse=True):
        result = result.replace(secret, "***REDACTED***")
    # Defensa adicional por si alguna URL externa conserva userinfo o tokens en query string.
    result = re.sub(r"(https?://)[^/@\s]+@", r"\1***REDACTED***@", result)
    return re.sub(
        r"([?&](?:colab-runtime-proxy-token|access_token|token)=)[^&\s'\"]+",
        r"\1***REDACTED***",
        result,
    )


def remote_repo_dir(repo_url: str) -> str:
    """Deriva la ruta de clonación en ``/content`` a partir del nombre del repositorio."""
    path = urlparse(repo_url).path.rstrip("/")
    repo_slug = Path(path).stem or "workspace"
    return f"/content/{repo_slug}"


@dataclass(frozen=True)
class MonitorConfig:
    """Configuración inmutable para una sesión de ejecución y monitoreo en Colab."""

    colab_bin: str
    session: str
    gpus: List[str]
    model: str
    split: str
    k: int
    out_dir: Path
    backup_interval_sec: int
    repo_url: str
    branch: str
    total_episodes: int

    @property
    def traces_name(self) -> str:
        return f"traces_{safe_model_name(self.model)}.jsonl"

    @property
    def local_traces(self) -> Path:
        return self.out_dir / self.traces_name

    @property
    def progress_log(self) -> Path:
        return self.out_dir / "colab_progress.log"

    @property
    def remote_dir(self) -> str:
        return remote_repo_dir(self.repo_url)

    @property
    def remote_traces(self) -> str:
        return f"{self.remote_dir}/outputs/{self.traces_name}"


def parse_args(argv: Optional[Sequence[str]] = None) -> MonitorConfig:
    """Analiza los argumentos CLI y construye un ``MonitorConfig`` validado."""
    parser = argparse.ArgumentParser(
        description="Ejecuta evaluation.run_baseline y training_data.generate_sft en Google Colab."
    )
    parser.add_argument("--session", default="train-sql", help="Nombre de la sesión Colab")
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=list(DEFAULT_GPUS),
        help="Lista ordenada de GPUs a intentar (fallback automático)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("TARGET_MODEL_NAME", DEFAULT_MODEL),
        help="Nombre del modelo en Ollama",
    )
    parser.add_argument("--split", default="train", help="Split del dataset (train/eval/test)")
    parser.add_argument("--k", type=int, default=8, help="Muestras por tarea")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directorio local de salida para trazas y logs",
    )
    parser.add_argument(
        "--backup-interval-min",
        type=int,
        default=10,
        help="Intervalo de respaldo automático en minutos",
    )
    parser.add_argument(
        "--repo-url",
        default=None,
        help="URL del repositorio (por defecto se detecta desde GITHUB_REPO_URL o git origin)",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Rama a clonar (por defecto se detecta desde GITHUB_BRANCH o git HEAD)",
    )

    args = parser.parse_args(argv)
    if args.backup_interval_min <= 0:
        raise ValueError(
            f"--backup-interval-min debe ser > 0 (recibido: {args.backup_interval_min})"
        )

    repo_url = _normalize_repo_url(args.repo_url) if args.repo_url else detect_repo_url()
    branch = args.branch.strip() if args.branch else detect_git_branch()
    total_episodes = compute_total_episodes(args.split, args.k)

    return MonitorConfig(
        colab_bin=resolve_colab_bin(),
        session=args.session,
        gpus=[g.upper() for g in args.gpus],
        model=args.model,
        split=args.split,
        k=args.k,
        out_dir=args.out_dir.resolve(),
        backup_interval_sec=args.backup_interval_min * 60,
        repo_url=repo_url,
        branch=branch,
        total_episodes=total_episodes,
    )


_REMOTE_COMMON = r'''
import json, os, pathlib, re, subprocess, time, urllib.request

REPO_URL = os.environ["REPO_URL"]
REPO_BRANCH = os.environ["REPO_BRANCH"]
TARGET_MODEL = os.environ["TARGET_MODEL"]
SPLIT = os.environ["SPLIT"]
K_SAMPLES = os.environ["K_SAMPLES"]
TOTAL_EPISODES = int(os.environ["TOTAL_EPISODES"])

_safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", TARGET_MODEL)
_repo_name = os.path.splitext(REPO_URL.rstrip("/").split("/")[-1])[0] or "workspace"
REMOTE_DIR = f"/content/{_repo_name}"
TRACES_FILE = pathlib.Path(f"{REMOTE_DIR}/outputs/traces_{_safe_model}.jsonl")
STATUS_FILE = pathlib.Path("/tmp/train_status.json")
PID_FILE = pathlib.Path("/tmp/worker.pid")


def is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def worker_active() -> bool:
    if not PID_FILE.exists():
        return False
    try:
        return is_pid_running(int(PID_FILE.read_text(encoding="utf-8").strip()))
    except Exception:
        return False


def count_remote_traces() -> int:
    if not TRACES_FILE.exists():
        return 0
    with TRACES_FILE.open(encoding="utf-8", errors="replace") as f:
        return sum(1 for line in f if line.strip())


def clone_with_askpass() -> None:
    """Clona con GIT_ASKPASS efímero para no guardar el token en .git/config ni argv."""
    token = os.environ.get("GH_TOKEN", "")
    askpass_path = pathlib.Path("/tmp/.git_askpass.sh")
    askpass_script = (
        "#!/bin/sh\n"
        'case "$1" in\n'
        '  *[Uu]sername*) echo "x-access-token" ;;\n'
        '  *) printf "%s\\n" "$GH_TOKEN" ;;\n'
        "esac\n"
    )
    askpass_path.write_text(askpass_script, encoding="utf-8", newline="\n")
    os.chmod(askpass_path, 0o700)
    try:
        clone_env = os.environ.copy()
        clone_env["GIT_ASKPASS"] = str(askpass_path)
        clone_env["GIT_TERMINAL_PROMPT"] = "0"
        clone_env["GH_TOKEN"] = token
        subprocess.run(
            ["git", "clone", "-b", REPO_BRANCH, REPO_URL, REMOTE_DIR],
            env=clone_env,
            check=True,
        )
    finally:
        askpass_path.unlink(missing_ok=True)


def ensure_env() -> None:
    if not os.path.exists(REMOTE_DIR):
        clone_with_askpass()
    # Retirar GH_TOKEN del entorno tras clonar para no propagarlo a procesos hijos.
    os.environ.pop("GH_TOKEN", None)

    os.makedirs(f"{REMOTE_DIR}/outputs", exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", f"{REMOTE_DIR}/requirements.txt"],
        check=True,
    )

    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2)
    except Exception:
        subprocess.run(
            "apt-get update -qq && apt-get install -y -qq zstd pciutils lshw > /dev/null || true",
            shell=True,
        )
        subprocess.run(
            "curl -fsSL https://ollama.com/install.sh | sh > /dev/null 2>&1",
            shell=True,
            check=True,
        )
        subprocess.Popen(
            ["ollama", "serve"],
            stdin=subprocess.DEVNULL,
            stdout=open("/tmp/ollama.log", "a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        for _ in range(30):
            try:
                urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1)
                break
            except Exception:
                time.sleep(1)

    subprocess.run(
        ["ollama", "pull", TARGET_MODEL],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    if not worker_active():
        subprocess.run(
            ["ollama", "run", TARGET_MODEL, "Responde OK"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
'''

REMOTE_SETUP_SCRIPT = (
    "import sys\n"
    + _REMOTE_COMMON
    + r'''
ensure_env()
print(f"REMOTE_WORKER_ACTIVE:{1 if worker_active() else 0}", flush=True)
print(f"REMOTE_TRACE_COUNT:{count_remote_traces()}", flush=True)
print("SETUP_COMPLETED_OK", flush=True)
'''
)

# Se mantiene la celda en ejecución (estado BUSY) para impedir el cierre por inactividad
# de 20 minutos en Colab y reenganchar al mismo worker si hay cortes de red locales.
REMOTE_BUSY_WATCHER = (
    "import sys\n"
    + _REMOTE_COMMON
    + r'''
ensure_env()

worker_code = """import json, os, pathlib, subprocess

status_file = pathlib.Path("/tmp/train_status.json")
model = os.environ["TARGET_MODEL"]
split = os.environ["SPLIT"]
k_samples = os.environ["K_SAMPLES"]
remote_dir = os.environ["WORKER_REMOTE_DIR"]


def set_status(st, err=None):
    status_file.write_text(json.dumps({"state": st, "error": err}), encoding="utf-8")


try:
    set_status("running_baseline")
    rc1 = subprocess.call(
        [
            "python3", "-u", "-m", "evaluation.run_baseline",
            "--models", model,
            "--split", split,
            "--k", k_samples,
            "--resume",
        ],
        cwd=remote_dir,
    )
    if rc1 != 0:
        raise RuntimeError(f"run_baseline finalizó con código {rc1}")

    set_status("running_sft")
    rc2 = subprocess.call(
        ["python3", "-u", "-m", "training_data.generate_sft"],
        cwd=remote_dir,
    )
    if rc2 != 0:
        raise RuntimeError(f"generate_sft finalizó con código {rc2}")

    set_status("done")
except Exception as exc:
    set_status("error", str(exc))
    raise
"""

if worker_active():
    old_pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    print(f"Re-conectado al worker en ejecución (PID {old_pid})", flush=True)
else:
    pathlib.Path("/tmp/remote_worker.py").write_text(worker_code, encoding="utf-8")
    worker_env = os.environ.copy()
    worker_env["WORKER_REMOTE_DIR"] = REMOTE_DIR
    worker_env.pop("GH_TOKEN", None)
    proc = subprocess.Popen(
        ["python3", "-u", "/tmp/remote_worker.py"],
        stdin=subprocess.DEVNULL,
        stdout=open("/tmp/worker.log", "a", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        env=worker_env,
        start_new_session=True,
        close_fds=True,
    )
    PID_FILE.write_text(str(proc.pid), encoding="utf-8")
    print(f"Worker lanzado en GPU (PID {proc.pid})", flush=True)

start_t = time.time()
start_count = None

while True:
    time.sleep(15)
    count = 0
    last_task = ""
    if TRACES_FILE.exists():
        with TRACES_FILE.open(encoding="utf-8", errors="replace") as f:
            lines = [line for line in f if line.strip()]
            count = len(lines)
            if lines:
                try:
                    rec = json.loads(lines[-1])
                    last_task = f"{rec.get('task_id')}[s={rec.get('sample')}]"
                except Exception:
                    pass
    if start_count is None:
        start_count = count

    state = "running_baseline"
    err = None
    if STATUS_FILE.exists():
        try:
            st = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
            state = st.get("state", "running_baseline")
            err = st.get("error")
        except Exception:
            pass

    elapsed = max(1.0, time.time() - start_t)
    new_eps = max(0, count - start_count)
    rate = new_eps / elapsed if new_eps > 0 else 0.0
    remaining = max(0, TOTAL_EPISODES - count)
    if rate > 0:
        eta_sec = remaining / rate
        eta = f"{int(eta_sec // 3600)}h {int((eta_sec % 3600) // 60):02d}m"
    else:
        eta = "calculando..."

    print(f"PROGRESS_LINE|{count}|{last_task}|{rate:.2f}|{eta}|{state}", flush=True)

    if state == "done":
        print("=== WORKER_FINISHED_OK ===", flush=True)
        break
    if state == "error":
        print(f"=== WORKER_FAILED: {err} ===", flush=True)
        break
'''
)


class ColabMonitor:
    """Administra el ciclo de vida, respaldo atómico y monitoreo en Colab."""

    def __init__(self, cfg: MonitorConfig, secrets: Sequence[str] = ()) -> None:
        self.cfg = cfg
        self._secrets = list(secrets)
        self._log_lock = threading.Lock()
        self._stop_backup = threading.Event()
        self._last_backup_count = 0
        self._last_backup_time = "inicio"

    def log(self, msg: str) -> None:
        """Escribe una línea de log con marca temporal y secretos redactados."""
        ts = datetime.now().strftime("%H:%M:%S")
        safe_msg = redact_secrets(msg, self._secrets)
        line = f"[{ts}] {safe_msg}"
        with self._log_lock:
            try:
                print(line, flush=True)
            except UnicodeEncodeError:
                print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
            self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
            with self.cfg.progress_log.open("a", encoding="utf-8", errors="replace") as f:
                f.write(line + "\n")

    def _run_colab(
        self,
        *args: str,
        input_text: Optional[str] = None,
        timeout: int = 300,
        env_overrides: Optional[Dict[str, str]] = None,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)
        return subprocess.run(
            [self.cfg.colab_bin, *args],
            input=input_text,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )

    def _exec_env_flags(self, gh_token: Optional[str] = None) -> List[str]:
        env_map = {
            "REPO_URL": self.cfg.repo_url,
            "REPO_BRANCH": self.cfg.branch,
            "TARGET_MODEL": self.cfg.model,
            "SPLIT": self.cfg.split,
            "K_SAMPLES": str(self.cfg.k),
            "TOTAL_EPISODES": str(self.cfg.total_episodes),
        }
        if gh_token:
            env_map["GH_TOKEN"] = gh_token

        flags: List[str] = []
        for key, value in env_map.items():
            flags.extend(["--env", f"{key}={value}"])
        return flags

    def count_local_traces(self) -> int:
        if not self.cfg.local_traces.is_file():
            return 0
        text = self.cfg.local_traces.read_text(encoding="utf-8", errors="replace")
        return sum(1 for line in text.splitlines() if line.strip())

    def safe_download_traces(self) -> int:
        """Descarga trazas a un archivo temporal, valida JSON y reemplaza atómicamente."""
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.cfg.out_dir / f"{self.cfg.traces_name}.tmp"
        res = self._run_colab(
            "download", "-s", self.cfg.session, self.cfg.remote_traces, str(tmp_path),
            timeout=300,
        )
        if res.returncode != 0 or not tmp_path.is_file():
            err = redact_secrets((res.stderr or res.stdout or "").strip(), self._secrets)
            raise RuntimeError(f"colab download falló: {err}")

        lines = [
            line
            for line in tmp_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        ]
        if not lines:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError("Archivo descargado vacío")
        json.loads(lines[-1])

        old_count = self.count_local_traces()
        if len(lines) >= old_count:
            shutil.move(str(tmp_path), str(self.cfg.local_traces))
            self._last_backup_count = len(lines)
            self._last_backup_time = datetime.now().strftime("%H:%M:%S")
            return len(lines)

        tmp_path.unlink(missing_ok=True)
        return old_count

    def download_all_final(self) -> None:
        """Descarga trazas y artefactos finales generados en el entorno remoto."""
        self.log(f"Descargando archivos finales hacia {self.cfg.out_dir} ...")
        n = self.safe_download_traces()
        self.log(f"  -> {self.cfg.traces_name}: {n:,} episodios verificados.")

        for fname in ("sft.jsonl", "baseline.json", "baseline_summary.csv"):
            remote = f"{self.cfg.remote_dir}/outputs/{fname}"
            local = self.cfg.out_dir / fname
            self._run_colab(
                "download", "-s", self.cfg.session, remote, str(local), timeout=180
            )
            if local.is_file():
                self.log(f"  -> {fname}: descargado ({local.stat().st_size:,} bytes).")

    def _backup_loop(self) -> None:
        while not self._stop_backup.wait(self.cfg.backup_interval_sec):
            try:
                n = self.safe_download_traces()
                pct = (n / self.cfg.total_episodes) * 100.0
                mins = self.cfg.backup_interval_sec // 60
                self.log(
                    f"[AUTO-BACKUP {mins}m] Guardados {n:,} / {self.cfg.total_episodes:,} "
                    f"episodios ({pct:.1f}%) en {self.cfg.local_traces.name}"
                )
            except Exception as exc:
                self.log(
                    f"[AUTO-BACKUP AVISO] Fallo parcial ({exc}); se reintentará en el siguiente ciclo."
                )

    def ensure_session(self) -> None:
        """Reutiliza una sesión activa o asigna GPU en orden ``cfg.gpus``."""
        ls_res = self._run_colab("ls", "-s", self.cfg.session, "/content", timeout=30)
        if ls_res.returncode == 0:
            self.log(f"Sesión '{self.cfg.session}' ya activa. Verificando entorno remoto...")
            return

        if not self.cfg.gpus:
            raise ValueError("Debe especificarse al menos una GPU en --gpus.")

        for idx, gpu in enumerate(self.cfg.gpus):
            retries = "0" if idx < len(self.cfg.gpus) - 1 else "2"
            self.log(f"Solicitando GPU {gpu} en Google Colab (sesión '{self.cfg.session}')...")
            res = self._run_colab(
                "new", "-s", self.cfg.session, "--gpu", gpu,
                env_overrides={"COLAB_CLI_ASSIGN_RETRIES": retries},
                timeout=300,
            )
            if res.returncode == 0:
                self.log(f"Sesión '{self.cfg.session}' asignada y lista con GPU {gpu}.")
                return
            self.log(f"GPU {gpu} no disponible en este momento; intentando siguiente opción...")

        raise RuntimeError(f"No se pudo asignar ninguna GPU de {self.cfg.gpus}.")

    def prepare_remote_and_upload(self, gh_token: str) -> None:
        """Prepara repo, dependencias y Ollama en Colab y sube trazas previas si faltan."""
        self.log("Configurando repositorio, dependencias y Ollama en Colab...")
        setup_res = self._run_colab(
            "exec", "-s", self.cfg.session, "--timeout", "600",
            *self._exec_env_flags(gh_token=gh_token),
            input_text=REMOTE_SETUP_SCRIPT,
            timeout=900,
        )
        if setup_res.returncode != 0 or "SETUP_COMPLETED_OK" not in setup_res.stdout:
            err = redact_secrets(
                (setup_res.stderr or setup_res.stdout or "").strip(), self._secrets
            )
            raise RuntimeError(f"Error en setup remoto: {err}")

        remote_active = "REMOTE_WORKER_ACTIVE:1" in setup_res.stdout
        remote_count = 0
        count_match = re.search(r"REMOTE_TRACE_COUNT:(\d+)", setup_res.stdout)
        if count_match:
            remote_count = int(count_match.group(1))

        self.log(
            f"Entorno remoto listo (worker activo={remote_active}, "
            f"trazas remotas={remote_count:,})."
        )

        local_count = self.count_local_traces()
        self._last_backup_count = max(local_count, remote_count)
        self._last_backup_time = datetime.now().strftime("%H:%M:%S")

        if local_count > remote_count and not remote_active:
            self.log(
                f"Subiendo {local_count:,} episodios locales a Colab para reanudar (--resume)..."
            )
            up_res = self._run_colab(
                "upload", "-s", self.cfg.session,
                str(self.cfg.local_traces), self.cfg.remote_traces,
                timeout=300,
            )
            if up_res.returncode != 0:
                err = redact_secrets((up_res.stderr or up_res.stdout or "").strip(), self._secrets)
                raise RuntimeError(f"Fallo al subir trazas locales: {err}")
            self.log(f"Subida completada ({local_count:,} episodios listos).")

    def _stream_watcher(self) -> bool:
        """Mantiene una celda BUSY escuchando al worker remoto con reconexión automática."""
        consecutive_errors = 0
        while True:
            self.log("Conectando stream activo con el kernel de Colab (estado BUSY)...")
            proc = subprocess.Popen(
                [
                    self.cfg.colab_bin,
                    "exec",
                    "-s",
                    self.cfg.session,
                    "--timeout",
                    "180",
                    *self._exec_env_flags(),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(REMOTE_BUSY_WATCHER)
            proc.stdin.close()

            session_dead = False
            finished_ok = False
            for raw_line in proc.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                lower = line.lower()
                if "not found" in lower or "appears to be lost" in lower:
                    self.log(f"[ERROR CRITICO] Sesión cerrada en el servidor: {line}")
                    session_dead = True
                    break
                if line.startswith("PROGRESS_LINE|"):
                    consecutive_errors = 0
                    self._handle_progress_line(line)
                elif "=== WORKER_FINISHED_OK ===" in line:
                    self.log("Ejecución remota y generate_sft completados con éxito.")
                    finished_ok = True
                    break
                elif "=== WORKER_FAILED:" in line:
                    self.log(f"[ERROR EN WORKER] {line}")
                    session_dead = True
                    break
                else:
                    self.log(f"[Colab] {line}")

            proc.wait()
            if finished_ok:
                return True
            if session_dead:
                return False

            consecutive_errors += 1
            if consecutive_errors > 10:
                self.log("[ERROR] Demasiados fallos consecutivos de reconexión.")
                return False

            self.log(
                f"[AVISO] Stream desconectado (intento {consecutive_errors}/10). "
                "Reintentando en 15s..."
            )
            time.sleep(15)

    def _handle_progress_line(self, line: str) -> None:
        parts = line.split("|")
        if len(parts) < 6:
            return
        cnt = int(parts[1])
        last_task, rate, eta, state = parts[2], parts[3], parts[4], parts[5]
        pct = (cnt / self.cfg.total_episodes) * 100.0
        self.log(
            f"Progreso: {cnt:,} / {self.cfg.total_episodes:,} ({pct:.1f}%) | "
            f"Tarea: {last_task} | Vel: {rate} ep/s | ETA: {eta} | Estado: {state} | "
            f"Backup local: {self._last_backup_count:,} ep ({self._last_backup_time})"
        )

    def run(self, gh_token: str) -> int:
        """Ejecuta todo el flujo de provisión, monitoreo, respaldo y cierre limpio."""
        finished_ok = False
        try:
            mins = self.cfg.backup_interval_sec // 60
            self.log("=" * 70)
            self.log(f" MONITOR COLAB ({self.cfg.model} · split={self.cfg.split} · k={self.cfg.k})")
            self.log(
                f" - Trazas locales previas: {self.count_local_traces():,} / "
                f"{self.cfg.total_episodes:,} episodios"
            )
            self.log(f" - Auto-backup atómico cada {mins} min en {self.cfg.out_dir}")
            self.log("=" * 70)

            self.ensure_session()
            self.prepare_remote_and_upload(gh_token)

            bk_thread = threading.Thread(target=self._backup_loop, daemon=True)
            bk_thread.start()

            finished_ok = self._stream_watcher()
            return 0 if finished_ok else 1

        except KeyboardInterrupt:
            self.log("Interrupción manual recibida. Guardando respaldo parcial...")
            return 130
        except Exception as exc:
            self.log(f"[EXCEPCION] {exc}\n{traceback.format_exc()}")
            return 1
        finally:
            self._stop_backup.set()
            self.log("Ejecutando descarga final de respaldo antes de finalizar...")
            try:
                if finished_ok:
                    self.download_all_final()
                else:
                    n = self.safe_download_traces()
                    self.log(f"Respaldo parcial guardado en disco local: {n:,} episodios.")
            except Exception as exc:
                self.log(f"Aviso en respaldo final: {exc}")

            if finished_ok:
                self.log(f"Deteniendo sesión '{self.cfg.session}' en Colab (colab stop)...")
                self._run_colab("stop", "-s", self.cfg.session, timeout=120)
                self.log("Sesión terminada.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg = parse_args(argv)
    gh_token = resolve_github_token()
    monitor = ColabMonitor(cfg, secrets=[gh_token])
    return monitor.run(gh_token)


if __name__ == "__main__":
    sys.exit(main())
