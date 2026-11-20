"""Run the Indic ASR pipeline on Modal (https://modal.com). Thin wrapper: every stage just calls
scripts/run_all.sh / scripts/smoke_test.sh inside a container; all data, checkpoints, logs and
results live on ONE persistent Volume so any stage can be re-launched and resumes.

One-time setup (human):
    pip install modal && modal setup
    modal secret create huggingface HF_TOKEN=hf_xxx        # + accept dataset terms on huggingface.co
Usage:
    modal run modal_app.py --stage smoke
    modal run modal_app.py --stage prepare --profile quick
    modal run --detach modal_app.py --stage train_whisper --profile full
    modal run --detach modal_app.py --stage all --profile full           # whole pipeline, server-side
    modal run modal_app.py --stage status
    modal volume get indic-asr-vol results ./results_modal --force       # download results
Options: --overrides "W_STEPS=4000 C_STEPS=8000 TRAIN_HOURS=60"   --indicconformer
GPU: env INDIC_GPU (default A100-40GB; e.g. L40S, A10G, H100). Set it on the command line:
    INDIC_GPU=L40S modal run --detach modal_app.py --stage train_ctc --profile full
"""
from __future__ import annotations

import os
import shlex
import subprocess
import threading
from pathlib import Path

import modal

GPU = os.environ.get("INDIC_GPU", "A100-40GB")
APP_DIR = "/root/indic-asr"
VOL_DIR = "/vol"
VOLUME_NAME = "indic-asr-vol"
SHARED_DIRS = ["data", "data_quick", "runs", "runs_quick", "results", "results_quick", "logs"]

app = modal.App("indic-asr")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")  # must contain HF_TOKEN

_here = Path(__file__).parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg", "git")
    .pip_install_from_requirements(str(_here / "requirements.txt"))
    .env({"HF_HOME": f"{VOL_DIR}/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(str(_here), remote_path=APP_DIR,
                   ignore=["data*", "runs*", "results*", "logs", ".venv", "venv", "__pycache__", "*.zip", ".git"])
)

COMMON = dict(image=image, volumes={VOL_DIR: vol}, secrets=[hf_secret])
GPU_KW = dict(gpu=GPU, cpu=8, memory=32768, timeout=24 * 3600, **COMMON)
CPU_KW = dict(cpu=4, memory=16384, timeout=24 * 3600, **COMMON)


def _setup() -> None:
    """Point the project's data/runs/results/logs dirs at the persistent volume."""
    for d in SHARED_DIRS:
        target = Path(VOL_DIR) / d
        target.mkdir(parents=True, exist_ok=True)
        link = Path(APP_DIR) / d
        if link.is_symlink() or link.exists():
            if link.is_symlink():
                link.unlink()
            else:
                import shutil
                shutil.rmtree(link)
        link.symlink_to(target, target_is_directory=True)


def _parse_overrides(s: str) -> dict:
    out = {}
    for tok in shlex.split(s or ""):
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def _run(cmd: list[str], env_extra: dict | None = None) -> None:
    vol.reload()
    _setup()
    env = {**os.environ, **(env_extra or {})}
    cv = Path(VOL_DIR) / "commonvoice" / "hi"
    if "COMMONVOICE_DIR" not in env and (cv / "test.tsv").exists():
        env["COMMONVOICE_DIR"] = str(cv)
    stop = threading.Event()

    def committer():  # persist checkpoints/logs periodically so a crash/timeout loses little
        while not stop.wait(300):
            try:
                vol.commit()
            except Exception as e:  # noqa: BLE001
                print("[modal_app] periodic volume commit failed:", e, flush=True)

    t = threading.Thread(target=committer, daemon=True)
    t.start()
    print("[modal_app] running:", " ".join(cmd), flush=True)
    try:
        rc = subprocess.run(cmd, cwd=APP_DIR, env=env).returncode
    finally:
        stop.set()
        vol.commit()
    if rc != 0:
        raise RuntimeError(f"command failed with exit code {rc}: {' '.join(cmd)}")


@app.function(**CPU_KW)
def smoke(overrides: str = "") -> None:
    """Offline end-to-end test on CPU (synthetic audio + tiny random models). No volume writes."""
    env = {"SMOKE_DIR": "/tmp/indic_asr_smoke", **_parse_overrides(overrides)}
    subprocess.run(["bash", "scripts/smoke_test.sh"], cwd=APP_DIR, env={**os.environ, **env}, check=True)


@app.function(**CPU_KW)
def prepare(profile: str = "quick", overrides: str = "") -> None:
    _run(["bash", "scripts/run_all.sh", profile, "prepare"], {"ALLOW_CPU": "1", **_parse_overrides(overrides)})


@app.function(**GPU_KW)
def train_whisper(profile: str = "quick", overrides: str = "") -> None:
    _run(["bash", "scripts/run_all.sh", profile, "train_whisper"], _parse_overrides(overrides))


@app.function(**GPU_KW)
def train_ctc(profile: str = "quick", overrides: str = "") -> None:
    _run(["bash", "scripts/run_all.sh", profile, "train_ctc"], _parse_overrides(overrides))


@app.function(**GPU_KW)
def evaluate(profile: str = "quick", overrides: str = "", indicconformer: bool = False) -> None:
    env = _parse_overrides(overrides)
    if indicconformer:
        env["INCLUDE_INDICCONFORMER"] = "1"
    _run(["bash", "scripts/run_all.sh", profile, "eval"], env)


@app.function(**CPU_KW)
def report(profile: str = "quick", overrides: str = "") -> None:
    _run(["bash", "scripts/run_all.sh", profile, "report"], _parse_overrides(overrides))


@app.function(**CPU_KW)
def pipeline(profile: str = "quick", overrides: str = "", indicconformer: bool = False) -> None:
    """Whole pipeline server-side (survives your laptop going away). Idempotent: re-launch after a
    timeout/crash; prepare is skipped if data exists, training resumes, evaluation skips finished pairs."""
    vol.reload()
    data = "data" if profile == "full" else "data_quick"
    if (Path(VOL_DIR) / data / "kathbath_train" / "meta.json").exists():
        print(f"[pipeline] {data}/kathbath_train exists -> skipping prepare", flush=True)
    else:
        prepare.remote(profile, overrides)
    # Whisper and CTC use separate GPU containers, so they can train in parallel.
    h1 = train_whisper.spawn(profile, overrides)
    h2 = train_ctc.spawn(profile, overrides)
    h1.get()
    h2.get()
    evaluate.remote(profile, overrides, indicconformer)
    report.remote(profile, overrides)
    print("[pipeline] done. Download: modal volume get indic-asr-vol "
          f"{'results' if profile == 'full' else 'results_quick'} ./results_modal --force", flush=True)


@app.function(**CPU_KW)
def status() -> None:
    """What is on the volume + tail of every log (cheap; use to monitor detached jobs)."""
    vol.reload()
    subprocess.run("du -sh /vol/* 2>/dev/null; echo; ls -la /vol/runs /vol/runs_quick 2>/dev/null; "
                   "for f in /vol/logs/*.log; do echo; echo \"=== $f\"; tr '\\r' '\\n' < \"$f\" | tail -n 6 | cut -c1-220; done",
                   shell=True)


@app.local_entrypoint()
def main(stage: str = "smoke", profile: str = "quick", overrides: str = "", indicconformer: bool = False):
    if profile not in ("quick", "full"):
        raise SystemExit("--profile must be quick or full")
    if stage == "smoke":
        smoke.remote(overrides)
    elif stage == "prepare":
        prepare.remote(profile, overrides)
    elif stage == "train_whisper":
        train_whisper.remote(profile, overrides)
    elif stage == "train_ctc":
        train_ctc.remote(profile, overrides)
    elif stage == "eval":
        evaluate.remote(profile, overrides, indicconformer)
    elif stage == "report":
        report.remote(profile, overrides)
    elif stage == "all":
        pipeline.remote(profile, overrides, indicconformer)
    elif stage == "status":
        status.remote()
    else:
        raise SystemExit("--stage: smoke|prepare|train_whisper|train_ctc|eval|report|all|status")
