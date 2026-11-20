"""Run one Lab 3 stage on Kaggle GPU through the Kaggle CLI (no browser), then fetch the outputs.

VI: Chạy từng bước của Member B trên Kaggle bằng Kaggle CLI (token ở ~/.kaggle/, không bao giờ nằm trong repo).
Repo pptthu/Lab-DL là private → Kaggle không clone được; code đưa lên bằng một Kaggle Dataset PRIVATE
tạo từ cây index (`git write-tree` → `git archive <tree>`: đúng code đã stage; mã <HEAD>+tree:<cây> ghi vào BUILD_INFO.json và vào log mỗi lần chạy).
Thư mục tạm nằm NGOÀI repo (biến môi trường LAB3_KAGGLE_WORK, mặc định %TEMP%/lab3_kaggle).

Chạy từ gốc repo / Run from the repo root:
    python -m labs.lab3.kaggle.kaggle_run publish-code            # tạo / cập nhật dataset code (code đã stage)
    python -m labs.lab3.kaggle.kaggle_run push --stage pilot      # tạo kernel và chạy (GPU T4, 1 GPU)
    python -m labs.lab3.kaggle.kaggle_run push --stage main --runs main_seed43
    python -m labs.lab3.kaggle.kaggle_run push --stage test --model-kernels lab3-main-main-seed42 ...
    python -m labs.lab3.kaggle.kaggle_run status --stage pilot
    python -m labs.lab3.kaggle.kaggle_run fetch --stage pilot     # tải outputs về labs/lab3/outputs/
Owner: Phúc (Member B).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from shared.paths import REPO_ROOT, output_dir

KAGGLE_USER = os.environ.get("LAB3_KAGGLE_USER", "")   # rỗng → lấy từ token (kaggle_user())
DATASET_SLUG = "lab3-redo-code"
CODE_PATHS = ("labs/__init__.py", "labs/lab3", "shared", "requirements.txt")
# Gói cài trên Kaggle: đúng phiên bản trong labs/lab3/requirements.txt (torch giữ bản của Kaggle).
PIP_PACKAGES = ("transformers==5.22.1", "datasets==5.1.0", "evaluate==0.4.6", "accelerate==1.15.0")
ACCELERATOR = "NvidiaTeslaT4"
TEMPLATE = Path(__file__).with_name("kernel_template.py")


def work_dir() -> Path:
    d = Path(os.environ.get("LAB3_KAGGLE_WORK", Path(tempfile.gettempdir()) / "lab3_kaggle"))
    d.mkdir(parents=True, exist_ok=True)
    if REPO_ROOT in d.resolve().parents:
        raise RuntimeError(f"work dir must be outside the repo: {d}")
    return d


def kaggle(*args: str, check: bool = True) -> str:
    """Run the Kaggle CLI; never prints the token (the CLI reads ~/.kaggle/access_token itself)."""
    exe = shutil.which("kaggle") or "kaggle"
    out = subprocess.run([exe, *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    text = (out.stdout or "") + (out.stderr or "")
    if check and out.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} failed:\n{text}")
    return text


def kaggle_user() -> str:
    """Kaggle username of the token in ~/.kaggle (`kaggle config view`), or LAB3_KAGGLE_USER.

    VI: Tên tài khoản lấy từ token đang dùng, không ghi cố định (đổi tài khoản = đổi token).
    """
    global KAGGLE_USER
    if not KAGGLE_USER:
        for line in kaggle("config", "view").splitlines():
            if line.strip().startswith("- username:"):
                KAGGLE_USER = line.split(":", 1)[1].strip()
        if not KAGGLE_USER or KAGGLE_USER == "None":
            raise RuntimeError("no Kaggle username: put a token in ~/.kaggle/access_token")
    return KAGGLE_USER


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def kernel_slug(stage: str, runs: list[str] | None = None, smoke: bool = False) -> str:
    """lab3-<stage>[-<run>...][-smoke], Kaggle slug rules: lower case, '-' only."""
    parts = ["lab3", stage, *(runs or [])] + (["smoke"] if smoke else [])
    return "-".join(parts).replace("_", "-").lower()


# ---------------------------------------------------------------- dataset code
def publish_code(message: str | None = None) -> dict:
    """Staged code (git index, only CODE_PATHS) -> private dataset <user>/lab3-redo-code (create or new version).

    VI: Không tạo commit (Phúc/Thư tự commit sau). Đóng gói đúng những gì đã `git add`: cây index
    (`git write-tree`, không có tác giả) → `git archive <tree>`. Mã ghi vào log: <HEAD>+tree:<tree>.
    Thay đổi chưa stage KHÔNG lên Kaggle (in cảnh báo). File bị .gitignore (data/raw, models, _smoke) không có.
    """
    head = git("rev-parse", "--short", "HEAD")
    tree = git("write-tree")
    unstaged = git("diff", "--name-only", "--", *CODE_PATHS)
    if unstaged:
        print("[publish-code] cảnh báo: thay đổi CHƯA stage, KHÔNG được đưa lên Kaggle:\n" + unstaged)
    commit = f"{head}+tree:{tree[:12]}"
    ds_dir = work_dir() / "dataset"
    shutil.rmtree(ds_dir, ignore_errors=True)
    ds_dir.mkdir(parents=True)
    zip_path = ds_dir / "lab3_code.zip"
    subprocess.run(["git", "archive", "--format=zip", "-o", str(zip_path), tree, *CODE_PATHS], cwd=REPO_ROOT,
                   check=True)
    info = {"commit": commit, "head": git("rev-parse", "HEAD"), "tree": tree,
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (ds_dir / "BUILD_INFO.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    meta = {"title": DATASET_SLUG, "id": f"{kaggle_user()}/{DATASET_SLUG}", "licenses": [{"name": "other"}]}
    (ds_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    # dataset chưa có thì `datasets status` trả 403 (đo 2026-11-20) → kiểm bằng danh sách dataset của mình
    exists = f"{kaggle_user()}/{DATASET_SLUG}" in kaggle("datasets", "list", "--mine", check=False)
    if exists:
        print(kaggle("datasets", "version", "-p", str(ds_dir), "-m", message or f"code {commit[:7]}"))
    else:
        print(kaggle("datasets", "create", "-p", str(ds_dir)))   # mặc định private
    for _ in range(60):
        status = kaggle("datasets", "status", f"{kaggle_user()}/{DATASET_SLUG}", check=False).strip()
        if "ready" in status.lower():
            break
        time.sleep(10)
    print(f"[publish-code] {kaggle_user()}/{DATASET_SLUG} @ {commit[:7]}: {status}")
    return info


# ---------------------------------------------------------------- kernel
def build_kernel(stage: str, runs: list[str] | None, smoke: bool, model_kernels: list[str] | None,
                 gpu: bool = True) -> Path:
    """Kernel folder: kernel-metadata.json + script from kernel_template.py with the stage parameters."""
    slug = kernel_slug(stage, runs, smoke)
    kdir = work_dir() / "kernels" / slug
    shutil.rmtree(kdir, ignore_errors=True)
    kdir.mkdir(parents=True)
    params = {"STAGE": stage, "RUNS": runs or [], "SMOKE": smoke, "PIP_PACKAGES": list(PIP_PACKAGES),
              "DATASET_SLUG": DATASET_SLUG}
    code = TEMPLATE.read_text(encoding="utf-8").replace("PARAMS = {}  # filled by kaggle_run.py",
                                                         f"PARAMS = json.loads({json.dumps(json.dumps(params))})")
    (kdir / "run.py").write_text(code, encoding="utf-8")
    meta = {
        "id": f"{kaggle_user()}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
        "enable_internet": True, "dataset_sources": [f"{kaggle_user()}/{DATASET_SLUG}"],
        "kernel_sources": [f"{kaggle_user()}/{k}" for k in (model_kernels or [])], "competition_sources": [],
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return kdir


def push(stage: str, runs=None, smoke=False, model_kernels=None, gpu=True) -> str:
    kdir = build_kernel(stage, runs, smoke, model_kernels, gpu)
    args = ["kernels", "push", "-p", str(kdir)]
    if gpu:
        args += ["--accelerator", ACCELERATOR]
    print(kaggle(*args))
    return kdir.name


def status(slug: str) -> str:
    return kaggle("kernels", "status", f"{kaggle_user()}/{slug}", check=False).strip()


def wait(slug: str, every_s: int = 600, max_hours: float = 12.5) -> str:
    """Poll the kernel status every ~10 minutes until it is complete / error / cancelled."""
    t0 = time.time()
    while True:
        s = status(slug)
        print(time.strftime("%H:%M:%S"), s, flush=True)
        low = s.lower()
        if any(k in low for k in ("complete", "error", "cancel")):
            return s
        if time.time() - t0 > max_hours * 3600:
            return s
        time.sleep(every_s)


# ---------------------------------------------------------------- tải kết quả
def fetch(slug: str, smoke: bool = False) -> dict:
    """Download the kernel output; copy logs / results / figures / demo into labs/lab3/outputs/.

    VI: <stage>_runs.csv được GỘP (không ghi trùng run); file khác chép đè (vd _config.json có cờ
    test_evaluated sau bước test). Checkpoint (models/) để lại trong thư mục tạm, không vào repo.
    """
    from labs.lab3.experiment.experiment import merge_run_tables

    out_dir = work_dir() / "outputs" / slug
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    print(kaggle("kernels", "output", f"{kaggle_user()}/{slug}", "-p", str(out_dir), "-o"))
    src_root = out_dir / "outputs"
    if not src_root.exists():
        raise RuntimeError(f"no outputs/ folder in the kernel output: {sorted(p.name for p in out_dir.iterdir())}")
    dst_root = output_dir("lab3", "_smoke") if smoke else output_dir("lab3", "logs").parent   # shared.paths, không viết cứng
    copied = {"merged": [], "copied": []}
    for kind in ("logs", "results", "figures", "demo"):
        src = src_root / kind
        if not src.exists():
            continue
        dst = dst_root / kind if smoke else output_dir("lab3", kind)
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.iterdir()):
            if f.is_file() and f.name.endswith("_runs.csv"):
                n = merge_run_tables(f, dst / f.name)
                copied["merged"].append(f"{kind}/{f.name} (+{n})")
            elif f.is_file():
                shutil.copy2(f, dst / f.name)
                copied["copied"].append(f"{kind}/{f.name}")
    (work_dir() / "outputs" / f"{slug}.fetched.json").write_text(json.dumps(copied, indent=1), encoding="utf-8")
    return copied


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Lab 3 on Kaggle (Member B)")
    sub = p.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("publish-code")
    pc.add_argument("-m", "--message")
    for name in ("push", "status", "wait", "fetch"):
        sp = sub.add_parser(name)
        sp.add_argument("--stage", required=True,
                        choices=("timing", "pilot", "main", "maxlen", "test", "figures"))
        sp.add_argument("--runs", nargs="*")
        sp.add_argument("--smoke", action="store_true")
        if name == "push":
            sp.add_argument("--model-kernels", nargs="*", help="--stage test: kernel của 3 seed (chứa checkpoint)")
            sp.add_argument("--cpu", action="store_true", help="chạy CPU (kiểm đường đi, không tốn quota GPU)")
    a = p.parse_args(argv)
    if a.cmd == "publish-code":
        print(publish_code(a.message))
        return
    slug = kernel_slug(a.stage, a.runs, a.smoke)
    if a.cmd == "push":
        print(push(a.stage, a.runs, a.smoke, a.model_kernels, gpu=not a.cpu))
    elif a.cmd == "status":
        print(status(slug))
    elif a.cmd == "wait":
        print(wait(slug))
    else:
        print(json.dumps(fetch(slug, smoke=a.smoke), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
