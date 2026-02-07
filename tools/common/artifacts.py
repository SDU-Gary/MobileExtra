from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


def repo_root() -> Path:
    # tools/common/*.py -> tools/common -> tools -> repo
    return Path(__file__).resolve().parents[2]


def default_artifacts_dir(root: Optional[Path] = None) -> Path:
    """Resolve the default artifacts directory.

    Priority:
    1) $MOBILEEXTRA_ARTIFACTS_DIR
    2) Sibling dir next to repo root (outside source tree)
    """
    root = root or repo_root()
    env = os.environ.get("MOBILEEXTRA_ARTIFACTS_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    # Default to a sibling directory (outside the source tree)
    return (root.parent / f"{root.name}_artifacts").resolve()


def _safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run_git(args, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def write_git_snapshot(out_dir: Path, root: Optional[Path] = None) -> None:
    root = root or repo_root()
    try:
        commit = _run_git(["rev-parse", "HEAD"], root).stdout.strip()
        diff = _run_git(["diff"], root).stdout
        status = _run_git(["status", "--porcelain=v1"], root).stdout
    except Exception as e:
        _safe_write_text(out_dir / "git_error.txt", f"{e}\n")
        return

    _safe_write_text(out_dir / "git_commit.txt", commit + "\n")
    _safe_write_text(out_dir / "git_status.txt", status)
    _safe_write_text(out_dir / "git_diff.patch", diff)


def write_env_snapshot(out_dir: Path) -> None:
    lines = []
    lines.append(f"python={sys.version.replace(os.linesep, ' ')}")
    lines.append(f"platform={platform.platform()}")
    try:
        import torch  # type: ignore

        lines.append(f"torch={torch.__version__}")
        lines.append(f"cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"cuda_version={torch.version.cuda}")
            try:
                lines.append(f"gpu_name={torch.cuda.get_device_name(0)}")
            except Exception:
                pass
    except Exception as e:
        lines.append(f"torch_import_error={e}")

    _safe_write_text(out_dir / "env.txt", "\n".join(lines) + "\n")


@dataclass(frozen=True)
class RunDirs:
    run_dir: Path
    checkpoints_dir: Path
    tensorboard_dir: Path
    logs_dir: Path


def create_run_dirs(
    artifacts_dir: Path,
    run_name: Optional[str] = None,
    prefix: str = "train",
) -> RunDirs:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_name = (run_name or "run").strip().replace(" ", "_")
    run_dir = (artifacts_dir / "runs" / f"{ts}_{prefix}_{safe_name}").resolve()
    checkpoints_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    logs_dir = run_dir / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return RunDirs(
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        tensorboard_dir=tensorboard_dir,
        logs_dir=logs_dir,
    )

