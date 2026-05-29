"""Git metadata helpers for experiment tracking."""

import subprocess
from pathlib import Path


def get_git_commit_id(repo_dir: Path | None = None) -> str:
    """Return current git commit hash or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repo_dir or Path.cwd(),
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
