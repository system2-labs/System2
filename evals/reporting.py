"""evals/reporting.py - Markdown table generation and structured JSON reporting."""

import datetime
import json
import os
import subprocess
from typing import Any, Dict, List, Sequence


def get_system_metadata(device: str) -> Dict[str, Any]:
    """Capture execution environment metadata for verifiable benchmarking."""
    import torch
    import transformers

    git_sha = "unknown"
    git_dirty = False
    git_branch = "unknown"

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        diff_out = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        git_dirty = len(diff_out) > 0
        git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    except Exception:
        pass

    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": git_sha,
        "git_branch": git_branch,
        "git_dirty": git_dirty,
        "device": device,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "mps_available": torch.backends.mps.is_available(),
    }


def render_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render a clean GitHub-flavored Markdown table."""
    if not headers or not rows:
        return ""

    str_headers = [str(h) for h in headers]
    str_rows = [[str(cell) for cell in row] for row in rows]

    col_widths = [len(h) for h in str_headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(cell))
            else:
                col_widths.append(len(cell))

    def make_row(cells):
        padded = [cell.ljust(w) for cell, w in zip(cells, col_widths)]
        return "| " + " | ".join(padded) + " |"

    header_line = make_row(str_headers)
    separator_line = "| " + " | ".join("-" * w for w in col_widths) + " |"
    body_lines = [make_row(row) for row in str_rows]

    return "\n".join([header_line, separator_line] + body_lines)


def save_json_report(path: str, data: Dict[str, Any]) -> None:
    """Save report dictionary to formatted JSON file, creating parent directories if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
