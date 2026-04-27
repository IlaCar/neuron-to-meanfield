#!/usr/bin/env python3
"""Run all notebooks in the repo and produce a validation report.

This script uses nbclient directly (same backend as ``jupyter nbconvert
--execute``) but with explicit environment control so the correct
Python interpreter and imports are used.

Usage::

    # Quick mode: skip heavy simulation notebooks
    python validate_notebooks.py --quick

    # Full mode: run everything with a long timeout
    python validate_notebooks.py --timeout 600

The script sets ``MPLBACKEND=Agg`` automatically so notebooks with
``%matplotlib widget`` do not crash in a headless environment.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

# Notebooks that are expected to need pre-generated fitting data
DATA_DEPENDENT = {
    "transfer_function/analysis/Analysing_params_distribution_FS.ipynb",
    "transfer_function/analysis/Analysing_params_distribution_RS.ipynb",
    "transfer_function/analysis/Analysing_params_distribution_RS_no_adapt.ipynb",
}

# Heavy Brian2 simulation notebooks that may need >5 min or large RAM
HEAVY_NOTEBOOKS = {
    "neural_network_simulation/NN_FS_RS.ipynb",
    "neural_network_simulation/NN_FS_RS_no_adapt.ipynb",
}


def find_repo_root() -> str:
    """Walk up from script location to find repo root (has AGENTS.md + ntmf/)."""
    here = Path(__file__).resolve().parent
    while not ((here / "ntmf").is_dir() and (here / "AGENTS.md").is_file()):
        parent = here.parent
        if parent == here:
            sys.exit("Error: Could not locate repo root (looked for AGENTS.md + ntmf/)")
        here = parent
    return str(here)


def discover_notebooks(repo: str) -> list[str]:
    """Return sorted list of notebook paths relative to repo root."""
    notebooks = []
    for root, _dirs, files in os.walk(repo):
        if ".git" in root or ".worktrees" in root:
            continue
        for f in files:
            if f.endswith(".ipynb") and "_executed" not in f:
                notebooks.append(os.path.relpath(os.path.join(root, f), repo))
    return sorted(notebooks)


def run_one(notebook_rel: str, repo: str, timeout: int) -> dict:
    """Execute a single notebook with nbclient and return result dict."""
    nb_path = os.path.join(repo, notebook_rel)
    nb_dir = os.path.dirname(nb_path)

    # Make sure repo root is on sys.path *inside* the notebook kernel
    env = os.environ.copy()
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    # Force headless matplotlib backend so %matplotlib widget doesn't crash
    env["MPLBACKEND"] = "Agg"

    with open(nb_path) as f:
        nb = nbformat.read(f, as_version=4)

    # Inject a sys.path cell at the very beginning so ``import utils`` works
    # when notebooks are executed from their own subdirectories.
    path_cell = nbformat.v4.new_code_cell(
        f'import sys, os\nrepo_root = {repr(repo)}\n'
        f'if repo_root not in sys.path:\n'
        f'    sys.path.insert(0, repo_root)'
    )
    nb.cells.insert(0, path_cell)

    orig_dir = os.getcwd()
    os.chdir(nb_dir)
    start = time.time()
    try:
        client = NotebookClient(nb, timeout=timeout, kernel_name="python3")
        client.execute(env=env)
        return {
            "status": "PASS",
            "elapsed_s": round(time.time() - start, 1),
            "error": "",
        }
    except Exception:
        # Find the first non-injected cell that raised an error
        found = None
        for i, cell in enumerate(nb.cells):
            if i == 0:
                continue
            if cell.get("cell_type") != "code":
                continue
            for out in cell.get("outputs", []):
                if out.get("output_type") == "error":
                    ename = out.get("ename", "Unknown")
                    evalue = out.get("evalue", "")
                    src_preview = "".join(cell["source"])[:200].replace("\n", " ")
                    found = {
                        "status": "FAIL",
                        "elapsed_s": round(time.time() - start, 1),
                        "error_type": ename,
                        "error": evalue[:250],
                        "cell_source": src_preview,
                    }
                    break
            if found:
                break
        return found or {
            "status": "FAIL",
            "elapsed_s": round(time.time() - start, 1),
            "error_type": "Unknown",
            "error": "Cell execution error (could not locate detail)",
            "cell_source": "",
        }
    finally:
        os.chdir(orig_dir)


def main():
    parser = argparse.ArgumentParser(description="Validate all notebooks in the repo.")
    parser.add_argument("--quick", action="store_true",
                        help="Skip heavy simulation notebooks that need a lot of RAM/time.")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Max seconds per notebook (default 300).")
    parser.add_argument("--output", type=str, default="notebook_report.json",
                        help="JSON report path.")
    parser.add_argument("--markdown", type=str, default="notebook_report.md",
                        help="Markdown report path.")
    args = parser.parse_args()

    repo = find_repo_root()
    print(f"Repo root: {repo}")
    notebooks = discover_notebooks(repo)
    print(f"Found {len(notebooks)} notebooks")

    results = {}
    for nb in notebooks:
        label = os.path.relpath(nb, repo) if os.path.isabs(nb) else nb
        decision = "RUN"
        if args.quick and label in HEAVY_NOTEBOOKS:
            decision = "SKIP_HEAVY"

        if decision.startswith("SKIP"):
            results[label] = {"status": decision, "elapsed_s": 0, "error": ""}
            print(f"{decision:12} {label}")
            continue

        print(f"{'RUNNING':12} {label} ...", end=" ", flush=True)
        res = run_one(label, repo, args.timeout)
        results[label] = res
        print(f"{res['status']} ({res['elapsed_s']}s)")

    # Aggregate
    report = {
        "repo_root": repo,
        "timeout_s": args.timeout,
        "quick_mode": args.quick,
        "total": len(results),
        "pass": sum(1 for r in results.values() if r["status"] == "PASS"),
        "fail": sum(1 for r in results.values() if r["status"] == "FAIL"),
        "timeout": sum(1 for r in results.values() if r["status"] == "TIMEOUT"),
        "skip_heavy": sum(1 for r in results.values() if r["status"] == "SKIP_HEAVY"),
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON report written to: {args.output}")

    # Markdown
    md = [
        "# Notebook Validation Report\n",
        f"- **Repo**: `{repo}`\n",
        f"- **Timeout**: `{args.timeout}s` per notebook\n",
        f"- **Quick mode**: `{args.quick}`\n",
        f"- **Total notebooks**: {len(results)}\n\n",
        "## Summary\n\n",
        "| Status | Count |\n|---|---:|\n",
        f"| PASS | {report['pass']} |\n",
        f"| FAIL | {report['fail']} |\n",
        f"| TIMEOUT | {report['timeout']} |\n",
        f"| SKIP_HEAVY | {report['skip_heavy']} |\n\n",
        "## Details\n\n",
        "| Notebook | Status | Time | Error |\n",
        "|---|---|---|---|\n",
    ]
    for nb, r in sorted(results.items()):
        err = r.get("error", "").replace("|", "\\|").replace("\n", " ")
        md.append(f"| `{nb}` | **{r['status']}** | {r['elapsed_s']}s | {err[:120]} |\n")
    with open(args.markdown, "w") as f:
        f.writelines(md)
    print(f"Markdown report written to: {args.markdown}")


if __name__ == "__main__":
    main()
