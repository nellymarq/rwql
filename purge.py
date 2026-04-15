"""
RWQL Purge — Dead Code & File Cleaner
======================================
Uses Claude to identify and safely remove:
- Unused Python files (not imported anywhere)
- Empty __init__.py stubs with no content
- Orphaned test files with no corresponding source
- Backup files left by RWQL patches
- Generated artifacts (.pyc, __pycache__, .coverage, etc.)

Usage:
  python3 purge.py [--project nutriforge|kiwi|all] [--dry-run]
"""

import asyncio
import os
import re
import sys
import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime, timezone

import anthropic

MODEL = "claude-opus-4-6"

PROJECTS = {
    "calsanova": Path("/home/nelly/calsanova/backend/app"),
    "kiwi": Path("/home/nelly/kiwi"),
    "scythene": Path("/home/nelly/scythene/src"),
}

PURGE_SYSTEM = """You are a code archaeology expert. Given a list of Python files and their import graphs, identify which files are truly dead (unreachable from any entry point).

Entry points for FastAPI projects:
- main.py (imports everything)
- Any file with if __name__ == "__main__"
- test files (they import modules directly)

Respond ONLY with JSON:
{
  "dead_files": ["relative/path/to/unused.py"],
  "empty_stubs": ["relative/path/__init__.py"],
  "reasoning": "Brief explanation of methodology"
}

Be conservative — only flag files you are CERTAIN are unused. When in doubt, exclude from the list."""


def build_import_graph(src_dir: Path) -> dict[str, list[str]]:
    """Map each file to the files it imports from the same project."""
    graph: dict[str, list[str]] = {}
    all_modules = set()

    for py_file in src_dir.rglob("*.py"):
        if ".venv" in str(py_file):
            continue
        rel = py_file.relative_to(src_dir)
        module_name = str(rel).replace("/", ".").removesuffix(".py")
        all_modules.add(module_name)
        graph[str(rel)] = []

    for py_file in src_dir.rglob("*.py"):
        if ".venv" in str(py_file):
            continue
        rel = str(py_file.relative_to(src_dir))
        text = py_file.read_text(errors="ignore")
        imports = []

        for line in text.splitlines():
            m = re.match(r"from\s+([\w.]+)\s+import", line)
            if m:
                imports.append(m.group(1))
            m = re.match(r"import\s+([\w.]+)", line)
            if m:
                imports.append(m.group(1))

        # Filter to project-internal imports
        graph[rel] = [i for i in imports if any(m.startswith(i) for m in all_modules)]

    return graph


def find_backup_files(root: Path) -> list[Path]:
    """Find .rwql_backup files left by patch operations."""
    return list(root.rglob("*.rwql_backup"))


def find_pycache(root: Path) -> list[Path]:
    """Find __pycache__ directories to clean."""
    return [p for p in root.rglob("__pycache__") if ".venv" not in str(p)]


async def analyze_dead_code(src_dir: Path, client: anthropic.AsyncAnthropic) -> tuple[list[str], list[str]]:
    """Ask Claude to identify dead files given the import graph."""
    graph = build_import_graph(src_dir)

    # Find empty stubs right now
    empty_stubs = []
    for py_file in src_dir.rglob("*.py"):
        if ".venv" in str(py_file):
            continue
        text = py_file.read_text(errors="ignore")
        real_lines = [l for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
        if len(real_lines) <= 3:
            empty_stubs.append(str(py_file.relative_to(src_dir)))

    graph_text = json.dumps(graph, indent=2)
    if len(graph_text) > 8000:
        # Summarize for context window
        graph_text = graph_text[:8000] + "\n... (truncated)"

    response = await client.messages.create(
        model=MODEL,
        max_tokens=2048,
        thinking={"type": "adaptive"},
        system=PURGE_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"""Project source directory: {src_dir}

Import graph (file -> imported_modules):
{graph_text}

Known empty stubs (≤3 real lines of code):
{json.dumps(empty_stubs, indent=2)}

Identify dead/unused files that can be safely removed."""
        }],
    )

    text = next((b.text for b in response.content if hasattr(b, "text")), "")

    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            data = json.loads(m.group())
            return data.get("dead_files", []), data.get("empty_stubs", [])
    except (json.JSONDecodeError, AttributeError):
        pass

    return [], empty_stubs


async def purge_project(project_name: str, src_dir: Path, dry_run: bool, client: anthropic.AsyncAnthropic):
    print(f"\n{'='*60}")
    print(f"  RWQL Purge: {project_name}")
    print(f"{'='*60}")

    # 1. Clean generated artifacts
    pycache_dirs = find_pycache(src_dir.parent)
    print(f"  Found {len(pycache_dirs)} __pycache__ directories")
    if not dry_run:
        for d in pycache_dirs:
            shutil.rmtree(d, ignore_errors=True)
        print(f"  Removed {len(pycache_dirs)} __pycache__ directories")
    else:
        for d in pycache_dirs[:5]:
            print(f"  [DRY] Would remove: {d}")

    # 2. Clean backup files
    backups = find_backup_files(src_dir.parent)
    print(f"  Found {len(backups)} .rwql_backup files")
    if not dry_run and backups:
        for f in backups:
            f.unlink()
        print(f"  Removed {len(backups)} backup files")

    # 3. Analyze dead code with Claude
    print(f"  Analyzing import graph...")
    dead_files, empty_stubs = await analyze_dead_code(src_dir, client)

    print(f"  Claude identified:")
    print(f"    Dead files:   {len(dead_files)}")
    print(f"    Empty stubs:  {len(empty_stubs)}")

    # Show findings
    for f in dead_files:
        full_path = src_dir / f
        status = "WOULD REMOVE" if dry_run else "REMOVING"
        print(f"  [{status}] Dead: {f}")
        if not dry_run and full_path.exists():
            # Extra safety: don't remove __init__.py from packages that have sibling .py files
            if full_path.name == "__init__.py":
                siblings = list(full_path.parent.glob("*.py"))
                if len(siblings) > 1:
                    print(f"    [skip] Has sibling .py files, keeping __init__.py")
                    continue
            full_path.unlink()

    for f in empty_stubs:
        # Only remove empty __init__.py if the package has no other Python files
        full_path = src_dir / f
        if not full_path.exists():
            continue
        siblings = [s for s in full_path.parent.glob("*.py") if s != full_path]
        if siblings:
            print(f"  [skip] Non-empty package, keeping: {f}")
        else:
            status = "WOULD REMOVE" if dry_run else "REMOVING"
            print(f"  [{status}] Empty stub: {f}")
            if not dry_run:
                full_path.unlink()


async def main():
    parser = argparse.ArgumentParser(description="RWQL Purge — Dead Code Cleaner")
    parser.add_argument("--project", default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = anthropic.AsyncAnthropic(api_key=api_key)
    projects = list(PROJECTS.items()) if args.project == "all" else [
        (args.project, PROJECTS[args.project])
    ]

    for name, path in projects:
        await purge_project(name, path, args.dry_run, client)

    print("\nPurge complete.")


if __name__ == "__main__":
    asyncio.run(main())
