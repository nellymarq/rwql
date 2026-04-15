"""
RWQL — Ralph Wiggum Quality Loop
===================================
An autonomous self-improving quality agent for Calsanova + Kiwi.

Pipeline:
  1. SCAN  — discover test failures, type errors, lint issues, dead code
  2. TRIAGE — rank issues by severity using Claude
  3. PATCH  — generate fixes via Claude Opus with adaptive thinking
  4. CRITIQUE — Ralph Wiggum Loop: score the patch (0-1), refine if < THRESHOLD
  5. APPLY  — write files, run tests/typecheck to confirm fix
  6. REPORT — human-readable summary, persisted to rwql_log.jsonl

Usage:
  python3 rwql.py [--project calsanova|kiwi|all] [--dry-run] [--once]

  --project: which project to scan (default: all)
  --dry-run: print proposed changes without writing
  --once:    run one pass then exit (default: loop every 30min)

Environment:
  ANTHROPIC_API_KEY — required
"""

import asyncio
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import anthropic

# ── Config ──────────────────────────────────────────────────────────────────

MODEL_PATCH = "claude-opus-4-6"          # Opus for patching (needs precision)
MODEL_TRIAGE = "claude-sonnet-4-6"       # Sonnet for scan triage (cheaper)
MODEL_CRITIQUE = "claude-sonnet-4-6"     # Sonnet for critique scoring (cheaper)
REFINEMENT_THRESHOLD = 0.72
LOG_FILE = Path(__file__).parent / "rwql_log.jsonl"
MEMORY_FILE = Path(__file__).parent / "rwql_memory.json"
REPORT_FILE = Path(__file__).parent / "rwql_report.md"
LOOP_INTERVAL_SECONDS = 1800  # 30 min default
MAX_PATCHES_PER_PASS = 5
MAX_FILE_CHARS = 12000  # Max chars to send in prompts (Opus handles large context well)

# Cost rates per million tokens (USD)
COST_RATES = {
    MODEL_PATCH: {"input": 15.0, "output": 75.0},
    MODEL_TRIAGE: {"input": 3.0, "output": 15.0},
    MODEL_CRITIQUE: {"input": 3.0, "output": 15.0},
}

# Paths/patterns to skip during scanning
SCAN_EXCLUDES = {
    ".venv", "node_modules", "__pycache__", ".next", ".git",
    "android", "migrations", "dist", "build", ".turbo",
}

# Known flaky tests (ordering issues — pass in isolation, fail in suite)
FLAKY_TESTS = {
    "test_injuries",
    "test_quick_meal_plan_self",
    "test_meal_plan_adherence",
    "test_video_expanded",
    "test_inflammation_expanded",
    "test_exercise_library",
    "test_food_reference",
}

PROJECTS = {
    "calsanova": {
        "root": Path("/home/nelly/calsanova"),
        "backend": Path("/home/nelly/calsanova/backend"),
        "frontend": Path("/home/nelly/calsanova/apps/web"),
        "packages": [Path("/home/nelly/calsanova/packages/core")],
        "test_cmd": ["python3", "-m", "pytest", "tests/", "-x", "-q", "--tb=short"],
        "test_timeout": 600,
        "packages_test_cmd": ["npx", "vitest", "run", "--reporter=verbose"],
        "lint_cmd": [
            str(Path("/home/nelly/calsanova/backend/.venv/bin/ruff")),
            "check", "app/", "--output-format=json",
        ],
        "lint_cwd": Path("/home/nelly/calsanova/backend"),
        "bandit_cmd": ["bandit", "-r", "app/", "-f", "json", "-q"],
        "typecheck_cmd": ["npx", "tsc", "--noEmit"],
        "venv": Path("/home/nelly/calsanova/backend/.venv"),
    },
    "kiwi": {
        "root": Path("/home/nelly/kiwi"),
        "backend": Path("/home/nelly/kiwi"),
        "frontend": None,
        "packages": [],
        "test_cmd": ["python3", "-m", "pytest", "tests/", "-x", "-q", "--tb=short"],
        "test_timeout": 300,
        "bandit_cmd": ["bandit", "-r", ".", "-f", "json", "-q", "--exclude", ".venv,tests"],
        "lint_cmd": None,
        "typecheck_cmd": None,
        "venv": None,
    },
    "scythene": {
        "root": Path("/home/nelly/scythene"),
        "backend": Path("/home/nelly/scythene/src"),
        "frontend": Path("/home/nelly/scythene"),
        "packages": [],
        "test_cmd": None,
        "lint_cmd": ["npx", "next", "lint"],
        "typecheck_cmd": ["npx", "tsc", "--noEmit"],
        "venv": None,
    },
}

SCAN_SYSTEM = """You are RWQL's scanner — an expert code auditor for Python (FastAPI/SQLAlchemy) and TypeScript (Next.js/React) projects.

Given raw tool output (test failures, type errors, lint errors, dead code analysis), produce a prioritized JSON list of issues.

IMPORTANT:
- Only report REAL issues. Do not invent problems not present in the scan data.
- For test failures, check if the test name matches a known-flaky pattern (ordering issue).
  If it does, set severity to "low" and note "known ordering issue" in the description.
- For TypeScript errors, include the exact error code (e.g., TS2304).

Respond ONLY with a JSON object:
{
  "issues": [
    {
      "id": "unique-slug",
      "severity": "critical|high|medium|low",
      "category": "test_failure|dead_code|lint|type_error|security|architecture",
      "file": "relative/path/to/file",
      "line": 42,
      "description": "Clear description of the problem",
      "suggested_fix": "Specific, actionable fix description"
    }
  ],
  "scan_summary": "1-2 sentence overview of findings"
}

Severity rules:
- critical: test failure (non-flaky), security issue, runtime crash risk
- high: type errors, unused imports in active files, broken patterns, logic bugs
- medium: dead code, unnecessary complexity, missing error handling
- low: style issues, minor cleanup, known-flaky test failures"""

PATCH_SYSTEM = """You are RWQL's patch engineer — you write precise, minimal code fixes.

Given an issue description and the current file content, produce a list of search/replace blocks.
Each block specifies exact lines to find and their replacement.

Rules:
- Make the smallest change that fixes the issue
- Never add features beyond the fix
- Never remove code unless it's provably dead/unused
- Maintain existing code style exactly
- The "search" string must match EXACTLY in the file (whitespace-sensitive)
- Include enough surrounding context in "search" to be unique in the file

Respond ONLY with a JSON object:
{
  "changes": [
    {
      "file": "relative/path/to/file (omit if same as the primary file)",
      "search": "exact lines to find in the file",
      "replace": "replacement lines"
    }
  ],
  "explanation": "one-line summary of what was changed"
}

If removing code, set "replace" to an empty string. Keep the number of changes minimal.
Most fixes only need changes in a single file. Only include other files if the fix genuinely requires it (e.g., renaming a function used by callers)."""

CRITIQUE_SYSTEM = """You are RWQL's critic — the Ralph Wiggum Loop evaluator for code patches.

Score the proposed patch on these dimensions (0.0 to 1.0):
- correctness: Does it actually fix the described issue without introducing bugs?
- minimality: Is it the smallest change that works? No over-engineering?
- safety: Does it avoid new security or stability risks?
- style_preservation: Does it match the existing codebase style?
- test_coverage: Does the fix address the root cause (not just symptoms)?

Respond ONLY with JSON:
{
  "score": 0.85,
  "dimension_scores": {
    "correctness": 0.9,
    "minimality": 0.8,
    "safety": 0.9,
    "style_preservation": 0.8,
    "test_coverage": 0.8
  },
  "critical_issues": ["list of blockers if score < 0.72"],
  "minor_issues": ["list of small concerns"],
  "strengths": ["what the patch does well"],
  "needs_refinement": false
}"""

ADVERSARIAL_CRITIQUE_SYSTEM = """You are a hostile code reviewer. Your ONLY job is to find flaws in this patch.

Assume the patch is broken until proven otherwise. Look for:
- Edge cases the patch doesn't handle
- Subtle bugs introduced by the change (off-by-one, null checks, type coercion)
- Security issues (injection, XSS, IDOR, mass assignment)
- Concurrency or race condition risks
- Silent behavior changes in untouched code paths
- Missing error handling for new failure modes
- Broken callers — does anything else reference the changed code?

If you find ANY critical flaw, set "has_critical_flaw" to true.
If the patch is genuinely clean, say so — don't fabricate issues.

Respond ONLY with JSON:
{
  "has_critical_flaw": true,
  "flaws": ["specific flaw descriptions"],
  "verdict": "one-line summary"
}"""


# ── Utilities ────────────────────────────────────────────────────────────────

def log_event(event: dict):
    entry = {"ts": datetime.now(timezone.utc).isoformat(), **event}
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


# ── Cost Tracking ───────────────────────────────────────────────────────────

class CostTracker:
    def __init__(self):
        self.calls: list[dict] = []

    def record(self, model: str, response):
        usage = getattr(response, "usage", None)
        if usage:
            self.calls.append({
                "model": model,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            })

    def summary(self) -> dict:
        total_input = sum(c["input_tokens"] for c in self.calls)
        total_output = sum(c["output_tokens"] for c in self.calls)
        total_cost = 0.0
        for c in self.calls:
            rates = COST_RATES.get(c["model"], {"input": 3.0, "output": 15.0})
            total_cost += (c["input_tokens"] / 1_000_000) * rates["input"]
            total_cost += (c["output_tokens"] / 1_000_000) * rates["output"]
        return {
            "api_calls": len(self.calls),
            "input_tokens": total_input,
            "output_tokens": total_output,
            "estimated_cost_usd": round(total_cost, 4),
        }


# ── Historical Memory ──────────────────────────────────────────────────────

def load_memory() -> dict:
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "patches_applied": [],
        "patches_reverted": [],
        "false_positives": [],
        "project_patterns": {},
    }


def save_memory(memory: dict):
    MEMORY_FILE.write_text(json.dumps(memory, indent=2, default=str))


def memory_context_for_prompt(memory: dict, project: str) -> str:
    lines = []
    applied = [p for p in memory.get("patches_applied", []) if p.get("project") == project]
    reverted = [p for p in memory.get("patches_reverted", []) if p.get("project") == project]
    false_pos = memory.get("false_positives", [])
    patterns = memory.get("project_patterns", {}).get(project, "")

    if applied:
        recent = applied[-5:]
        lines.append(f"Recently applied patches ({len(applied)} total, showing last {len(recent)}):")
        for p in recent:
            lines.append(f"  - {p.get('file', '?')}: {p.get('issue_id', '?')} (score {p.get('score', '?')})")
    if reverted:
        lines.append(f"Reverted patches ({len(reverted)} — these fixes FAILED verification):")
        for p in reverted[-3:]:
            lines.append(f"  - {p.get('file', '?')}: {p.get('reason', '?')}")
    if false_pos:
        lines.append(f"Known false positives (do NOT flag these again):")
        for fp in false_pos[-5:]:
            lines.append(f"  - {fp.get('description', '?')}")
    if patterns:
        lines.append(f"Project notes: {patterns}")
    return "\n".join(lines) if lines else ""


def detect_regressions(memory: dict, test_output: str, project: str) -> list[dict]:
    """Check if any recently-patched files appear in test failure tracebacks."""
    regressions = []
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    recent_patches = [
        p for p in memory.get("patches_applied", [])
        if p.get("project") == project and p.get("date", "") >= cutoff
    ]
    if not recent_patches:
        return []
    for patch in recent_patches:
        patch_file = patch.get("file", "")
        # Extract just the filename for matching against tracebacks
        filename = Path(patch_file).name
        if filename and filename in test_output:
            regressions.append({
                "file": patch_file,
                "issue_id": patch.get("issue_id", "?"),
                "patched_date": patch.get("date", "?"),
            })
    return regressions


def generate_report(
    projects_data: list[dict],
    costs: dict,
    elapsed: float,
    dry_run: bool,
):
    """Write a human-readable markdown report to rwql_report.md."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# RWQL Report — {now}",
        f"",
        f"**Mode:** {'dry run' if dry_run else 'live'} | **Duration:** {elapsed:.1f}s",
        f"**API:** {costs['api_calls']} calls | "
        f"{costs['input_tokens']:,} in / {costs['output_tokens']:,} out | "
        f"~${costs['estimated_cost_usd']:.4f}",
        f"",
    ]
    for proj in projects_data:
        name = proj["name"]
        issues = proj.get("issues", [])
        results = proj.get("results", [])
        regressions = proj.get("regressions", [])
        applied = [r for r in results if r.get("applied")]
        rejected = [r for r in results if not r.get("applied")]

        lines.append(f"## {name}")
        lines.append(f"")
        if not issues:
            lines.append("No issues found — clean.")
        else:
            actionable = [i for i in issues if i.get("severity") in ("critical", "high")]
            info = [i for i in issues if i.get("severity") in ("medium", "low")]
            lines.append(f"**Issues:** {len(actionable)} actionable, {len(info)} informational")

            if applied:
                lines.append(f"")
                lines.append(f"### Patches Applied ({len(applied)})")
                for r in applied:
                    issue = r.get("issue", {})
                    lines.append(f"- `{issue.get('file', '?')}` — {issue.get('description', '?')[:100]} "
                                 f"(score {r.get('score', '?'):.2f})")

            if rejected:
                lines.append(f"")
                lines.append(f"### Patches Rejected/Skipped ({len(rejected)})")
                for r in rejected:
                    issue = r.get("issue", {})
                    lines.append(f"- `{issue.get('file', '?')}` — {issue.get('description', '?')[:100]} "
                                 f"(score {r.get('score', '?'):.2f})")

            if info:
                lines.append(f"")
                lines.append(f"### Informational ({len(info)})")
                for i in info[:10]:
                    lines.append(f"- [{i.get('severity')}] `{i.get('file', '?')}:{i.get('line', '?')}` "
                                 f"— {i.get('description', '?')[:100]}")

        if regressions:
            lines.append(f"")
            lines.append(f"### ⚠ Regressions Detected ({len(regressions)})")
            for reg in regressions:
                lines.append(f"- `{reg['file']}` (patched {reg['patched_date'][:10]}, "
                             f"issue {reg['issue_id']})")

        lines.append(f"")

    REPORT_FILE.write_text("\n".join(lines))
    print(f"  Report written to {REPORT_FILE}")


def run_cmd(cmd: list[str], cwd: Path, env: dict | None = None,
            timeout: int = 300, venv: Path | None = None) -> tuple[int, str, str]:
    """Run a shell command, return (returncode, stdout, stderr).
    If venv is provided, prepend its bin/ to PATH so the correct Python is used."""
    run_env = {**os.environ, **(env or {})}
    if venv:
        venv_bin = str(venv / "bin")
        run_env["PATH"] = f"{venv_bin}:{run_env.get('PATH', '')}"
        run_env["VIRTUAL_ENV"] = str(venv)
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=run_env,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", f"TIMEOUT after {timeout}s"
    except FileNotFoundError as e:
        return 1, "", str(e)


def should_skip(path: Path) -> bool:
    """Check if a path should be excluded from scanning."""
    return any(exc in path.parts for exc in SCAN_EXCLUDES)


# Modules and names that produce false positives with naive string counting.
# These are commonly used in type annotations, test assertions, or expressions
# where the counter sees only 1 occurrence (the import line itself).
PY_SKIP_MODULES = {
    "typing", "collections", "abc", "dataclasses", "__future__",
    "pathlib", "datetime", "enum",
}
PY_SKIP_NAMES = {
    "Any", "Optional", "Union", "Dict", "List", "Tuple", "Set",
    "Callable", "Literal", "TypeVar", "Protocol", "ClassVar",
    "ABC", "abstractmethod", "annotations", "dataclass", "field",
    "Path", "datetime", "timedelta", "timezone", "date",
    "pytest", "math", "json", "os", "sys", "re",
}


def find_dead_imports_py(src_dir: Path) -> str:
    """Find unused imports in Python files.
    Skips __init__.py, common typing/annotation modules, and standard
    library names that produce false positives with naive string counting."""
    lines = []
    for py_file in src_dir.rglob("*.py"):
        if should_skip(py_file):
            continue
        if py_file.name == "__init__.py":
            continue
        text = py_file.read_text(errors="ignore")
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if not (stripped.startswith("import ") or stripped.startswith("from ")):
                continue
            # Skip entire modules known to produce false positives
            from_match = re.match(r"from\s+(\S+)\s+import", stripped)
            if from_match and from_match.group(1).split(".")[0] in PY_SKIP_MODULES:
                continue
            # Check the imported name(s)
            import_match = re.search(r"(?:import|from)\s+(\S+)", stripped)
            if import_match:
                name = import_match.group(1).split(".")[0]
                if name in PY_SKIP_NAMES:
                    continue
                # For 'from X import Y, Z' — check the imported names too
                from_names_match = re.search(r"import\s+(.+)$", stripped)
                if from_names_match:
                    imported = [n.strip().split(" as ")[-1].strip()
                                for n in from_names_match.group(1).split(",")]
                    if all(n in PY_SKIP_NAMES for n in imported if n):
                        continue
                occurrences = text.count(name)
                if occurrences == 1:
                    rel = py_file.relative_to(src_dir)
                    lines.append(f"{rel}:{lineno}: possibly unused: {stripped}")
    return "\n".join(lines[:50]) if lines else "No obviously unused imports found."


def find_dead_imports_ts(src_dir: Path) -> str:
    """Find unused imports in TypeScript/TSX files.
    Skips type-only imports (import type { X }) since tsc is the
    authoritative checker for those."""
    lines = []
    for ext in ("*.tsx", "*.ts"):
        for ts_file in src_dir.rglob(ext):
            if should_skip(ts_file):
                continue
            if ext == "*.ts" and ts_file.suffix == ".tsx":
                continue
            text = ts_file.read_text(errors="ignore")
            for lineno, line in enumerate(text.splitlines(), 1):
                stripped = line.strip()
                if not stripped.startswith("import "):
                    continue
                # Skip type-only imports — tsc handles these
                if stripped.startswith("import type "):
                    continue
                # Skip type keyword inside braces: import { type Foo }
                brace_match = re.search(r"\{([^}]+)\}", stripped)
                if brace_match:
                    names = []
                    for part in brace_match.group(1).split(","):
                        part = part.strip()
                        if part.startswith("type "):
                            continue
                        name = part.split(" as ")[-1].strip()
                        if name:
                            names.append(name)
                    for name in names:
                        if text.count(name) == 1:
                            rel = ts_file.relative_to(src_dir)
                            lines.append(f"{rel}:{lineno}: possibly unused import: {name}")
    return "\n".join(lines[:50]) if lines else "No obviously unused TS/TSX imports found."


# Files that are expected to be small — not dead code
SMALL_FILE_PATTERNS = {
    "__init__.py",          # Python package markers
    "twitter-image.tsx",    # Next.js OG image routes (single export)
    "opengraph-image.tsx",  # Next.js OG image routes
    "setup.ts",             # Test setup files
    "setup.js",
}


def find_empty_modules(src_dir: Path, extensions: tuple[str, ...] = ("*.py",)) -> str:
    """Find files that are effectively empty (< 3 lines of real code).
    Excludes __init__.py, OG image routes, and other expected small files."""
    empty = []
    for ext in extensions:
        for f in src_dir.rglob(ext):
            if should_skip(f):
                continue
            if f.name in SMALL_FILE_PATTERNS:
                continue
            text = f.read_text(errors="ignore")
            real_lines = [l for l in text.splitlines()
                          if l.strip() and not l.strip().startswith("#")
                          and not l.strip().startswith("//")]
            if len(real_lines) == 0:
                rel = f.relative_to(src_dir)
                empty.append(f"{rel}: empty")
    return "\n".join(empty) if empty else "No effectively empty modules found."


def filter_flaky_tests(output: str) -> str:
    """Annotate known-flaky test failures in test output."""
    annotated_lines = []
    for line in output.splitlines():
        is_flaky = any(flaky in line for flaky in FLAKY_TESTS)
        if is_flaky:
            annotated_lines.append(f"{line}  [KNOWN FLAKY — ordering issue, ignore]")
        else:
            annotated_lines.append(line)
    return "\n".join(annotated_lines)


def generate_unified_diff(original: str, patched: str, filepath: str) -> str:
    """Generate a readable unified diff between original and patched content."""
    orig_lines = original.splitlines(keepends=True)
    patch_lines = patched.splitlines(keepends=True)
    diff = difflib.unified_diff(orig_lines, patch_lines,
                                fromfile=f"a/{filepath}",
                                tofile=f"b/{filepath}",
                                lineterm="")
    return "".join(diff)


def resolve_file_path(config: dict, relative_path: str) -> Path | None:
    """Resolve an issue's relative file path against backend or frontend root."""
    # Try backend first
    backend_path = config["backend"] / relative_path
    if backend_path.exists() and backend_path.is_file():
        return backend_path

    # Try frontend
    if config.get("frontend"):
        frontend_path = config["frontend"] / relative_path
        if frontend_path.exists() and frontend_path.is_file():
            return frontend_path

    # Try project root
    root_path = config["root"] / relative_path
    if root_path.exists() and root_path.is_file():
        return root_path

    return None


# ── Scan Phase ───────────────────────────────────────────────────────────────

async def scan_project(project_name: str, config: dict,
                       client: anthropic.AsyncAnthropic,
                       cost_tracker: "CostTracker | None" = None) -> list[dict]:
    """Run all scanners and ask Claude to triage the findings."""
    backend = config["backend"]
    frontend = config.get("frontend")
    venv = config.get("venv")
    findings = []

    # ── Pre-flight: verify tests pass before scanning ──
    preflight_output = ""
    if config.get("test_cmd"):
        test_timeout = config.get("test_timeout", 300)
        print(f"\n  [preflight] Verifying {project_name} tests pass (timeout {test_timeout}s)...")
        rc, stdout, stderr = run_cmd(config["test_cmd"], cwd=backend, venv=venv, timeout=test_timeout)
        preflight_output = f"{stdout}\n{stderr}"
        if rc != 0:
            combined = f"{stdout}\n{stderr}"
            non_flaky = any(
                "FAILED" in line and not any(f in line for f in FLAKY_TESTS)
                for line in combined.splitlines()
            )
            if non_flaky:
                print(f"  [preflight] ABORT — tests already failing before scan. "
                      f"Fix existing failures first.")
                print(f"  {combined[-500:]}")
                log_event({
                    "phase": "preflight_abort", "project": project_name,
                    "reason": "pre-existing test failures",
                    "output": combined[-500:],
                })
                return []
            else:
                findings.append(f"=== BACKEND TESTS PASS (flaky-only failures) ===\n{stdout[-500:]}")
        else:
            findings.append(f"=== BACKEND TESTS PASS ===\n{stdout[-500:]}")
    else:
        findings.append("=== NO BACKEND TESTS CONFIGURED ===")

    # ── Regression detection ──
    if preflight_output:
        memory = load_memory()
        regressions = detect_regressions(memory, preflight_output, project_name)
        if regressions:
            reg_text = "\n".join(
                f"  - {r['file']} (patched {r['patched_date'][:10]})"
                for r in regressions
            )
            print(f"  [regression] ⚠ {len(regressions)} recently-patched files in test output:")
            print(reg_text)
            findings.append(f"=== REGRESSION WARNING ===\n"
                            f"These recently-patched files appear in test output:\n{reg_text}")
            log_event({
                "phase": "regression_detected",
                "project": project_name,
                "files": [r["file"] for r in regressions],
            })

    # ── Parallel scan: typecheck + lint + static analysis + security ──

    # Frontend typecheck
    async def run_typecheck():
        if frontend and config.get("typecheck_cmd"):
            print(f"  [scan] Running typecheck for {project_name}...")
            rc, stdout, stderr = run_cmd(config["typecheck_cmd"], cwd=frontend)
            if rc != 0:
                return f"=== TYPE ERRORS ===\n{stderr[-3000:]}\n{stdout[-1000:]}"
            return "=== TYPECHECK PASS ==="
        return None

    # Lint
    async def run_lint():
        if config.get("lint_cmd"):
            print(f"  [scan] Running lint for {project_name}...")
            lint_cwd = config.get("lint_cwd", config["root"])
            rc, stdout, stderr = run_cmd(config["lint_cmd"], cwd=lint_cwd)
            if stdout.strip() and stdout.strip() != "[]":
                try:
                    lint_issues = json.loads(stdout)
                    return (
                        f"=== LINT ISSUES ({len(lint_issues)}) ===\n" +
                        "\n".join(
                            f"{i['filename']}:{i['location']['row']}: [{i['code']}] {i['message']}"
                            for i in lint_issues[:20]
                        )
                    )
                except json.JSONDecodeError:
                    return f"=== LINT OUTPUT ===\n{(stdout + stderr)[:2000]}"
        return None

    # Static analysis (runs in thread pool since it's CPU-bound)
    async def run_static_analysis():
        results = []
        print(f"  [scan] Analyzing imports for {project_name}...")
        dead_py = find_dead_imports_py(backend)
        results.append(f"=== DEAD PYTHON IMPORTS ===\n{dead_py}")

        if frontend:
            src_dir = frontend / "src" if (frontend / "src").exists() else frontend
            dead_ts = find_dead_imports_ts(src_dir)
            results.append(f"=== DEAD TS/TSX IMPORTS ===\n{dead_ts}")

        print(f"  [scan] Checking for empty modules...")
        empty_py = find_empty_modules(backend, ("*.py",))
        results.append(f"=== EMPTY PYTHON MODULES ===\n{empty_py}")

        if frontend:
            src_dir = frontend / "src" if (frontend / "src").exists() else frontend
            empty_ts = find_empty_modules(src_dir, ("*.tsx", "*.ts"))
            results.append(f"=== EMPTY TS/TSX MODULES ===\n{empty_ts}")

        # Scan packages (e.g., packages/core)
        for pkg in config.get("packages", []):
            if pkg.exists():
                print(f"  [scan] Analyzing package: {pkg.name}...")
                dead_ts = find_dead_imports_ts(pkg / "src" if (pkg / "src").exists() else pkg)
                results.append(f"=== DEAD IMPORTS ({pkg.name}) ===\n{dead_ts}")
        return results

    # Security scan (bandit)
    async def run_security_scan():
        if config.get("bandit_cmd") and shutil.which("bandit"):
            print(f"  [scan] Running security scan for {project_name}...")
            rc, stdout, stderr = run_cmd(
                config["bandit_cmd"], cwd=backend, venv=venv
            )
            if stdout.strip():
                try:
                    bandit_data = json.loads(stdout)
                    results = bandit_data.get("results", [])
                    if results:
                        bandit_lines = "\n".join(
                            f"{r['filename']}:{r['line_number']}: [{r['test_id']}] "
                            f"({r['issue_severity']}) {r['issue_text']}"
                            for r in results[:20]
                        )
                        return f"=== SECURITY ISSUES ({len(results)}) ===\n{bandit_lines}"
                except json.JSONDecodeError:
                    if "No issues" not in stdout:
                        return f"=== SECURITY SCAN OUTPUT ===\n{stdout[:2000]}"
            return "=== SECURITY SCAN PASS ==="
        return None

    scan_results = await asyncio.gather(
        run_typecheck(), run_lint(), run_static_analysis(), run_security_scan()
    )

    for result in scan_results:
        if result is None:
            continue
        if isinstance(result, list):
            findings.extend(result)
        else:
            findings.append(result)

    combined = "\n\n".join(findings)

    # Ask Claude to triage (with memory context)
    memory = load_memory()
    mem_ctx = memory_context_for_prompt(memory, project_name)
    memory_block = f"\n\n=== RWQL MEMORY ===\n{mem_ctx}" if mem_ctx else ""

    print(f"  [scan] Claude triaging {project_name} findings...")
    response = await client.messages.create(
        model=MODEL_TRIAGE,
        max_tokens=4096,
        system=SCAN_SYSTEM,
        messages=[{"role": "user", "content": f"Project: {project_name}\n\n{combined}{memory_block}"}],
    )
    if cost_tracker:
        cost_tracker.record(MODEL_TRIAGE, response)

    text = next((b.text for b in response.content if hasattr(b, "text")), "")

    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            data = json.loads(m.group())
            issues = data.get("issues", [])
            summary = data.get("scan_summary", "")
            print(f"  [scan] {len(issues)} issues found: {summary}")
            log_event({
                "phase": "scan", "project": project_name,
                "issue_count": len(issues), "summary": summary,
            })
            return issues
    except (json.JSONDecodeError, AttributeError):
        pass

    print(f"  [scan] No structured issues from triage")
    return []


# ── Patch Phase ──────────────────────────────────────────────────────────────

def apply_search_replace(content: str, changes: list[dict]) -> str | None:
    """Apply search/replace blocks to file content. Returns None if any search fails.
    Only applies changes that don't have a 'file' key or have a matching file key."""
    result = content
    for change in changes:
        if change.get("file"):
            continue
        search = change.get("search", "")
        replace = change.get("replace", "")
        if not search:
            continue
        if search not in result:
            return None
        result = result.replace(search, replace, 1)
    return result


def apply_multi_file_changes(
    changes: list[dict],
    primary_file: Path,
    primary_content: str,
    project_config: dict,
) -> dict[Path, tuple[str, str]] | None:
    """Apply search/replace changes across multiple files.
    Returns {path: (original, patched)} or None if any search fails."""
    file_changes: dict[Path, list[dict]] = {}
    for change in changes:
        rel = change.get("file")
        if rel:
            path = resolve_file_path(project_config, rel)
            if not path:
                return None
        else:
            path = primary_file
        file_changes.setdefault(path, []).append(change)

    results: dict[Path, tuple[str, str]] = {}
    for path, path_changes in file_changes.items():
        original = primary_content if path == primary_file else path.read_text(errors="ignore")
        patched = original
        for change in path_changes:
            search = change.get("search", "")
            replace = change.get("replace", "")
            if not search:
                continue
            if search not in patched:
                return None
            patched = patched.replace(search, replace, 1)
        if patched != original:
            results[path] = (original, patched)
    return results


async def generate_patch(issue: dict, project_config: dict,
                         client: anthropic.AsyncAnthropic,
                         cost_tracker: "CostTracker | None" = None) -> str | None:
    """Generate a code patch for an issue using search/replace format."""
    file_path = resolve_file_path(project_config, issue.get("file", ""))
    if not file_path:
        return None

    current_content = file_path.read_text(errors="ignore")
    content_for_prompt = current_content
    if len(content_for_prompt) > MAX_FILE_CHARS:
        target_line = issue.get("line", 0)
        if target_line > 0:
            lines = content_for_prompt.splitlines(keepends=True)
            start = max(0, target_line - 100)
            end = min(len(lines), target_line + 100)
            content_for_prompt = (
                f"[... truncated, showing lines {start+1}-{end} of {len(lines)} ...]\n"
                + "".join(lines[start:end])
            )
        else:
            content_for_prompt = content_for_prompt[:MAX_FILE_CHARS] + "\n[... truncated ...]"

    response = await client.messages.create(
        model=MODEL_PATCH,
        max_tokens=8000,
        thinking={"type": "adaptive"},
        system=PATCH_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"""Issue to fix:
Severity: {issue['severity']}
Category: {issue['category']}
File: {issue['file']}
Line: {issue.get('line', 'N/A')}
Description: {issue['description']}
Suggested fix: {issue.get('suggested_fix', 'Fix the described issue')}

Current file content:
```
{content_for_prompt}
```"""
        }],
    )
    if cost_tracker:
        cost_tracker.record(MODEL_PATCH, response)

    text = next((b.text for b in response.content if hasattr(b, "text")), "").strip()

    # Parse search/replace JSON
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            data = json.loads(m.group())
            changes = data.get("changes", [])
            if changes:
                patched = apply_search_replace(current_content, changes)
                if patched is not None:
                    explanation = data.get("explanation", "")
                    if explanation:
                        print(f"    [patch] {explanation}")
                    return patched
                else:
                    print(f"    [patch] Search/replace failed — search string not found in file")
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: if model returned raw file content instead of JSON
    if text and not text.startswith("{"):
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        if text:
            print(f"    [patch] Fell back to full-file output (model didn't use search/replace)")
            return text

    return None


# ── Ralph Wiggum Loop ────────────────────────────────────────────────────────

async def ralph_wiggum_loop(
    issue: dict,
    patch: str,
    original_content: str,
    client: anthropic.AsyncAnthropic,
    max_refinements: int = 4,
    cost_tracker: "CostTracker | None" = None,
) -> tuple[str, float, dict]:
    """
    Critique a patch and refine it if score < REFINEMENT_THRESHOLD.
    Returns (final_patch, score, critique_data).
    """
    current_patch = patch
    final_score = 0.0
    final_critique: dict = {}

    diff = generate_unified_diff(original_content, current_patch, issue.get("file", ""))
    diff_for_prompt = diff[:MAX_FILE_CHARS] if diff else "(no diff — identical content)"

    for attempt in range(max_refinements + 1):
        critique_response = await client.messages.create(
            model=MODEL_CRITIQUE,
            max_tokens=2048,
            system=CRITIQUE_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"""Issue: {issue['description']}

Diff (unified):
```
{diff_for_prompt}
```"""
            }],
        )
        if cost_tracker:
            cost_tracker.record(MODEL_CRITIQUE, critique_response)

        critique_text = next(
            (b.text for b in critique_response.content if hasattr(b, "text")), ""
        )

        try:
            m = re.search(r"\{.*\}", critique_text, re.DOTALL)
            if m:
                critique = json.loads(m.group())
                final_critique = critique
                final_score = float(critique.get("score", 0.0))
                needs_refinement = critique.get("needs_refinement", False)
                critical_issues = critique.get("critical_issues", [])

                print(f"    [rwl] Attempt {attempt + 1}: score={final_score:.2f}, "
                      f"refine={needs_refinement}")

                if (final_score >= REFINEMENT_THRESHOLD
                        or not needs_refinement
                        or attempt == max_refinements):
                    break

                print(f"    [rwl] Score below threshold "
                      f"({final_score:.2f} < {REFINEMENT_THRESHOLD}), refining...")
                orig_for_prompt = original_content[:MAX_FILE_CHARS]
                refinement_response = await client.messages.create(
                    model=MODEL_PATCH,
                    max_tokens=8000,
                    thinking={"type": "adaptive"},
                    system=PATCH_SYSTEM,
                    messages=[{
                        "role": "user",
                        "content": f"""The previous patch scored {final_score:.2f} (below threshold {REFINEMENT_THRESHOLD}).

Critical issues to address:
{chr(10).join(f'- {ci}' for ci in critical_issues)}

Original issue: {issue['description']}

Original file:
```
{orig_for_prompt}
```

Previous diff:
```
{diff_for_prompt}
```

Generate improved search/replace blocks that address the critical issues."""
                    }],
                )
                if cost_tracker:
                    cost_tracker.record(MODEL_PATCH, refinement_response)

                refined_text = next(
                    (b.text for b in refinement_response.content if hasattr(b, "text")),
                    "",
                ).strip()

                # Try to parse search/replace from refinement
                try:
                    rm = re.search(r"\{.*\}", refined_text, re.DOTALL)
                    if rm:
                        rdata = json.loads(rm.group())
                        rchanges = rdata.get("changes", [])
                        if rchanges:
                            refined = apply_search_replace(original_content, rchanges)
                            if refined is not None:
                                current_patch = refined
                            else:
                                print(f"    [rwl] Refinement search/replace failed")
                except (json.JSONDecodeError, AttributeError):
                    # Fallback: raw file content
                    if refined_text and not refined_text.startswith("{"):
                        if refined_text.startswith("```"):
                            refined_text = re.sub(r"^```\w*\n?", "", refined_text)
                            refined_text = re.sub(r"\n?```$", "", refined_text)
                        if refined_text:
                            current_patch = refined_text

                diff = generate_unified_diff(original_content, current_patch, issue.get("file", ""))
                diff_for_prompt = diff[:MAX_FILE_CHARS] if diff else "(no diff)"
        except (json.JSONDecodeError, ValueError):
            break

    # Adversarial pass: if patch passed standard critique, run a hostile review
    if final_score >= REFINEMENT_THRESHOLD:
        try:
            adv_response = await client.messages.create(
                model=MODEL_CRITIQUE,
                max_tokens=2048,
                system=ADVERSARIAL_CRITIQUE_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": f"""Issue: {issue['description']}

Diff:
```
{diff_for_prompt}
```"""
                }],
            )
            if cost_tracker:
                cost_tracker.record(MODEL_CRITIQUE, adv_response)

            adv_text = next(
                (b.text for b in adv_response.content if hasattr(b, "text")), ""
            )
            adv_m = re.search(r"\{.*\}", adv_text, re.DOTALL)
            if adv_m:
                adv_data = json.loads(adv_m.group())
                has_flaw = adv_data.get("has_critical_flaw", False)
                adv_flaws = adv_data.get("flaws", [])
                adv_verdict = adv_data.get("verdict", "")
                if has_flaw:
                    print(f"    [adversarial] FLAW FOUND: {adv_verdict}")
                    for flaw in adv_flaws[:3]:
                        print(f"      - {flaw[:100]}")
                    final_score = max(final_score - 0.15, 0.0)
                    final_critique["adversarial_flaws"] = adv_flaws
                    final_critique["adversarial_verdict"] = adv_verdict
                else:
                    print(f"    [adversarial] Clean: {adv_verdict}")
        except (json.JSONDecodeError, AttributeError, Exception) as e:
            print(f"    [adversarial] Skipped: {e}")

    return current_patch, final_score, final_critique


# ── Apply Phase ───────────────────────────────────────────────────────────────

def apply_patch(file_path: Path, original: str, new_content: str,
                dry_run: bool) -> bool:
    """Write the patch to disk (or show diff in dry-run mode)."""
    if dry_run:
        rel = str(file_path)
        diff = generate_unified_diff(original, new_content, rel)
        if diff:
            print(f"\n    [DRY RUN] Diff for {file_path}:")
            # Show diff with color hints
            for line in diff.splitlines()[:60]:
                if line.startswith("+") and not line.startswith("+++"):
                    print(f"      \033[32m{line}\033[0m")
                elif line.startswith("-") and not line.startswith("---"):
                    print(f"      \033[31m{line}\033[0m")
                else:
                    print(f"      {line}")
            total_lines = len(diff.splitlines())
            if total_lines > 60:
                print(f"      ... ({total_lines - 60} more diff lines)")
        else:
            print(f"\n    [DRY RUN] No changes to {file_path}")
        return True

    # Backup original
    backup = file_path.with_suffix(file_path.suffix + ".rwql_backup")
    backup.write_text(original)
    file_path.write_text(new_content)
    return True


# ── Verification ─────────────────────────────────────────────────────────────

def verify_after_patch(config: dict, file_path: Path) -> tuple[bool, str]:
    """Run relevant checks after a patch is applied. Returns (pass, details)."""
    errors = []
    venv = config.get("venv")

    # Backend tests
    if config.get("test_cmd") and str(file_path).startswith(str(config["backend"])):
        rc, stdout, stderr = run_cmd(
            config["test_cmd"], cwd=config["backend"], venv=venv
        )
        if rc != 0:
            combined = stdout + stderr
            non_flaky_failures = any(
                "FAILED" in line and not any(f in line for f in FLAKY_TESTS)
                for line in combined.splitlines()
            )
            if non_flaky_failures:
                errors.append(f"Backend tests failed:\n{combined[-1000:]}")

    # Frontend typecheck
    frontend = config.get("frontend")
    if frontend and config.get("typecheck_cmd"):
        if str(file_path).startswith(str(frontend)):
            rc, stdout, stderr = run_cmd(
                config["typecheck_cmd"], cwd=frontend, timeout=120
            )
            if rc != 0:
                errors.append(f"TypeScript errors:\n{stderr[-1000:]}\n{stdout[-500:]}")

    if errors:
        return False, "\n".join(errors)
    return True, "All checks pass"


# ── Git Integration ──────────────────────────────────────────────────────────

def git_commit_patch(config: dict, file_path: Path, issue: dict, score: float):
    """Create a git commit for the applied patch."""
    root = config["root"]
    rel_path = file_path.relative_to(root)
    msg = (
        f"rwql: fix {issue.get('id', 'issue')} in {rel_path}\n\n"
        f"Category: {issue.get('category', 'unknown')}\n"
        f"Severity: {issue.get('severity', 'unknown')}\n"
        f"RWL Score: {score:.2f}\n"
        f"Description: {issue.get('description', 'N/A')[:200]}\n\n"
        f"Co-Authored-By: RWQL <noreply@rwql.local>"
    )
    run_cmd(["git", "add", str(rel_path)], cwd=root)
    run_cmd(["git", "commit", "-m", msg], cwd=root)


# ── Main Loop ─────────────────────────────────────────────────────────────────

async def run_quality_pass(projects: list[str], dry_run: bool,
                           client: anthropic.AsyncAnthropic):
    """One full quality pass: scan → patch → critique → apply → report."""
    start = time.time()
    all_results = []
    projects_data = []
    cost_tracker = CostTracker()
    memory = load_memory()

    for project_name in projects:
        if project_name not in PROJECTS:
            print(f"[!] Unknown project: {project_name}")
            continue

        config = PROJECTS[project_name]
        print(f"\n{'='*60}")
        print(f"  RWQL Pass: {project_name}")
        print(f"{'='*60}")

        proj_data = {"name": project_name, "issues": [], "results": [], "regressions": []}

        # 1. Scan
        issues = await scan_project(project_name, config, client, cost_tracker)
        if not issues:
            print(f"  No issues found in {project_name} — clean!")
            projects_data.append(proj_data)
            continue

        proj_data["issues"] = issues

        # Only process critical + high issues automatically; report others
        actionable = [i for i in issues if i["severity"] in ("critical", "high")]
        informational = [i for i in issues if i["severity"] in ("medium", "low")]

        print(f"  Actionable: {len(actionable)} | Informational: {len(informational)}")
        if informational:
            print(f"  Informational issues (not auto-patched):")
            for info in informational[:10]:
                print(f"    [{info['severity']}] {info.get('file', '?')}:{info.get('line', '?')}"
                      f" — {info['description'][:80]}")

        # 2-4. Patch → Critique → Refine for each actionable issue
        for issue in actionable[:MAX_PATCHES_PER_PASS]:
            print(f"\n  [{issue['severity'].upper()}] {issue.get('file', '?')}"
                  f":{issue.get('line', '?')} — {issue['description'][:80]}")

            file_path = resolve_file_path(config, issue.get("file", ""))
            if not file_path:
                print(f"    [skip] File not found: {issue.get('file')}")
                continue

            original = file_path.read_text(errors="ignore")

            # Generate patch
            print(f"    [patch] Generating fix...")
            patch = await generate_patch(issue, config, client, cost_tracker)
            if not patch:
                print(f"    [skip] No patch generated")
                continue

            # Ralph Wiggum Loop
            final_patch, score, critique = await ralph_wiggum_loop(
                issue, patch, original, client,
                cost_tracker=cost_tracker,
            )

            if score < 0.5:
                print(f"    [reject] Score too low ({score:.2f}), skipping")
                log_event({
                    "phase": "patch_rejected",
                    "project": project_name,
                    "issue_id": issue.get("id"),
                    "score": score,
                    "reason": "score < 0.5",
                    "critique": critique,
                })
                continue

            # Apply
            applied = apply_patch(file_path, original, final_patch, dry_run)
            if applied and not dry_run:
                # Verify
                passed, details = verify_after_patch(config, file_path)
                if not passed:
                    print(f"    [revert] Verification failed, reverting...")
                    print(f"    {details[:200]}")
                    backup = file_path.with_suffix(file_path.suffix + ".rwql_backup")
                    if backup.exists():
                        file_path.write_text(backup.read_text())
                        backup.unlink()
                    log_event({
                        "phase": "patch_reverted",
                        "project": project_name,
                        "file": str(file_path),
                        "reason": "verification_failed",
                        "details": details[:500],
                    })
                    memory["patches_reverted"].append({
                        "project": project_name,
                        "file": str(file_path),
                        "issue_id": issue.get("id"),
                        "reason": "verification_failed",
                        "date": datetime.now(timezone.utc).isoformat(),
                    })
                else:
                    print(f"    [success] Patch applied, verified (score={score:.2f})")
                    git_commit_patch(config, file_path, issue, score)
                    log_event({
                        "phase": "patch_applied",
                        "project": project_name,
                        "file": str(file_path),
                        "issue": issue.get("id"),
                        "score": score,
                        "critique": critique,
                    })
                    memory["patches_applied"].append({
                        "project": project_name,
                        "file": str(file_path),
                        "issue_id": issue.get("id"),
                        "score": score,
                        "date": datetime.now(timezone.utc).isoformat(),
                    })
                    backup = file_path.with_suffix(file_path.suffix + ".rwql_backup")
                    if backup.exists():
                        backup.unlink()

            result_entry = {
                "project": project_name,
                "issue": issue,
                "score": score,
                "applied": applied and not dry_run,
            }
            all_results.append(result_entry)
            proj_data["results"].append(result_entry)

        projects_data.append(proj_data)

    # Persist memory
    save_memory(memory)

    # Cost summary
    costs = cost_tracker.summary()
    elapsed = time.time() - start
    applied_count = sum(1 for r in all_results if r["applied"])

    print(f"\n{'='*60}")
    print(f"  RWQL Pass complete in {elapsed:.1f}s")
    print(f"  Patches attempted: {len(all_results)} | Applied: {applied_count}")
    print(f"  API: {costs['api_calls']} calls | "
          f"{costs['input_tokens']:,} in / {costs['output_tokens']:,} out | "
          f"~${costs['estimated_cost_usd']:.4f}")
    print(f"{'='*60}\n")

    log_event({
        "phase": "pass_complete",
        "projects": projects,
        "patches_attempted": len(all_results),
        "patches_applied": applied_count,
        "elapsed_s": round(elapsed, 1),
        "dry_run": dry_run,
        "cost": costs,
    })

    # Generate report
    generate_report(projects_data, costs, elapsed, dry_run)


async def run_scan_only(projects: list[str]):
    """Run all computational checks without API calls. Outputs findings to report."""
    start = time.time()
    projects_data = []

    for project_name in projects:
        if project_name not in PROJECTS:
            print(f"[!] Unknown project: {project_name}")
            continue

        config = PROJECTS[project_name]
        backend = config["backend"]
        frontend = config.get("frontend")
        venv = config.get("venv")

        print(f"\n{'='*60}")
        print(f"  RWQL Scan: {project_name}")
        print(f"{'='*60}")

        findings = []
        issue_count = 0

        # Tests
        if config.get("test_cmd"):
            test_timeout = config.get("test_timeout", 300)
            print(f"  [scan] Running tests (timeout {test_timeout}s)...")
            rc, stdout, stderr = run_cmd(config["test_cmd"], cwd=backend, venv=venv, timeout=test_timeout)
            if rc != 0:
                combined = filter_flaky_tests(f"{stdout}\n{stderr}")
                findings.append(("TESTS", "FAIL", combined[-1000:]))
                issue_count += 1
            else:
                findings.append(("TESTS", "PASS", stdout[-200:]))

        # Typecheck
        if frontend and config.get("typecheck_cmd"):
            print(f"  [scan] Running typecheck...")
            rc, stdout, stderr = run_cmd(config["typecheck_cmd"], cwd=frontend)
            if rc != 0:
                findings.append(("TYPECHECK", "FAIL", f"{stderr[-2000:]}\n{stdout[-500:]}"))
                issue_count += 1
            else:
                findings.append(("TYPECHECK", "PASS", ""))

        # Lint
        if config.get("lint_cmd"):
            print(f"  [scan] Running lint...")
            lint_cwd = config.get("lint_cwd", config["root"])
            rc, stdout, stderr = run_cmd(config["lint_cmd"], cwd=lint_cwd)
            if stdout.strip() and stdout.strip() != "[]":
                try:
                    lint_issues = json.loads(stdout)
                    lint_text = "\n".join(
                        f"  {i['filename']}:{i['location']['row']}: [{i['code']}] {i['message']}"
                        for i in lint_issues[:30]
                    )
                    findings.append(("LINT", f"{len(lint_issues)} issues", lint_text))
                    issue_count += len(lint_issues)
                except json.JSONDecodeError:
                    output = (stdout + stderr).strip()
                    if output and "No ESLint" not in output:
                        findings.append(("LINT", "OUTPUT", output[:2000]))
            else:
                findings.append(("LINT", "PASS", ""))

        # Security (bandit)
        if config.get("bandit_cmd") and shutil.which("bandit"):
            print(f"  [scan] Running security scan...")
            rc, stdout, stderr = run_cmd(config["bandit_cmd"], cwd=backend, venv=venv)
            if stdout.strip():
                try:
                    bandit_data = json.loads(stdout)
                    results = bandit_data.get("results", [])
                    if results:
                        sec_text = "\n".join(
                            f"  {r['filename']}:{r['line_number']}: [{r['test_id']}] "
                            f"({r['issue_severity']}) {r['issue_text']}"
                            for r in results[:20]
                        )
                        findings.append(("SECURITY", f"{len(results)} issues", sec_text))
                        issue_count += len(results)
                    else:
                        findings.append(("SECURITY", "PASS", ""))
                except json.JSONDecodeError:
                    findings.append(("SECURITY", "PASS", ""))

        # Dead imports — Python
        print(f"  [scan] Analyzing imports...")
        dead_py = find_dead_imports_py(backend)
        if "No obviously unused" not in dead_py:
            py_count = len(dead_py.strip().splitlines())
            findings.append(("DEAD PYTHON IMPORTS", f"{py_count} found", dead_py))
            issue_count += py_count
        else:
            findings.append(("DEAD PYTHON IMPORTS", "CLEAN", ""))

        # Dead imports — TypeScript
        if frontend:
            src_dir = frontend / "src" if (frontend / "src").exists() else frontend
            dead_ts = find_dead_imports_ts(src_dir)
            if "No obviously unused" not in dead_ts:
                ts_count = len(dead_ts.strip().splitlines())
                findings.append(("DEAD TS/TSX IMPORTS", f"{ts_count} found", dead_ts))
                issue_count += ts_count
            else:
                findings.append(("DEAD TS/TSX IMPORTS", "CLEAN", ""))

        # Packages
        for pkg in config.get("packages", []):
            if pkg.exists():
                print(f"  [scan] Analyzing package: {pkg.name}...")
                dead_pkg = find_dead_imports_ts(pkg / "src" if (pkg / "src").exists() else pkg)
                if "No obviously unused" not in dead_pkg:
                    pkg_count = len(dead_pkg.strip().splitlines())
                    findings.append((f"DEAD IMPORTS ({pkg.name})", f"{pkg_count} found", dead_pkg))
                    issue_count += pkg_count

        # Empty modules
        empty_py = find_empty_modules(backend, ("*.py",))
        if "No effectively empty" not in empty_py:
            findings.append(("EMPTY PYTHON MODULES", "found", empty_py))

        if frontend:
            src_dir = frontend / "src" if (frontend / "src").exists() else frontend
            empty_ts = find_empty_modules(src_dir, ("*.tsx", "*.ts"))
            if "No effectively empty" not in empty_ts:
                findings.append(("EMPTY TS/TSX MODULES", "found", empty_ts))

        # Print results
        print(f"\n  Results for {project_name}: {issue_count} issues")
        print(f"  {'-'*50}")
        for category, status, detail in findings:
            icon = "✓" if status == "PASS" or status == "CLEAN" else "✗"
            print(f"  {icon} {category}: {status}")
            if detail and status not in ("PASS", "CLEAN", ""):
                for line in detail.splitlines()[:10]:
                    print(f"    {line}")
                total = len(detail.splitlines())
                if total > 10:
                    print(f"    ... ({total - 10} more)")

        projects_data.append({
            "name": project_name,
            "issues": [{"severity": "info", "description": f"{c}: {s}", "detail": d}
                       for c, s, d in findings if s not in ("PASS", "CLEAN", "")],
            "results": [],
            "regressions": [],
        })

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  Scan complete in {elapsed:.1f}s")
    print(f"  No API calls made (--scan-only mode)")
    print(f"{'='*60}\n")

    generate_report(
        projects_data,
        {"api_calls": 0, "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0},
        elapsed,
        dry_run=True,
    )

    log_event({
        "phase": "scan_only_complete",
        "projects": [p["name"] for p in projects_data],
        "elapsed_s": round(elapsed, 1),
    })


async def main():
    parser = argparse.ArgumentParser(description="RWQL — Ralph Wiggum Quality Loop")
    parser.add_argument(
        "--project", default="all",
        help="Project to scan: calsanova, kiwi, or all",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print changes without applying")
    parser.add_argument("--once", action="store_true",
                        help="Run one pass and exit")
    parser.add_argument("--scan-only", action="store_true",
                        help="Run scans only (no API calls, no patching). Outputs findings to report.")
    args = parser.parse_args()

    projects = list(PROJECTS.keys()) if args.project == "all" else [args.project]

    if args.scan_only:
        print(f"RWQL — Scan Only Mode (no API calls)")
        print(f"Projects: {', '.join(projects)}")
        await run_scan_only(projects)
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set (use --scan-only to skip API calls)")
        sys.exit(1)

    client = anthropic.AsyncAnthropic(api_key=api_key)

    print(f"RWQL — Ralph Wiggum Quality Loop")
    print(f"Projects: {', '.join(projects)}")
    print(f"Dry run: {args.dry_run}")
    print(f"Mode: {'once' if args.once else f'loop every {LOOP_INTERVAL_SECONDS}s'}")
    print(f"Models: patch={MODEL_PATCH}, triage={MODEL_TRIAGE}, critique={MODEL_CRITIQUE}")

    if args.once:
        await run_quality_pass(projects, args.dry_run, client)
        return

    # Continuous loop
    while True:
        try:
            await run_quality_pass(projects, args.dry_run, client)
        except Exception as e:
            print(f"[!] Pass failed: {e}")
            log_event({"phase": "error", "error": str(e)})

        print(f"Next pass in {LOOP_INTERVAL_SECONDS // 60} minutes...")
        await asyncio.sleep(LOOP_INTERVAL_SECONDS)


if __name__ == "__main__":
    asyncio.run(main())
