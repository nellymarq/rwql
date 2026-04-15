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
import subprocess
import sys
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic

# ── Config ──────────────────────────────────────────────────────────────────

MODEL_PATCH = "claude-opus-4-6"          # Opus for patching (needs precision)
MODEL_TRIAGE = "claude-sonnet-4-6"       # Sonnet for scan triage (cheaper)
MODEL_CRITIQUE = "claude-sonnet-4-6"     # Sonnet for critique scoring (cheaper)
REFINEMENT_THRESHOLD = 0.72
LOG_FILE = Path(__file__).parent / "rwql_log.jsonl"
LOOP_INTERVAL_SECONDS = 1800  # 30 min default
MAX_PATCHES_PER_PASS = 5
MAX_FILE_CHARS = 12000  # Max chars to send in prompts (Opus handles large context well)

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
        "packages_test_cmd": ["npx", "vitest", "run", "--reporter=verbose"],
        "lint_cmd": [
            str(Path("/home/nelly/calsanova/backend/.venv/bin/ruff")),
            "check", "app/", "--output-format=json",
        ],
        "lint_cwd": Path("/home/nelly/calsanova/backend"),
        "typecheck_cmd": ["npx", "tsc", "--noEmit"],
        "venv": Path("/home/nelly/calsanova/backend/.venv"),
    },
    "kiwi": {
        "root": Path("/home/nelly/kiwi"),
        "backend": Path("/home/nelly/kiwi"),
        "frontend": None,
        "packages": [],
        "test_cmd": ["python3", "-m", "pytest", "tests/", "-x", "-q", "--tb=short"],
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

Given an issue description and the current file content, produce ONLY the fixed content.
- Make the smallest change that fixes the issue
- Never add features beyond the fix
- Never remove code unless it's provably dead/unused
- Maintain existing code style exactly
- For TypeScript/React files, preserve all existing imports and type annotations

Respond with the complete fixed file content. No preamble, no explanation, no markdown fences."""

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


# ── Utilities ────────────────────────────────────────────────────────────────

def log_event(event: dict):
    entry = {"ts": datetime.now(timezone.utc).isoformat(), **event}
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


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


def find_dead_imports_py(src_dir: Path) -> str:
    """Find unused imports in Python files."""
    lines = []
    for py_file in src_dir.rglob("*.py"):
        if should_skip(py_file):
            continue
        text = py_file.read_text(errors="ignore")
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                import_match = re.search(r"(?:import|from)\s+(\S+)", stripped)
                if import_match:
                    name = import_match.group(1).split(".")[0]
                    occurrences = text.count(name)
                    if occurrences == 1:
                        rel = py_file.relative_to(src_dir)
                        lines.append(f"{rel}:{lineno}: possibly unused: {stripped}")
    return "\n".join(lines[:50]) if lines else "No obviously unused imports found."


def find_dead_imports_ts(src_dir: Path) -> str:
    """Find unused imports in TypeScript/TSX files."""
    lines = []
    for ts_file in src_dir.rglob("*.tsx"):
        if should_skip(ts_file):
            continue
        text = ts_file.read_text(errors="ignore")
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("import "):
                # Extract imported names from: import { Foo, Bar } from "..."
                brace_match = re.search(r"\{([^}]+)\}", stripped)
                if brace_match:
                    names = [n.strip().split(" as ")[-1].strip()
                             for n in brace_match.group(1).split(",")]
                    for name in names:
                        if name and text.count(name) == 1:
                            rel = ts_file.relative_to(src_dir)
                            lines.append(f"{rel}:{lineno}: possibly unused import: {name}")
    for ts_file in src_dir.rglob("*.ts"):
        if should_skip(ts_file) or ts_file.suffix == ".tsx":
            continue
        text = ts_file.read_text(errors="ignore")
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("import "):
                brace_match = re.search(r"\{([^}]+)\}", stripped)
                if brace_match:
                    names = [n.strip().split(" as ")[-1].strip()
                             for n in brace_match.group(1).split(",")]
                    for name in names:
                        if name and text.count(name) == 1:
                            rel = ts_file.relative_to(src_dir)
                            lines.append(f"{rel}:{lineno}: possibly unused import: {name}")
    return "\n".join(lines[:50]) if lines else "No obviously unused TS/TSX imports found."


def find_empty_modules(src_dir: Path, extensions: tuple[str, ...] = ("*.py",)) -> str:
    """Find files that are effectively empty (< 5 lines of real code)."""
    empty = []
    for ext in extensions:
        for f in src_dir.rglob(ext):
            if should_skip(f):
                continue
            text = f.read_text(errors="ignore")
            real_lines = [l for l in text.splitlines()
                          if l.strip() and not l.strip().startswith("#")
                          and not l.strip().startswith("//")]
            if len(real_lines) <= 3:
                rel = f.relative_to(src_dir)
                empty.append(f"{rel}: {len(real_lines)} real lines")
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
                       client: anthropic.AsyncAnthropic) -> list[dict]:
    """Run all scanners and ask Claude to triage the findings."""
    backend = config["backend"]
    frontend = config.get("frontend")
    venv = config.get("venv")
    findings = []

    # ── Pre-flight: verify tests pass before scanning ──
    if config.get("test_cmd"):
        print(f"\n  [preflight] Verifying {project_name} tests pass...")
        rc, stdout, stderr = run_cmd(config["test_cmd"], cwd=backend, venv=venv)
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

    # ── Parallel scan: typecheck + lint + static analysis ──

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

    scan_results = await asyncio.gather(
        run_typecheck(), run_lint(), run_static_analysis()
    )

    for result in scan_results:
        if result is None:
            continue
        if isinstance(result, list):
            findings.extend(result)
        else:
            findings.append(result)

    combined = "\n\n".join(findings)

    # Ask Claude to triage
    print(f"  [scan] Claude triaging {project_name} findings...")
    response = await client.messages.create(
        model=MODEL_TRIAGE,
        max_tokens=4096,
        system=SCAN_SYSTEM,
        messages=[{"role": "user", "content": f"Project: {project_name}\n\n{combined}"}],
    )

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

async def generate_patch(issue: dict, project_config: dict,
                         client: anthropic.AsyncAnthropic) -> str | None:
    """Generate a code patch for an issue."""
    file_path = resolve_file_path(project_config, issue.get("file", ""))
    if not file_path:
        return None

    current_content = file_path.read_text(errors="ignore")
    # Truncate very large files but try to include the relevant section
    content_for_prompt = current_content
    if len(content_for_prompt) > MAX_FILE_CHARS:
        target_line = issue.get("line", 0)
        if target_line > 0:
            lines = content_for_prompt.splitlines(keepends=True)
            # Center window around the target line
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
        max_tokens=16000,
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

    patch = next((b.text for b in response.content if hasattr(b, "text")), "").strip()

    # Strip any accidental markdown fences
    if patch.startswith("```"):
        patch = re.sub(r"^```\w*\n?", "", patch)
        patch = re.sub(r"\n?```$", "", patch)

    return patch if patch else None


# ── Ralph Wiggum Loop ────────────────────────────────────────────────────────

async def ralph_wiggum_loop(
    issue: dict,
    patch: str,
    original_content: str,
    client: anthropic.AsyncAnthropic,
    max_refinements: int = 4,
) -> tuple[str, float, dict]:
    """
    Critique a patch and refine it if score < REFINEMENT_THRESHOLD.
    Returns (final_patch, score, critique_data).
    """
    current_patch = patch
    final_score = 0.0
    final_critique: dict = {}

    # Truncate for prompt to avoid token bloat
    orig_for_prompt = original_content[:MAX_FILE_CHARS]
    patch_for_prompt = current_patch[:MAX_FILE_CHARS]

    for attempt in range(max_refinements + 1):
        critique_response = await client.messages.create(
            model=MODEL_CRITIQUE,
            max_tokens=2048,
            system=CRITIQUE_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"""Issue: {issue['description']}

Original file:
```
{orig_for_prompt}
```

Proposed patch:
```
{patch_for_prompt}
```"""
            }],
        )

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

                # Refine using Opus
                print(f"    [rwl] Score below threshold "
                      f"({final_score:.2f} < {REFINEMENT_THRESHOLD}), refining...")
                refinement_response = await client.messages.create(
                    model=MODEL_PATCH,
                    max_tokens=16000,
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

Previous (rejected) patch:
```
{patch_for_prompt}
```

Generate an improved patch that addresses the critical issues."""
                    }],
                )
                current_patch = next(
                    (b.text for b in refinement_response.content if hasattr(b, "text")),
                    current_patch,
                ).strip()

                if current_patch.startswith("```"):
                    current_patch = re.sub(r"^```\w*\n?", "", current_patch)
                    current_patch = re.sub(r"\n?```$", "", current_patch)
                patch_for_prompt = current_patch[:MAX_FILE_CHARS]
        except (json.JSONDecodeError, ValueError):
            break

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

    for project_name in projects:
        if project_name not in PROJECTS:
            print(f"[!] Unknown project: {project_name}")
            continue

        config = PROJECTS[project_name]
        print(f"\n{'='*60}")
        print(f"  RWQL Pass: {project_name}")
        print(f"{'='*60}")

        # 1. Scan
        issues = await scan_project(project_name, config, client)
        if not issues:
            print(f"  No issues found in {project_name} — clean!")
            continue

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
            patch = await generate_patch(issue, config, client)
            if not patch:
                print(f"    [skip] No patch generated")
                continue

            # Ralph Wiggum Loop
            final_patch, score, critique = await ralph_wiggum_loop(
                issue, patch, original, client
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
                else:
                    print(f"    [success] Patch applied, verified (score={score:.2f})")
                    # Git commit
                    git_commit_patch(config, file_path, issue, score)
                    log_event({
                        "phase": "patch_applied",
                        "project": project_name,
                        "file": str(file_path),
                        "issue": issue.get("id"),
                        "score": score,
                        "critique": critique,
                    })
                    # Clean up backup
                    backup = file_path.with_suffix(file_path.suffix + ".rwql_backup")
                    if backup.exists():
                        backup.unlink()

            all_results.append({
                "project": project_name,
                "issue": issue,
                "score": score,
                "applied": applied and not dry_run,
            })

    elapsed = time.time() - start
    applied_count = sum(1 for r in all_results if r["applied"])
    print(f"\n{'='*60}")
    print(f"  RWQL Pass complete in {elapsed:.1f}s")
    print(f"  Patches attempted: {len(all_results)} | Applied: {applied_count}")
    print(f"{'='*60}\n")

    log_event({
        "phase": "pass_complete",
        "projects": projects,
        "patches_attempted": len(all_results),
        "patches_applied": applied_count,
        "elapsed_s": round(elapsed, 1),
        "dry_run": dry_run,
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
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = anthropic.AsyncAnthropic(api_key=api_key)

    projects = list(PROJECTS.keys()) if args.project == "all" else [args.project]

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
