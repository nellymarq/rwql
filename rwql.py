"""
RWQL — Ralph Wiggum Quality Loop
===================================
An autonomous self-improving quality agent for NutriForge + Kiwi.

Pipeline:
  1. SCAN  — discover test failures, dead code, lint issues, unused imports
  2. TRIAGE — rank issues by severity using RWL critique scoring
  3. PATCH  — generate fixes via Claude claude-opus-4-6 with adaptive thinking
  4. CRITIQUE — Ralph Wiggum Loop: score the patch (0-1), refine if < THRESHOLD
  5. APPLY  — write files, run tests to confirm fix
  6. REPORT — human-readable summary, persisted to rwql_log.jsonl

Usage:
  python3 rwql.py [--project nutriforge|kiwi|all] [--dry-run] [--once]

  --project: which project to scan (default: all)
  --dry-run: print proposed changes without writing
  --once:    run one pass then exit (default: loop every 30min)

Environment:
  ANTHROPIC_API_KEY — required
"""

import asyncio
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

MODEL = "claude-opus-4-6"
REFINEMENT_THRESHOLD = 0.72
LOG_FILE = Path(__file__).parent / "rwql_log.jsonl"
LOOP_INTERVAL_SECONDS = 1800  # 30 min default

PROJECTS = {
    "nutriforge": {
        "root": Path("/home/nelly/nutriforge"),
        "backend": Path("/home/nelly/nutriforge/backend"),
        "frontend": Path("/home/nelly/nutriforge/apps/web"),
        "test_cmd": ["python3", "-m", "pytest", "tests/", "-v", "--tb=short"],
        "lint_cmd": ["python3", "-m", "ruff", "check", "app/", "--output-format=json"],
        "typecheck_cmd": None,  # No mypy configured
        "venv": Path("/home/nelly/nutriforge/backend/.venv"),
    },
    "kiwi": {
        "root": Path("/home/nelly/kiwi"),
        "backend": Path("/home/nelly/kiwi"),
        "frontend": None,
        "test_cmd": ["python3", "-m", "pytest", "tests/", "-v", "--tb=short"],
        "lint_cmd": ["python3", "-m", "ruff", "check", ".", "--output-format=json"],
        "typecheck_cmd": None,
        "venv": None,
    },
}

SCAN_SYSTEM = """You are RWQL's scanner — an expert code auditor for Python (FastAPI/SQLAlchemy) and TypeScript (Next.js) projects.

Given raw tool output (test failures, lint errors, imports, dead code analysis), produce a prioritized JSON list of issues.

Respond ONLY with a JSON object:
{
  "issues": [
    {
      "id": "unique-slug",
      "severity": "critical|high|medium|low",
      "category": "test_failure|dead_code|lint|type_error|security|architecture",
      "file": "relative/path/to/file.py",
      "line": 42,
      "description": "Clear description of the problem",
      "suggested_fix": "Specific, actionable fix description"
    }
  ],
  "scan_summary": "1-2 sentence overview of findings"
}

Severity rules:
- critical: test failure, security issue, runtime crash risk
- high: unused imports in active files, broken patterns, logic bugs
- medium: dead code, unnecessary complexity, missing error handling
- low: style issues, minor cleanup"""

PATCH_SYSTEM = """You are RWQL's patch engineer — you write precise, minimal code fixes.

Given an issue description and the current file content, produce ONLY the fixed content.
- Make the smallest change that fixes the issue
- Never add features beyond the fix
- Never remove code unless it's provably dead/unused
- Maintain existing code style exactly

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


def run_cmd(cmd: list[str], cwd: Path, env: dict | None = None) -> tuple[int, str, str]:
    """Run a shell command, return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, **(env or {})},
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "TIMEOUT"
    except FileNotFoundError as e:
        return 1, "", str(e)


def find_dead_imports(src_dir: Path) -> str:
    """Find unused imports using basic grep analysis."""
    lines = []
    for py_file in src_dir.rglob("*.py"):
        if ".venv" in str(py_file):
            continue
        text = py_file.read_text(errors="ignore")
        # Find import lines
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                # Check if the imported name appears elsewhere in the file
                import_match = re.search(r"(?:import|from)\s+(\S+)", stripped)
                if import_match:
                    name = import_match.group(1).split(".")[0]
                    occurrences = text.count(name)
                    if occurrences == 1:  # Only in the import line itself
                        lines.append(f"{py_file.relative_to(src_dir)}:{lineno}: possibly unused: {stripped}")
    return "\n".join(lines[:50]) if lines else "No obviously unused imports found."


def find_empty_modules(src_dir: Path) -> str:
    """Find Python files that are effectively empty (< 5 lines of real code)."""
    empty = []
    for py_file in src_dir.rglob("*.py"):
        if ".venv" in str(py_file):
            continue
        text = py_file.read_text(errors="ignore")
        real_lines = [l for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
        if len(real_lines) <= 3:
            empty.append(f"{py_file.relative_to(src_dir)}: {len(real_lines)} real lines")
    return "\n".join(empty) if empty else "No effectively empty modules found."


# ── Scan Phase ───────────────────────────────────────────────────────────────

async def scan_project(project_name: str, config: dict, client: anthropic.AsyncAnthropic) -> list[dict]:
    """Run all scanners and ask Claude to triage the findings."""
    backend = config["backend"]
    findings = []

    print(f"\n  [scan] Running tests for {project_name}...")
    rc, stdout, stderr = run_cmd(config["test_cmd"], cwd=backend)
    if rc != 0:
        findings.append(f"=== TEST FAILURES ===\n{stdout}\n{stderr}")
    else:
        findings.append(f"=== TESTS PASS ===\n{stdout[-500:]}")

    # Lint (ruff — may not be installed)
    print(f"  [scan] Running lint for {project_name}...")
    rc, stdout, stderr = run_cmd(config["lint_cmd"], cwd=backend)
    if stdout.strip() and stdout.strip() != "[]":
        try:
            lint_issues = json.loads(stdout)
            # Summarize
            findings.append(f"=== LINT ISSUES ({len(lint_issues)}) ===\n" +
                          "\n".join(f"{i['filename']}:{i['location']['row']}: [{i['code']}] {i['message']}"
                                    for i in lint_issues[:20]))
        except json.JSONDecodeError:
            findings.append(f"=== LINT OUTPUT ===\n{stdout[:2000]}")

    # Dead imports
    print(f"  [scan] Analyzing imports for {project_name}...")
    dead_imports = find_dead_imports(backend)
    findings.append(f"=== DEAD IMPORT ANALYSIS ===\n{dead_imports}")

    # Empty modules
    empty = find_empty_modules(backend)
    findings.append(f"=== EMPTY MODULE ANALYSIS ===\n{empty}")

    combined = "\n\n".join(findings)

    # Ask Claude to triage
    print(f"  [scan] Claude triaging {project_name} findings...")
    response = await client.messages.create(
        model=MODEL,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=SCAN_SYSTEM,
        messages=[{"role": "user", "content": f"Project: {project_name}\n\n{combined}"}],
    )

    # Extract text content (skip thinking blocks)
    text = next((b.text for b in response.content if hasattr(b, "text")), "")

    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            data = json.loads(m.group())
            issues = data.get("issues", [])
            summary = data.get("scan_summary", "")
            print(f"  [scan] {len(issues)} issues found: {summary}")
            log_event({"phase": "scan", "project": project_name, "issue_count": len(issues), "summary": summary})
            return issues
    except (json.JSONDecodeError, AttributeError):
        pass

    print(f"  [scan] No structured issues from triage")
    return []


# ── Patch Phase ──────────────────────────────────────────────────────────────

async def generate_patch(issue: dict, project_config: dict, client: anthropic.AsyncAnthropic) -> str | None:
    """Generate a code patch for an issue."""
    backend = project_config["backend"]
    file_path = backend / issue.get("file", "")

    if not file_path.exists() or not file_path.is_file():
        return None

    current_content = file_path.read_text(errors="ignore")

    response = await client.messages.create(
        model=MODEL,
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
{current_content}
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
    max_refinements: int = 2,
) -> tuple[str, float]:
    """
    Critique a patch and refine it if score < REFINEMENT_THRESHOLD.
    Returns (final_patch, score).
    """
    current_patch = patch
    final_score = 0.0

    for attempt in range(max_refinements + 1):
        # Critique
        critique_response = await client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=CRITIQUE_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"""Issue: {issue['description']}

Original file:
```
{original_content[:3000]}
```

Proposed patch:
```
{current_patch[:3000]}
```"""
            }],
        )

        critique_text = next((b.text for b in critique_response.content if hasattr(b, "text")), "")

        try:
            m = re.search(r"\{.*\}", critique_text, re.DOTALL)
            if m:
                critique = json.loads(m.group())
                final_score = float(critique.get("score", 0.0))
                needs_refinement = critique.get("needs_refinement", False)
                critical_issues = critique.get("critical_issues", [])

                print(f"    [rwl] Attempt {attempt + 1}: score={final_score:.2f}, "
                      f"refine={needs_refinement}")

                if final_score >= REFINEMENT_THRESHOLD or not needs_refinement or attempt == max_refinements:
                    break

                # Refine
                print(f"    [rwl] Score below threshold ({final_score:.2f} < {REFINEMENT_THRESHOLD}), refining...")
                refinement_response = await client.messages.create(
                    model=MODEL,
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
{original_content[:3000]}
```

Previous (rejected) patch:
```
{current_patch[:3000]}
```

Generate an improved patch that addresses the critical issues."""
                    }],
                )
                current_patch = next(
                    (b.text for b in refinement_response.content if hasattr(b, "text")), current_patch
                ).strip()

                # Strip markdown fences
                if current_patch.startswith("```"):
                    current_patch = re.sub(r"^```\w*\n?", "", current_patch)
                    current_patch = re.sub(r"\n?```$", "", current_patch)
        except (json.JSONDecodeError, ValueError):
            break

    return current_patch, final_score


# ── Apply Phase ───────────────────────────────────────────────────────────────

def apply_patch(file_path: Path, new_content: str, dry_run: bool) -> bool:
    """Write the patch to disk (or print it in dry-run mode)."""
    if dry_run:
        print(f"\n    [DRY RUN] Would write {file_path}")
        print(f"    First 200 chars: {new_content[:200]}...")
        return True

    # Backup original
    backup = file_path.with_suffix(file_path.suffix + ".rwql_backup")
    backup.write_text(file_path.read_text(errors="ignore"))

    file_path.write_text(new_content)
    return True


# ── Main Loop ─────────────────────────────────────────────────────────────────

async def run_quality_pass(projects: list[str], dry_run: bool, client: anthropic.AsyncAnthropic):
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

        # 2-4. Patch → Critique → Refine for each actionable issue
        for issue in actionable[:5]:  # Cap at 5 patches per pass
            print(f"\n  [{issue['severity'].upper()}] {issue['file']}:{issue.get('line','?')} "
                  f"— {issue['description'][:80]}")

            backend = config["backend"]
            file_path = backend / issue.get("file", "")

            if not file_path.exists():
                print(f"    [skip] File not found: {file_path}")
                continue

            original = file_path.read_text(errors="ignore")

            # Generate patch
            print(f"    [patch] Generating fix...")
            patch = await generate_patch(issue, config, client)
            if not patch:
                print(f"    [skip] No patch generated")
                continue

            # Ralph Wiggum Loop
            final_patch, score = await ralph_wiggum_loop(issue, patch, original, client)

            if score < 0.5:
                print(f"    [reject] Score too low ({score:.2f}), skipping")
                log_event({
                    "phase": "patch_rejected",
                    "project": project_name,
                    "issue_id": issue.get("id"),
                    "score": score,
                    "reason": "score < 0.5",
                })
                continue

            # Apply
            applied = apply_patch(file_path, final_patch, dry_run)
            if applied and not dry_run:
                # Verify tests still pass
                rc, stdout, _ = run_cmd(config["test_cmd"], cwd=backend)
                if rc != 0:
                    print(f"    [revert] Tests failed after patch, reverting...")
                    backup = file_path.with_suffix(file_path.suffix + ".rwql_backup")
                    if backup.exists():
                        file_path.write_text(backup.read_text())
                    log_event({
                        "phase": "patch_reverted",
                        "project": project_name,
                        "file": str(file_path),
                        "reason": "tests_failed",
                    })
                else:
                    print(f"    [success] Patch applied, tests pass (score={score:.2f})")
                    log_event({
                        "phase": "patch_applied",
                        "project": project_name,
                        "file": str(file_path),
                        "issue": issue.get("id"),
                        "score": score,
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
    print(f"\n{'='*60}")
    print(f"  RWQL Pass complete in {elapsed:.1f}s | {len(all_results)} patches attempted")
    print(f"{'='*60}\n")

    log_event({
        "phase": "pass_complete",
        "projects": projects,
        "patches_attempted": len(all_results),
        "elapsed_s": round(elapsed, 1),
        "dry_run": dry_run,
    })


async def main():
    parser = argparse.ArgumentParser(description="RWQL — Ralph Wiggum Quality Loop")
    parser.add_argument("--project", default="all", help="Project to scan: nutriforge, kiwi, or all")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without applying")
    parser.add_argument("--once", action="store_true", help="Run one pass and exit")
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
