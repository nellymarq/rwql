# RWQL — Ralph Wiggum Quality Loop

Autonomous quality agent for Calsanova and Kiwi. Periodically scans for test failures, type errors, lint issues, dead code, and unused imports — generates patches, critiques them through the Ralph Wiggum self-critique loop, and applies fixes that pass the quality threshold.

## Components

| Script | Purpose |
|--------|---------|
| `rwql.py` | Main quality loop — scan, patch, critique, apply |
| `kwql.py` | Kiwi-specific quality loop |
| `kiwi_autonomy.py` | Kiwi autonomous agent |
| `purge.py` | Dead code and file cleaner |

## Usage

```bash
# Set API key
export ANTHROPIC_API_KEY=sk-...

# Run one quality pass (dry-run first)
python3 rwql.py --project calsanova --once --dry-run

# Run one real pass on Calsanova
python3 rwql.py --project calsanova --once

# Run against all projects
python3 rwql.py --once

# Continuous loop (every 30 min)
python3 rwql.py --project calsanova
```

## Architecture

```
SCAN → TRIAGE → PATCH → [RALPH WIGGUM LOOP] → APPLY → VERIFY → COMMIT
                              ↑         ↓
                         CRITIQUE  REFINE (if score < 0.72)
```

### Scan Phase
- **Backend tests**: pytest -x -q --tb=short
- **Frontend typecheck**: tsc --noEmit (Calsanova)
- **Lint**: ruff check (Python)
- **Dead imports**: Python + TypeScript/TSX
- **Empty modules**: Python + TypeScript/TSX
- Known-flaky tests are annotated and deprioritized

### Model Strategy
- **Opus** for patching and refinement (precision matters)
- **Sonnet** for scan triage and critique scoring (fast, cheap)

### Ralph Wiggum Loop Dimensions
- **correctness** — Does it fix the issue without new bugs?
- **minimality** — Smallest change that works?
- **safety** — No new security/stability risks?
- **style_preservation** — Matches codebase style?
- **test_coverage** — Addresses root cause?

### Safety Features
- Dry-run mode with unified diff output
- Test suite + typecheck verification after each patch
- Automatic rollback if verification fails
- Score threshold of 0.72 — rejects low-quality patches
- Hard reject below 0.5
- Max 5 patches per pass
- Backup files before overwriting
- Git commit per applied patch with metadata
- Known-flaky test exclusion (avoids wasting API calls)

## Projects

| Project | Backend | Frontend | Tests | Lint |
|---------|---------|----------|-------|------|
| calsanova | FastAPI + SQLAlchemy | Next.js 15 + React 19 | pytest + tsc | ruff |
| kiwi | Python agents | — | pytest | — |

## Log

All events are appended to `rwql_log.jsonl` in JSONL format.
