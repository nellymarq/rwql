# RWQL — Ralph Wiggum Quality Loop

Autonomous quality agent for NutriForge and Kiwi. Periodically scans for test failures, dead code, lint errors, and unused files — generates patches, critiques them through the Ralph Wiggum self-critique loop, and applies fixes that pass the quality threshold.

## Components

| Script | Purpose |
|--------|---------|
| `rwql.py` | Main quality loop — scan, patch, critique, apply |
| `purge.py` | Dead code and file cleaner |

## Usage

```bash
# Set API key
export ANTHROPIC_API_KEY=sk-...

# Run one quality pass (dry-run first)
python3 rwql.py --once --dry-run

# Run one real pass on NutriForge
python3 rwql.py --project nutriforge --once

# Continuous loop (every 30 min)
python3 rwql.py --project all

# Clean up dead code and generated artifacts
python3 purge.py --dry-run
python3 purge.py --project nutriforge
```

## Architecture

```
SCAN → TRIAGE → PATCH → [RALPH WIGGUM LOOP] → APPLY → REPORT
                              ↑         ↓
                         CRITIQUE  REFINE (if score < 0.72)
```

### Ralph Wiggum Loop Dimensions
- **correctness** — Does it fix the issue without new bugs?
- **minimality** — Smallest change that works?
- **safety** — No new security/stability risks?
- **style_preservation** — Matches codebase style?
- **test_coverage** — Addresses root cause?

### Safety Features
- Dry-run mode before applying anything
- Test suite verification after each patch
- Automatic rollback if tests fail
- Score threshold of 0.72 — rejects low-quality patches
- Hard cap of 5 patches per pass
- Backup files before overwriting

## Log

All events are appended to `rwql_log.jsonl` in JSONL format.
