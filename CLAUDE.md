# RWQL — Ralph Wiggum Quality Loop

## Quick Reference
- Dry run: `python3 rwql.py --project calsanova --once --dry-run`
- Real pass: `python3 rwql.py --project calsanova --once`
- Scythene: `python3 rwql.py --project scythene --once --dry-run`
- All projects: `python3 rwql.py --once`
- Continuous: `python3 rwql.py --project calsanova`
- Purge: `python3 purge.py --project calsanova --dry-run`
- GitHub: `nellymarq/rwql`
- Also contains: `kwql.py`, `kiwi_autonomy.py`, `purge.py`

## Projects
- **calsanova** — FastAPI backend + Next.js 15 frontend + packages/core (247 tests)
- **kiwi** — Python research agents (825 tests)
- **scythene** — Next.js 16 + Stripe DTC supplement store (tsc + next lint)

## Model config
- Patching: claude-opus-4-6 (precision)
- Triage + critique: claude-sonnet-4-6 (cost)

## Key behaviors
- Pre-flight validation: tests must pass BEFORE scanning (prevents fixing pre-existing failures)
- Venv activation: backend tests use project venv automatically
- Parallel scanning: typecheck + lint + static analysis run concurrently
- Refinement: up to 4 critique rounds (actor-critic pattern)
- Score threshold: 0.72 to apply, hard reject below 0.5
