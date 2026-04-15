"""
KWQL — Knowledge Quality Loop for Kiwi
=========================================
Autonomous knowledge extraction from Kiwi's episodic memory into semantic memory.

Pipeline:
  1. SCAN    — identify unprocessed high-quality episodic exchanges
  2. TRIAGE  — rank knowledge gaps by priority, cap at N per pass
  3. EXTRACT — synthesize reusable knowledge entries via Claude
  4. CRITIQUE — Knowledge Wiggum Loop (KWL): score on 5 dimensions, refine if needed
  5. APPLY   — write passing entries to semantic memory via KiwiMemory

Usage:
  python3 kwql.py --once --dry-run          # Preview without writing
  python3 kwql.py --once                    # Run one pass, apply
  python3 kwql.py                           # Loop every 30min
  python3 kwql.py --min-score 0.8           # Only process high-quality exchanges
  python3 kwql.py --max-entries 3           # Cap extractions per pass

Environment:
  ANTHROPIC_API_KEY — required
"""

import asyncio
import json
import os
import re
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
DIMENSION_FLOOR = 0.50
REJECT_FLOOR = 0.50
MAX_REFINEMENTS = 2
DEFAULT_MIN_EPISODIC_SCORE = 0.7
DEFAULT_MAX_ENTRIES = 5
LOOP_INTERVAL_SECONDS = 1800  # 30 min

LOG_FILE = Path(__file__).parent / "kwql_log.jsonl"
MEMORY_JSON = Path.home() / ".kiwi" / "memory.json"

# Add Kiwi to path so we can import KiwiMemory
sys.path.insert(0, str(Path("/home/nelly/kiwi")))


# ── Prompts ─────────────────────────────────────────────────────────────────

SCAN_SYSTEM = """You are KWQL's scanner — an expert at identifying valuable knowledge in research conversations.

Given a list of episodic research exchanges and the current semantic topic list, identify knowledge gaps: topics covered in the exchanges that are NOT yet captured (or are outdated) in semantic memory.

Respond ONLY with JSON:
{
  "gaps": [
    {
      "source_ts": "timestamp of the episodic exchange",
      "priority": "high|medium|low",
      "proposed_topic": "concise topic key (lowercase, descriptive)",
      "rationale": "Why this should be extracted into semantic memory"
    }
  ],
  "scan_summary": "1-2 sentence overview"
}

Priority rules:
- high: actionable, evidence-based knowledge with specific recommendations
- medium: useful context or background that enriches future answers
- low: tangential or already partially covered"""

EXTRACT_SYSTEM = """You are KWQL's knowledge extractor — you synthesize clean, reusable knowledge from research conversations.

Given an episodic exchange (query + response), extract a concise knowledge entry suitable for future context injection.

Rules:
- Focus on factual, evidence-based content
- Include specific numbers, citations, or dosages when present
- Cap at 2000 characters
- Write in a neutral, reference-style tone (not conversational)
- If existing semantic content for this topic is provided, MERGE new insights with existing knowledge — don't discard what's already there

Respond ONLY with JSON:
{
  "topic": "the topic key (lowercase)",
  "knowledge": "the extracted knowledge text (max 2000 chars)"
}"""

CRITIQUE_SYSTEM = """You are KWQL's critic — the Knowledge Wiggum Loop evaluator.

Score the proposed knowledge entry on these dimensions (0.0 to 1.0):
- accuracy (weight 0.25): Factually correct per the source exchange?
- completeness (weight 0.25): Captures key insights, not just surface-level?
- clarity (weight 0.25): Written clearly enough for future context injection?
- coherence (weight 0.15): Fits with existing semantic memory, no contradictions?
- applicability (weight 0.10): Practically useful for future queries?

Respond ONLY with JSON:
{
  "score": 0.85,
  "dimensions": {
    "accuracy": 0.9,
    "completeness": 0.8,
    "clarity": 0.9,
    "coherence": 0.8,
    "applicability": 0.8
  },
  "issues": ["list of problems if score < 0.72"],
  "strengths": ["what the entry does well"],
  "refinement_guidance": "specific instructions for improvement if needed"
}"""


# ── Utilities ────────────────────────────────────────────────────────────────

def log_event(event: dict):
    entry = {"ts": datetime.now(timezone.utc).isoformat(), **event}
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def load_memory() -> dict[str, Any]:
    """Load memory.json read-only."""
    if not MEMORY_JSON.exists():
        print("[!] No memory.json found at", MEMORY_JSON)
        return {"episodic": [], "semantic": {}}
    try:
        return json.loads(MEMORY_JSON.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"[!] Failed to load memory.json: {e}")
        return {"episodic": [], "semantic": {}}


def get_processed_timestamps() -> set[str]:
    """Read kwql_log.jsonl to find already-processed episodic timestamps."""
    processed = set()
    if not LOG_FILE.exists():
        return processed
    for line in LOG_FILE.read_text().splitlines():
        try:
            entry = json.loads(line)
            if entry.get("phase") in ("applied", "rejected"):
                ts = entry.get("source_ts")
                if ts:
                    processed.add(ts)
        except json.JSONDecodeError:
            continue
    return processed


def extract_json(text: str) -> dict | None:
    """Extract first JSON object from Claude's response."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def get_text(response) -> str:
    """Extract text content from Claude response, skipping thinking blocks."""
    return next((b.text for b in response.content if hasattr(b, "text")), "")


# ── Phase 1: SCAN ────────────────────────────────────────────────────────────

async def scan_gaps(
    episodic: list[dict],
    semantic: dict,
    processed: set[str],
    min_score: float,
    client: anthropic.AsyncAnthropic,
) -> list[dict]:
    """Identify knowledge gaps from unprocessed episodic exchanges."""
    # Filter: quality >= min_score AND not already processed
    unprocessed = [
        e for e in episodic
        if e.get("quality_score", 0) >= min_score
        and e.get("ts") not in processed
    ]

    if not unprocessed:
        print("  [scan] No unprocessed high-quality exchanges found")
        return []

    print(f"  [scan] {len(unprocessed)} unprocessed exchanges (min_score={min_score})")

    # Format for Claude
    exchanges_text = "\n\n".join(
        f"[{e['ts']}] (score: {e.get('quality_score', '?')})\n"
        f"Q: {e.get('query', '')[:300]}\n"
        f"A: {e.get('response_preview', '')[:500]}"
        for e in unprocessed
    )

    semantic_topics = list(semantic.keys()) if semantic else []
    topics_text = ", ".join(semantic_topics) if semantic_topics else "(none)"

    response = await client.messages.create(
        model=MODEL,
        max_tokens=2048,
        thinking={"type": "adaptive"},
        system=SCAN_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Existing semantic topics: {topics_text}\n\n"
                       f"Unprocessed episodic exchanges:\n{exchanges_text}"
        }],
    )

    data = extract_json(get_text(response))
    if not data:
        print("  [scan] No structured response from Claude")
        return []

    gaps = data.get("gaps", [])
    summary = data.get("scan_summary", "")
    print(f"  [scan] {len(gaps)} gaps identified: {summary}")
    log_event({"phase": "scan", "episodic_count": len(unprocessed), "gaps_found": len(gaps)})
    return gaps


# ── Phase 2: TRIAGE ──────────────────────────────────────────────────────────

def triage_gaps(gaps: list[dict], max_entries: int) -> list[dict]:
    """Sort gaps by priority and cap at max_entries."""
    priority_order = {"high": 0, "medium": 1, "low": 2}
    sorted_gaps = sorted(gaps, key=lambda g: priority_order.get(g.get("priority", "low"), 2))
    triaged = sorted_gaps[:max_entries]
    print(f"  [triage] {len(triaged)} gaps selected (max {max_entries})")
    return triaged


# ── Phase 3: EXTRACT ─────────────────────────────────────────────────────────

async def extract_knowledge(
    gap: dict,
    episodic: list[dict],
    semantic: dict,
    client: anthropic.AsyncAnthropic,
) -> dict | None:
    """Extract a knowledge entry from the source episodic exchange."""
    source_ts = gap.get("source_ts", "")
    proposed_topic = gap.get("proposed_topic", "")

    # Find the source exchange
    source = next((e for e in episodic if e.get("ts") == source_ts), None)
    if not source:
        print(f"    [extract] Source exchange not found for ts={source_ts}")
        return None

    # Check if topic already exists in semantic memory
    existing = ""
    if proposed_topic and proposed_topic.lower().strip() in semantic:
        existing_entry = semantic[proposed_topic.lower().strip()]
        existing = existing_entry.get("content", "") if isinstance(existing_entry, dict) else str(existing_entry)

    existing_block = ""
    if existing:
        existing_block = f"\n\nExisting semantic content for '{proposed_topic}':\n{existing[:1500]}\n\nMerge new insights with existing content."

    response = await client.messages.create(
        model=MODEL,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=EXTRACT_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Proposed topic: {proposed_topic}\n\n"
                       f"Source exchange:\n"
                       f"Q: {source.get('query', '')}\n"
                       f"A: {source.get('response_preview', '')}"
                       f"{existing_block}"
        }],
    )

    data = extract_json(get_text(response))
    if not data or "knowledge" not in data:
        print(f"    [extract] No structured extraction for '{proposed_topic}'")
        return None

    topic = data.get("topic", proposed_topic).lower().strip()
    knowledge = data["knowledge"][:2000]

    print(f"    [extract] '{topic}' — {len(knowledge)} chars")
    log_event({
        "phase": "extract",
        "source_ts": source_ts,
        "topic": topic,
        "chars": len(knowledge),
    })
    return {"topic": topic, "knowledge": knowledge, "source_ts": source_ts}


# ── Phase 4: CRITIQUE — Knowledge Wiggum Loop ────────────────────────────────

DIMENSION_WEIGHTS = {
    "accuracy": 0.25,
    "completeness": 0.25,
    "clarity": 0.25,
    "coherence": 0.15,
    "applicability": 0.10,
}


async def knowledge_wiggum_loop(
    entry: dict,
    source_exchange: dict,
    semantic: dict,
    client: anthropic.AsyncAnthropic,
) -> tuple[dict, float]:
    """
    Critique a knowledge entry and refine if score < threshold.
    Returns (final_entry, score).
    """
    current = entry.copy()
    final_score = 0.0

    for attempt in range(MAX_REFINEMENTS + 1):
        # Build context for critique
        existing_topics = ", ".join(list(semantic.keys())[:20]) if semantic else "(none)"

        critique_response = await client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=CRITIQUE_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"Topic: {current['topic']}\n\n"
                           f"Knowledge entry:\n{current['knowledge']}\n\n"
                           f"Source query: {source_exchange.get('query', '')[:300]}\n"
                           f"Source response: {source_exchange.get('response_preview', '')[:500]}\n\n"
                           f"Existing semantic topics: {existing_topics}"
            }],
        )

        critique = extract_json(get_text(critique_response))
        if not critique:
            print(f"    [kwl] Attempt {attempt + 1}: no structured critique")
            break

        final_score = float(critique.get("score", 0.0))
        dimensions = critique.get("dimensions", {})
        issues = critique.get("issues", [])

        # Check dimension floor
        any_below_floor = any(
            float(v) < DIMENSION_FLOOR
            for v in dimensions.values()
        )

        print(f"    [kwl] Attempt {attempt + 1}: score={final_score:.2f}, "
              f"dims={json.dumps({k: round(float(v), 2) for k, v in dimensions.items()})}")

        log_event({
            "phase": "critique",
            "topic": current["topic"],
            "score": final_score,
            "dimensions": {k: round(float(v), 2) for k, v in dimensions.items()},
            "attempt": attempt + 1,
        })

        needs_refinement = final_score < REFINEMENT_THRESHOLD or any_below_floor

        if not needs_refinement or attempt == MAX_REFINEMENTS:
            break

        # Refine
        print(f"    [kwl] Score below threshold ({final_score:.2f} < {REFINEMENT_THRESHOLD}), refining...")
        guidance = critique.get("refinement_guidance", "Improve the entry based on the issues found.")

        refine_response = await client.messages.create(
            model=MODEL,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=EXTRACT_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"The previous knowledge entry scored {final_score:.2f} "
                           f"(below threshold {REFINEMENT_THRESHOLD}).\n\n"
                           f"Issues:\n" + "\n".join(f"- {i}" for i in issues) + "\n\n"
                           f"Refinement guidance: {guidance}\n\n"
                           f"Previous entry:\nTopic: {current['topic']}\n{current['knowledge']}\n\n"
                           f"Source exchange:\n"
                           f"Q: {source_exchange.get('query', '')}\n"
                           f"A: {source_exchange.get('response_preview', '')}\n\n"
                           f"Generate an improved knowledge entry."
            }],
        )

        refined = extract_json(get_text(refine_response))
        if refined and "knowledge" in refined:
            current["knowledge"] = refined["knowledge"][:2000]
            current["topic"] = refined.get("topic", current["topic"]).lower().strip()

    return current, final_score


# ── Phase 5: APPLY ───────────────────────────────────────────────────────────

def apply_entry(entry: dict, score: float, dry_run: bool) -> bool:
    """Write a knowledge entry to semantic memory via KiwiMemory."""
    topic = entry["topic"]
    knowledge = entry["knowledge"]

    if dry_run:
        print(f"\n    [DRY RUN] Would write semantic entry:")
        print(f"      Topic: {topic}")
        print(f"      Score: {score:.2f}")
        print(f"      Content ({len(knowledge)} chars): {knowledge[:200]}...")
        return True

    from memory.store import KiwiMemory
    mem = KiwiMemory()
    mem.add_semantic(topic, knowledge)
    print(f"    [applied] '{topic}' → semantic memory ({len(knowledge)} chars, score={score:.2f})")
    return True


# ── Main Pipeline ────────────────────────────────────────────────────────────

async def run_knowledge_pass(
    dry_run: bool,
    min_score: float,
    max_entries: int,
    client: anthropic.AsyncAnthropic,
):
    """One full KWQL pass: scan → triage → extract → critique → apply."""
    start = time.time()

    print(f"\n{'='*60}")
    print(f"  KWQL — Knowledge Quality Loop")
    print(f"{'='*60}")

    # Load data
    memory = load_memory()
    episodic = memory.get("episodic", [])
    semantic = memory.get("semantic", {})
    processed = get_processed_timestamps()

    print(f"  Memory: {len(episodic)} episodic, {len(semantic)} semantic, "
          f"{len(processed)} previously processed")

    if not episodic:
        print("  No episodic memory found — nothing to extract")
        return

    # 1. SCAN
    gaps = await scan_gaps(episodic, semantic, processed, min_score, client)
    if not gaps:
        print("  No knowledge gaps found — semantic memory is up to date")
        log_event({"phase": "pass_complete", "entries_applied": 0,
                    "entries_rejected": 0, "elapsed_s": round(time.time() - start, 1)})
        return

    # 2. TRIAGE
    triaged = triage_gaps(gaps, max_entries)

    # 3-4. EXTRACT → CRITIQUE for each gap
    applied_count = 0
    rejected_count = 0

    for gap in triaged:
        proposed_topic = gap.get("proposed_topic", "unknown")
        source_ts = gap.get("source_ts", "")
        print(f"\n  [{gap.get('priority', '?').upper()}] {proposed_topic}")

        # Find source exchange for later use
        source = next((e for e in episodic if e.get("ts") == source_ts), None)
        if not source:
            print(f"    [skip] Source exchange not found")
            continue

        # EXTRACT
        extracted = await extract_knowledge(gap, episodic, semantic, client)
        if not extracted:
            continue

        # CRITIQUE (KWL)
        final_entry, score = await knowledge_wiggum_loop(
            extracted, source, semantic, client
        )

        if score < REJECT_FLOOR:
            print(f"    [reject] Score too low ({score:.2f} < {REJECT_FLOOR})")
            log_event({
                "phase": "rejected",
                "topic": final_entry["topic"],
                "score": score,
                "source_ts": source_ts,
                "reason": f"score {score:.2f} < {REJECT_FLOOR}",
            })
            rejected_count += 1
            continue

        # APPLY
        applied = apply_entry(final_entry, score, dry_run)
        if applied:
            log_event({
                "phase": "applied" if not dry_run else "dry_run",
                "topic": final_entry["topic"],
                "score": score,
                "source_ts": source_ts,
                "chars": len(final_entry["knowledge"]),
            })
            applied_count += 1

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  KWQL Pass complete in {elapsed:.1f}s")
    print(f"  Applied: {applied_count} | Rejected: {rejected_count}")
    print(f"{'='*60}\n")

    log_event({
        "phase": "pass_complete",
        "entries_applied": applied_count,
        "entries_rejected": rejected_count,
        "elapsed_s": round(elapsed, 1),
        "dry_run": dry_run,
    })


# ── CLI ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="KWQL — Knowledge Quality Loop for Kiwi")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview extractions without writing to memory")
    parser.add_argument("--once", action="store_true",
                        help="Run one pass and exit (default: loop every 30min)")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_EPISODIC_SCORE,
                        help=f"Minimum episodic quality_score to process (default: {DEFAULT_MIN_EPISODIC_SCORE})")
    parser.add_argument("--max-entries", type=int, default=DEFAULT_MAX_ENTRIES,
                        help=f"Max extractions per pass (default: {DEFAULT_MAX_ENTRIES})")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = anthropic.AsyncAnthropic(api_key=api_key)

    print(f"KWQL — Knowledge Quality Loop for Kiwi")
    print(f"Dry run: {args.dry_run}")
    print(f"Min score: {args.min_score}")
    print(f"Max entries: {args.max_entries}")
    print(f"Mode: {'once' if args.once else f'loop every {LOOP_INTERVAL_SECONDS}s'}")

    if args.once:
        await run_knowledge_pass(args.dry_run, args.min_score, args.max_entries, client)
        return

    # Continuous loop
    while True:
        try:
            await run_knowledge_pass(args.dry_run, args.min_score, args.max_entries, client)
        except Exception as e:
            print(f"[!] Pass failed: {e}")
            log_event({"phase": "error", "error": str(e)})

        print(f"Next pass in {LOOP_INTERVAL_SECONDS // 60} minutes...")
        await asyncio.sleep(LOOP_INTERVAL_SECONDS)


if __name__ == "__main__":
    asyncio.run(main())
