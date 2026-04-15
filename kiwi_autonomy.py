"""
Kiwi Autonomy — Self-Directed Research Daemon
================================================
Autonomous 5-phase loop that makes Kiwi rationalize on its own:
decide what to research, execute it, extract knowledge, evaluate quality,
and feed forward to the next cycle.

Pipeline:
  1. RATIONALIZE — Self-reflect on knowledge state (gaps, staleness, contradictions)
  2. AGENDA      — Prioritize and formulate research queries
  3. RESEARCH    — Execute headless Kiwi pipeline + PubMed
  4. EXTRACT     — Delegate to KWQL for knowledge crystallization
  5. REFLECT     — Meta-evaluate, persist open questions for next pass

Usage:
  python3 kiwi_autonomy.py --once --dry-run          # Preview what Kiwi would research
  python3 kiwi_autonomy.py --once                    # One full pass
  python3 kiwi_autonomy.py                           # Daemon loop every 30min
  python3 kiwi_autonomy.py --max-queries 5           # More queries per pass
  python3 kiwi_autonomy.py --staleness-days 7        # Aggressive staleness threshold

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
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import anthropic

# ── Paths ──────────────────────────────────────────────────────────────────

KIWI_ROOT = Path("/home/nelly/kiwi")
sys.path.insert(0, str(KIWI_ROOT))

# Also need rwql on path for kwql import
RWQL_ROOT = Path(__file__).parent
sys.path.insert(0, str(RWQL_ROOT))

KIWI_DIR = Path.home() / ".kiwi"
AGENDA_FILE = KIWI_DIR / "agenda.json"
LOG_FILE = RWQL_ROOT / "kiwi_autonomy_log.jsonl"

# ── Config ─────────────────────────────────────────────────────────────────

MODEL = "claude-opus-4-6"
REFINEMENT_THRESHOLD = 0.72
DEFAULT_MAX_QUERIES = 3
DEFAULT_STALENESS_DAYS = 14
LOOP_INTERVAL_SECONDS = 1800  # 30 min

# Agenda priority weights
W_USER_RELEVANCE = 0.35
W_GAP_SEVERITY = 0.30
W_STALENESS_URGENCY = 0.20
W_CROSS_TOPIC_VALUE = 0.15

AUTONOMY_THREAD = "kiwi-autonomy"

# ── Data Structures ────────────────────────────────────────────────────────


@dataclass
class AgendaItem:
    category: str           # gap | stale | contradiction | depth | user_relevant | cross_topic
    priority: float         # 0.0–1.0
    query: str              # Research query to execute
    rationale: str          # Why this was identified
    source_topics: list[str] = field(default_factory=list)
    existing_knowledge: str = ""  # For staleness refreshes
    status: str = "pending"       # pending | researched | failed
    score: float = 0.0            # RWL score after research


# ── Prompts ────────────────────────────────────────────────────────────────

RATIONALIZE_SYSTEM = """\
You are Kiwi's self-reflection engine. You analyze the current state of Kiwi's \
knowledge base and identify what needs to be researched next.

You will receive:
- All semantic memory entries (topic → content, with staleness annotations)
- Recent episodic history (last research exchanges)
- User profile (sport, goals, conditions)
- Open questions from previous autonomy passes

Your job: identify knowledge gaps across these categories:
1. **gap** — Important topics not yet covered in semantic memory
2. **stale** — Entries older than the staleness threshold needing refresh
3. **contradiction** — Conflicting claims across semantic entries
4. **depth** — Surface-level entries lacking mechanisms, dosages, or citations
5. **user_relevant** — Topics relevant to user's sport/goals/conditions not yet covered
6. **cross_topic** — Topics that should reference each other but don't

Respond ONLY with JSON:
{
  "gaps": [
    {
      "category": "gap|stale|contradiction|depth|user_relevant|cross_topic",
      "severity": 0.0-1.0,
      "topic": "short topic description",
      "rationale": "why this gap matters",
      "source_topics": ["existing topics this relates to"],
      "existing_knowledge": "current content if stale/depth (empty string otherwise)"
    }
  ],
  "summary": "1-2 sentence overview of knowledge state"
}"""

AGENDA_SYSTEM = """\
You are Kiwi's research agenda planner. Given a list of knowledge gaps, \
formulate specific, high-quality research queries that Kiwi's pipeline can execute.

Each query should be:
- Specific enough to produce actionable, evidence-based results
- Framed as a research question (not a command)
- Include context about what's already known (for staleness refreshes)

For each gap, produce exactly one research query. Rank by priority using this formula:
- user_relevance (weight 0.35)
- gap_severity (weight 0.30)
- staleness_urgency (weight 0.20)
- cross_topic_value (weight 0.15)

Respond ONLY with JSON:
{
  "agenda": [
    {
      "category": "gap|stale|contradiction|depth|user_relevant|cross_topic",
      "priority": 0.0-1.0,
      "query": "The specific research query to execute",
      "rationale": "Why this was prioritized",
      "source_topics": ["related existing topics"],
      "existing_knowledge": "what's already known (for context injection)"
    }
  ]
}"""

REFLECT_SYSTEM = """\
You are Kiwi's meta-evaluator. After a research pass, you review what happened \
and plan forward.

You will receive:
- The original rationalization (gaps identified)
- The agenda (queries planned)
- The outcomes (which queries succeeded, their scores)

Produce a reflection:
1. Which gaps were filled vs. remaining
2. New questions that emerged from the research
3. Quality assessment of this pass
4. Priorities for the next pass

Respond ONLY with JSON:
{
  "gaps_filled": ["list of gap topics successfully researched"],
  "gaps_remaining": ["list of gaps not yet addressed"],
  "new_questions": ["new research questions that emerged"],
  "quality_assessment": "1-2 sentence quality summary",
  "next_priorities": ["top 3 priorities for next pass"]
}"""


# ── Utilities ──────────────────────────────────────────────────────────────

def log_event(event: dict):
    entry = {"ts": datetime.now(timezone.utc).isoformat(), **event}
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


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


def compute_staleness(updated_iso: str, staleness_days: int) -> tuple[bool, int]:
    """Check if an entry is stale. Returns (is_stale, days_old)."""
    try:
        updated = datetime.fromisoformat(updated_iso.replace("Z", "+00:00"))
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        age = now - updated
        return age > timedelta(days=staleness_days), age.days
    except (ValueError, TypeError):
        return True, 999  # If we can't parse, consider stale


def load_agenda() -> dict:
    """Load persistent agenda from disk."""
    if AGENDA_FILE.exists():
        try:
            return json.loads(AGENDA_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "items": [],
        "last_reflection": None,
        "open_questions": [],
        "gaps_filled": [],
        "gaps_remaining": [],
    }


def save_agenda(agenda: dict):
    """Persist agenda to disk."""
    KIWI_DIR.mkdir(parents=True, exist_ok=True)
    AGENDA_FILE.write_text(json.dumps(agenda, indent=2, default=str))


# ── Phase 1: RATIONALIZE ──────────────────────────────────────────────────

async def rationalize(
    client: anthropic.AsyncAnthropic,
    staleness_days: int,
) -> tuple[list[dict], str]:
    """
    Self-reflect on Kiwi's knowledge state.
    Returns (gaps, summary).
    """
    from memory.store import KiwiMemory
    from memory.profile import UserProfile

    mem = KiwiMemory()
    profile = UserProfile()

    semantic = mem.data.get("semantic", {})
    episodic = mem.data.get("episodic", [])

    # Annotate semantic entries with staleness (uses KiwiMemory's built-in method)
    semantic_entries = mem.get_semantic_with_staleness()
    stale_count = sum(1 for e in semantic_entries if e["is_stale"])
    annotated_semantic = []
    for e in semantic_entries:
        age_label = f" [STALE — {e['days_old']}d old]" if e["is_stale"] else f" [{e['days_old']}d old]"
        annotated_semantic.append(f"• {e['topic']}{age_label}: {e['content'][:300]}")

    semantic_block = "\n".join(annotated_semantic) if annotated_semantic else "(no semantic memory)"

    # Archived episodic (for deeper historical context)
    archive_stats = mem.archive_stats()
    archive_block = ""
    if archive_stats.get("archived_entries", 0) > 0:
        archive_block = f"\n\n=== ARCHIVED RESEARCH ({archive_stats['archived_entries']} entries) ===\n(searchable via archive)"

    # Recent episodic
    recent = episodic[-10:] if episodic else []
    episodic_block = "\n".join(
        f"[{e.get('ts', '')[:10]}] (score: {e.get('quality_score', '?')}) {e.get('query', '')[:200]}"
        for e in recent
    ) if recent else "(no episodic history)"

    # Profile
    profile_block = profile.to_summary() if profile.is_complete() else "(no profile configured)"

    # Open questions from previous passes
    agenda = load_agenda()
    open_questions = agenda.get("open_questions", [])
    oq_block = "\n".join(f"- {q}" for q in open_questions) if open_questions else "(none)"
    gaps_remaining = agenda.get("gaps_remaining", [])
    gr_block = "\n".join(f"- {g}" for g in gaps_remaining) if gaps_remaining else "(none)"

    user_msg = (
        f"=== SEMANTIC MEMORY ({len(semantic)} topics, {stale_count} stale) ===\n"
        f"{semantic_block}\n\n"
        f"=== RECENT EPISODIC HISTORY ({len(recent)} exchanges) ===\n"
        f"{episodic_block}\n"
        f"{archive_block}\n\n"
        f"=== USER PROFILE ===\n"
        f"{profile_block}\n\n"
        f"=== OPEN QUESTIONS FROM PREVIOUS PASS ===\n"
        f"{oq_block}\n\n"
        f"=== UNRESOLVED GAPS FROM PREVIOUS PASS ===\n"
        f"{gr_block}\n\n"
        f"Staleness threshold: {staleness_days} days\n"
        f"Identify all knowledge gaps."
    )

    response = await client.messages.create(
        model=MODEL,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=RATIONALIZE_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    data = extract_json(get_text(response))
    if not data:
        print("  [rationalize] No structured response from Claude")
        return [], "Failed to parse rationalization"

    gaps = data.get("gaps", [])
    summary = data.get("summary", "")

    print(f"  [rationalize] {len(gaps)} gaps found "
          f"(semantic: {len(semantic)}, stale: {stale_count})")
    print(f"  Summary: {summary}")

    log_event({
        "phase": "rationalize",
        "gaps_found": len(gaps),
        "semantic_count": len(semantic),
        "stale_count": stale_count,
        "summary": summary,
    })

    return gaps, summary


# ── Phase 2: AGENDA ───────────────────────────────────────────────────────

async def build_agenda(
    gaps: list[dict],
    max_queries: int,
    client: anthropic.AsyncAnthropic,
) -> list[AgendaItem]:
    """
    Prioritize gaps and formulate research queries.
    Returns a ranked list of AgendaItems, capped at max_queries.
    """
    if not gaps:
        return []

    from memory.profile import UserProfile
    profile = UserProfile()
    profile_block = profile.to_summary() if profile.is_complete() else "(no profile)"

    gaps_block = json.dumps(gaps, indent=2, default=str)

    user_msg = (
        f"=== USER PROFILE ===\n{profile_block}\n\n"
        f"=== KNOWLEDGE GAPS ({len(gaps)}) ===\n{gaps_block}\n\n"
        f"Max queries to plan: {max_queries}\n\n"
        f"Priority weights: user_relevance={W_USER_RELEVANCE}, "
        f"gap_severity={W_GAP_SEVERITY}, staleness_urgency={W_STALENESS_URGENCY}, "
        f"cross_topic_value={W_CROSS_TOPIC_VALUE}\n\n"
        f"Formulate research queries for the top {max_queries} gaps."
    )

    response = await client.messages.create(
        model=MODEL,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=AGENDA_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    data = extract_json(get_text(response))
    if not data:
        print("  [agenda] No structured response from Claude")
        return []

    raw_items = data.get("agenda", [])

    # Convert to AgendaItems, sorted by priority
    items = []
    for raw in raw_items:
        items.append(AgendaItem(
            category=raw.get("category", "gap"),
            priority=float(raw.get("priority", 0.5)),
            query=raw.get("query", ""),
            rationale=raw.get("rationale", ""),
            source_topics=raw.get("source_topics", []),
            existing_knowledge=raw.get("existing_knowledge", ""),
        ))

    items.sort(key=lambda x: x.priority, reverse=True)
    items = items[:max_queries]

    categories = [item.category for item in items]
    print(f"  [agenda] {len(items)} queries planned: {categories}")

    log_event({
        "phase": "agenda",
        "items_planned": len(items),
        "categories": categories,
    })

    return items


# ── Phase 3: RESEARCH ─────────────────────────────────────────────────────

async def research_item(
    item: AgendaItem,
    client: anthropic.AsyncAnthropic,
    dry_run: bool,
) -> AgendaItem:
    """
    Execute the Kiwi research pipeline for a single agenda item.
    Mutates item.status and item.score in-place.
    """
    query = item.query

    if dry_run:
        print(f"    [DRY RUN] Would research: {query[:100]}")
        print(f"      Category: {item.category}, Priority: {item.priority:.2f}")
        print(f"      Rationale: {item.rationale[:150]}")
        item.status = "pending"
        return item

    try:
        from agents.orchestrator import KiwiOrchestrator
        from tools.pubmed import PubMedClient
        from tools.openalex import OpenAlexClient
        from memory.store import KiwiMemory
        from memory.profile import UserProfile

        mem = KiwiMemory()
        profile = UserProfile()
        pubmed = PubMedClient()
        openalex = OpenAlexClient()

        # PubMed pre-fetch
        print(f"    [pubmed] Searching: {query[:80]}...")
        articles = pubmed.search_and_fetch(query, max_results=5, years_back=10)
        pubmed_context = pubmed.build_context_block(articles)
        pubmed_count = len(articles)
        seen_dois = {a.doi.lower() for a in articles if a.doi}
        print(f"    [pubmed] {pubmed_count} articles fetched")

        # OpenAlex supplementary search (sports nutrition journals)
        print(f"    [openalex] Searching sports nutrition journals...")
        oa_works = openalex.search_sports_nutrition(query, max_results=4, years_back=10)
        oa_unique = [w for w in oa_works if not w.doi or w.doi.lower() not in seen_dois][:3]
        if oa_unique:
            pubmed_context += "\n\n" + openalex.build_context_block(oa_unique)
            print(f"    [openalex] {len(oa_unique)} additional articles")

        # Inject existing knowledge for staleness refreshes
        if item.existing_knowledge:
            pubmed_context = (
                f"=== EXISTING KNOWLEDGE (to be updated) ===\n"
                f"{item.existing_knowledge[:1500]}\n\n"
                f"{pubmed_context}"
            )

        # Ensure autonomy thread exists
        if AUTONOMY_THREAD not in mem.data.get("threads", {}):
            mem.create_thread(AUTONOMY_THREAD, "Autonomous research by Kiwi Autonomy daemon")

        # Fresh messages — no cross-contamination between queries
        messages = []

        # Memory & profile context
        memory_summary = mem.get_history_summary()
        profile_summary = profile.to_summary()

        # Execute headless pipeline (no Rich, no streaming callbacks)
        print(f"    [pipeline] Running full Kiwi pipeline...")
        orchestrator = KiwiOrchestrator(client)
        result = await orchestrator.run_full_pipeline(
            query=query,
            messages=messages,
            memory_summary=memory_summary,
            profile_summary=profile_summary,
            pubmed_context=pubmed_context,
            on_status=lambda s: print(f"    [pipeline] {s}"),
        )

        score = result.get("score", 0.0)
        final_response = result.get("final_response", "")

        # Save to episodic memory
        mem.add_exchange(
            query=query,
            response=final_response,
            score=score,
            thread=AUTONOMY_THREAD,
        )

        item.status = "researched"
        item.score = score
        print(f"    [research] Done — score: {score:.2f}, "
              f"refined: {result.get('refined', False)}")

        log_event({
            "phase": "research",
            "query": query[:200],
            "score": score,
            "pubmed_articles": pubmed_count,
            "refined": result.get("refined", False),
        })

    except Exception as e:
        item.status = "failed"
        print(f"    [research] FAILED: {e}")
        log_event({
            "phase": "research",
            "query": query[:200],
            "error": str(e),
        })

    return item


async def research_phase(
    agenda: list[AgendaItem],
    client: anthropic.AsyncAnthropic,
    dry_run: bool,
) -> list[AgendaItem]:
    """Execute research for all agenda items (sequential, isolated)."""
    for i, item in enumerate(agenda):
        print(f"\n  [{i+1}/{len(agenda)}] {item.category.upper()}: {item.query[:80]}")
        await research_item(item, client, dry_run)
    return agenda


# ── Phase 4: EXTRACT ──────────────────────────────────────────────────────

async def extract_phase(
    dry_run: bool,
    client: anthropic.AsyncAnthropic,
):
    """Delegate to KWQL for knowledge crystallization."""
    if dry_run:
        print("\n  [extract] DRY RUN — skipping KWQL")
        return

    print("\n  [extract] Running KWQL knowledge pass...")
    try:
        from kwql import run_knowledge_pass
        await run_knowledge_pass(
            dry_run=False,
            min_score=0.7,
            max_entries=5,
            client=client,
        )
    except Exception as e:
        print(f"  [extract] KWQL failed: {e}")
        log_event({"phase": "extract", "error": str(e)})


# ── Phase 5: REFLECT ──────────────────────────────────────────────────────

async def reflect_phase(
    gaps: list[dict],
    agenda: list[AgendaItem],
    client: anthropic.AsyncAnthropic,
    dry_run: bool,
) -> dict:
    """
    Meta-evaluate the pass and persist reflection for next cycle.
    Returns the reflection data.
    """
    if dry_run:
        print("\n  [reflect] DRY RUN — skipping reflection")
        return {}

    # Build outcomes summary
    outcomes = []
    for item in agenda:
        outcomes.append({
            "query": item.query[:200],
            "category": item.category,
            "status": item.status,
            "score": item.score,
        })

    user_msg = (
        f"=== ORIGINAL RATIONALIZATION ===\n"
        f"{json.dumps(gaps[:10], indent=2, default=str)}\n\n"
        f"=== AGENDA ===\n"
        f"{json.dumps([asdict(a) for a in agenda], indent=2, default=str)}\n\n"
        f"=== OUTCOMES ===\n"
        f"{json.dumps(outcomes, indent=2, default=str)}\n\n"
        f"Reflect on this pass."
    )

    response = await client.messages.create(
        model=MODEL,
        max_tokens=2048,
        thinking={"type": "adaptive"},
        system=REFLECT_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    data = extract_json(get_text(response))
    if not data:
        print("  [reflect] No structured response from Claude")
        return {}

    gaps_filled = data.get("gaps_filled", [])
    gaps_remaining = data.get("gaps_remaining", [])
    new_questions = data.get("new_questions", [])
    quality = data.get("quality_assessment", "")

    print(f"\n  [reflect] Gaps filled: {len(gaps_filled)}, "
          f"Remaining: {len(gaps_remaining)}, "
          f"New questions: {len(new_questions)}")
    print(f"  Quality: {quality}")

    # Persist to agenda.json for next pass's feed-forward
    persistent = load_agenda()
    persistent["last_reflection"] = datetime.now(timezone.utc).isoformat()
    persistent["open_questions"] = new_questions
    persistent["gaps_filled"] = gaps_filled
    persistent["gaps_remaining"] = gaps_remaining
    persistent["items"] = [asdict(a) for a in agenda]
    save_agenda(persistent)

    log_event({
        "phase": "reflect",
        "gaps_filled": len(gaps_filled),
        "gaps_remaining": len(gaps_remaining),
        "new_questions": len(new_questions),
    })

    return data


# ── Main Pipeline ─────────────────────────────────────────────────────────

async def run_autonomy_pass(
    dry_run: bool,
    max_queries: int,
    staleness_days: int,
    client: anthropic.AsyncAnthropic,
):
    """One full autonomy pass: rationalize → agenda → research → extract → reflect."""
    start = time.time()

    print(f"\n{'='*60}")
    print(f"  Kiwi Autonomy — Self-Directed Research Pass")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Phase 1: RATIONALIZE
    print(f"\n  Phase 1: RATIONALIZE")
    gaps, summary = await rationalize(client, staleness_days)

    if not gaps:
        print("  No knowledge gaps found — Kiwi's knowledge is up to date")
        log_event({
            "phase": "pass_complete",
            "researched": 0,
            "failed": 0,
            "elapsed_s": round(time.time() - start, 1),
        })
        return

    # Phase 2: AGENDA
    print(f"\n  Phase 2: AGENDA")
    agenda = await build_agenda(gaps, max_queries, client)

    if not agenda:
        print("  No agenda items generated")
        return

    # Phase 3: RESEARCH
    print(f"\n  Phase 3: RESEARCH")
    agenda = await research_phase(agenda, client, dry_run)

    # Phase 4: EXTRACT (delegate to KWQL)
    print(f"\n  Phase 4: EXTRACT")
    await extract_phase(dry_run, client)

    # Phase 5: REFLECT
    print(f"\n  Phase 5: REFLECT")
    await reflect_phase(gaps, agenda, client, dry_run)

    # Summary
    elapsed = time.time() - start
    researched = sum(1 for a in agenda if a.status == "researched")
    failed = sum(1 for a in agenda if a.status == "failed")

    print(f"\n{'='*60}")
    print(f"  Autonomy pass complete in {elapsed:.1f}s")
    print(f"  Researched: {researched} | Failed: {failed} | "
          f"Dry run: {dry_run}")
    print(f"{'='*60}\n")

    log_event({
        "phase": "pass_complete",
        "researched": researched,
        "failed": failed,
        "elapsed_s": round(elapsed, 1),
        "dry_run": dry_run,
    })


# ── CLI ────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="Kiwi Autonomy — Self-Directed Research Daemon"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview rationalization + agenda without executing research",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run one pass and exit (default: loop every 30min)",
    )
    parser.add_argument(
        "--max-queries", type=int, default=DEFAULT_MAX_QUERIES,
        help=f"Max research queries per pass (default: {DEFAULT_MAX_QUERIES})",
    )
    parser.add_argument(
        "--staleness-days", type=int, default=DEFAULT_STALENESS_DAYS,
        help=f"Days before semantic entry is considered stale (default: {DEFAULT_STALENESS_DAYS})",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = anthropic.AsyncAnthropic(api_key=api_key)

    print(f"Kiwi Autonomy — Self-Directed Research Daemon")
    print(f"Dry run: {args.dry_run}")
    print(f"Max queries: {args.max_queries}")
    print(f"Staleness threshold: {args.staleness_days} days")
    print(f"Mode: {'once' if args.once else f'loop every {LOOP_INTERVAL_SECONDS // 60}min'}")

    if args.once:
        await run_autonomy_pass(
            args.dry_run, args.max_queries, args.staleness_days, client,
        )
        return

    # Continuous loop
    while True:
        try:
            await run_autonomy_pass(
                args.dry_run, args.max_queries, args.staleness_days, client,
            )
        except Exception as e:
            print(f"[!] Pass failed: {e}")
            log_event({"phase": "error", "error": str(e)})

        print(f"Next pass in {LOOP_INTERVAL_SECONDS // 60} minutes...")
        await asyncio.sleep(LOOP_INTERVAL_SECONDS)


if __name__ == "__main__":
    asyncio.run(main())
