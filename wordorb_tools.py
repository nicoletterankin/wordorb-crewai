"""
CrewAI tool wrappers for the WordOrb.ai API.

Provides five tools using the ``@tool`` decorator:
  - lookup_word        -- dictionary lookup (162K words, 47 languages)
  - get_lesson         -- structured 5-phase lesson
  - get_quiz           -- quiz / assessment questions
  - get_random_quiz    -- random questions from the question bank
  - query_word_graph   -- knowledge graph connections

Usage::

    from wordorb_tools import lookup_word, get_lesson, get_quiz, query_word_graph
    from crewai import Agent

    researcher = Agent(
        role="Vocabulary Researcher",
        tools=[lookup_word, get_lesson, query_word_graph],
        ...
    )

Requires: crewai, httpx
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx
from crewai.tools import tool

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORDORB_API_KEY: str = os.environ.get("WORDORB_API_KEY", "")
WORDORB_BASE_URL: str = os.environ.get("WORDORB_BASE_URL", "https://api.wordorb.ai")
WORDORB_TIMEOUT: float = 30.0


def _headers() -> Dict[str, str]:
    """Build authorization headers."""
    if not WORDORB_API_KEY:
        raise RuntimeError(
            "WORDORB_API_KEY environment variable is not set. "
            "Get a free key at https://wordorb.ai/pricing?ref=crewai"
        )
    return {
        "Authorization": f"Bearer {WORDORB_API_KEY}",
        "User-Agent": "crewai-wordorb/1.0.0",
        "Accept": "application/json",
    }


def _get(path: str, params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Make a GET request to the WordOrb API and return parsed JSON."""
    url = f"{WORDORB_BASE_URL}{path}"
    with httpx.Client(timeout=WORDORB_TIMEOUT) as client:
        resp = client.get(url, headers=_headers(), params=params or {})
    if resp.status_code == 429:
        retry_after = resp.headers.get("Retry-After", "60")
        raise RuntimeError(
            f"WordOrb rate limit exceeded. Retry after {retry_after}s. "
            "Upgrade at https://wordorb.ai/pricing?ref=crewai"
        )
    if resp.status_code == 401:
        raise RuntimeError(
            "WordOrb authentication failed. Check WORDORB_API_KEY."
        )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool("lookup_word")
def lookup_word(word: str) -> str:
    """Look up any English word via the WordOrb dictionary API.

    Returns IPA pronunciation, definition, etymology, part of speech,
    example sentences, and translations in 47 languages.
    Backed by a 162K-word corpus.

    Args:
        word: The English word to look up (e.g. 'courage', 'serendipity').
    """
    data = _get(f"/api/word/{word}")

    parts = [f"Word: {data.get('word', word)}"]
    if data.get("ipa"):
        parts.append(f"IPA: {data['ipa']}")
    if data.get("pos"):
        parts.append(f"Part of speech: {data['pos']}")
    if data.get("definition") or data.get("def"):
        parts.append(f"Definition: {data.get('definition') or data.get('def')}")
    if data.get("etymology"):
        parts.append(f"Etymology: {data['etymology']}")
    if data.get("example"):
        parts.append(f"Example: {data['example']}")
    if data.get("synonyms"):
        syns = data["synonyms"]
        if isinstance(syns, list):
            syns = ", ".join(syns)
        parts.append(f"Synonyms: {syns}")
    if data.get("translations"):
        translations = data["translations"]
        if isinstance(translations, dict):
            tr_items = [f"  {k}: {v}" for k, v in list(translations.items())[:5]]
            parts.append("Translations:\n" + "\n".join(tr_items))

    return "\n".join(parts)


@tool("get_lesson")
def get_lesson(
    day: int = 1,
    track: str = "learn",
    age: str = "adult",
) -> str:
    """Get a structured 5-phase vocabulary lesson from the Orb Platform.

    Each lesson has five phases: hook, story, wonder, action, wisdom.
    365 days x 4 tracks x 3 age groups = 4,380 unique lessons.

    Args:
        day: Day of the year, 1-365. Each day has unique content.
        track: Learning track -- learn, grow, teach, or trivia.
        age: Age group -- kid, adult, or elder.
    """
    params = {"day": str(day), "track": track, "age": age}
    data = _get("/api/lesson", params)

    parts = [
        f"Lesson: Day {data.get('day', day)} | Track: {data.get('track', track)} "
        f"| Age: {data.get('age_group', age)}",
    ]
    if data.get("title"):
        parts.append(f"Title: {data['title']}")
    if data.get("theme"):
        parts.append(f"Theme: {data['theme']}")

    phases = data.get("phases", {})
    for phase_name in ("hook", "story", "wonder", "action", "wisdom"):
        content = phases.get(phase_name)
        if content:
            parts.append(f"\n--- {phase_name.upper()} ---\n{content}")

    return "\n".join(parts)


@tool("get_quiz")
def get_quiz(
    day: int = 1,
    track: str = "learn",
    question_type: str = "",
) -> str:
    """Get quiz questions aligned to a specific lesson day.

    Supports six question types: mc (multiple choice), tf (true/false),
    ab (A/B comparison), open (free response), reflection, and quiz.

    Args:
        day: Day of the year, 1-365.
        track: Learning track -- learn, grow, teach, or trivia.
        question_type: Optional filter: mc, tf, ab, open, reflection, quiz.
            Leave empty for all types.
    """
    params: Dict[str, str] = {"day": str(day), "track": track}
    if question_type:
        params["type"] = question_type
    data = _get("/api/quiz", params)

    interactions = data.get("interactions", [])
    if not interactions:
        return f"No quiz questions found for day {day}, track '{track}'."

    parts = [f"Quiz: {len(interactions)} question(s) | Day {day} | Track: {track}"]
    for i, q in enumerate(interactions, 1):
        parts.append(f"\nQ{i} [{q.get('type', '?')}]: {q.get('prompt', '')}")
        if q.get("choices"):
            choices = q["choices"]
            if isinstance(choices, list):
                for idx, c in enumerate(choices):
                    parts.append(f"  {chr(65 + idx)}. {c}")
        if q.get("answer"):
            parts.append(f"  Answer: {q['answer']}")
        if q.get("explanation"):
            parts.append(f"  Explanation: {q['explanation']}")

    return "\n".join(parts)


@tool("get_random_quiz")
def get_random_quiz(
    track: str = "learn",
    count: int = 5,
    question_type: str = "",
) -> str:
    """Get random quiz questions from the WordOrb question bank.

    Use this for mixed practice, warm-ups, or assessments that are not
    tied to a specific lesson day.

    Args:
        track: Learning track -- learn, grow, teach, or trivia.
        count: Number of random questions (1-20).
        question_type: Optional filter: mc, tf, ab, open, reflection.
            Leave empty for all types.
    """
    params: Dict[str, str] = {"track": track, "count": str(min(count, 20))}
    if question_type:
        params["type"] = question_type
    data = _get("/api/quiz/random", params)

    interactions = data.get("interactions", [])
    if not interactions:
        return f"No random quiz questions found for track '{track}'."

    parts = [f"Random Quiz: {len(interactions)} question(s) | Track: {track}"]
    for i, q in enumerate(interactions, 1):
        parts.append(
            f"\nQ{i} [Day {q.get('day', '?')}, {q.get('type', '?')}]: "
            f"{q.get('prompt', '')}"
        )
        if q.get("choices"):
            choices = q["choices"]
            if isinstance(choices, list):
                for idx, c in enumerate(choices):
                    parts.append(f"  {chr(65 + idx)}. {c}")
        if q.get("answer"):
            parts.append(f"  Answer: {q['answer']}")

    return "\n".join(parts)


@tool("query_word_graph")
def query_word_graph(word: str) -> str:
    """Query the WordOrb knowledge graph for word-to-lesson connections.

    Returns all lessons where a word appears, related words, and importance
    scores. Useful for building learning paths and finding thematic
    connections across the 365-day curriculum.

    Args:
        word: Word to look up in the knowledge graph.
    """
    data = _get("/api/graph/word", {"word": word})

    parts = [
        f"Knowledge Graph: '{data.get('word', word)}' appears in "
        f"{data.get('appears_in', 0)} lesson(s)"
    ]

    lessons = data.get("lessons", [])
    for lesson in lessons[:10]:
        parts.append(
            f"  Day {lesson.get('day', '?')} ({lesson.get('track', '?')}) "
            f"- {lesson.get('phase', '?')} phase, "
            f"importance={lesson.get('importance', '?')}"
        )
        if lesson.get("context"):
            parts.append(f"    Context: {lesson['context']}")

    related = data.get("related_words", [])
    if related:
        parts.append(f"\nRelated words: {', '.join(related)}")

    return "\n".join(parts)
