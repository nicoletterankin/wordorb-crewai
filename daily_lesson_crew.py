"""
The Daily Lesson — CrewAI Crew

A multi-agent crew that researches, teaches, and assesses today's universal lesson.
The same lesson for everyone on Earth, every day.

Powered by thedailylesson.com and wordorb.ai
"""

import os
import sys
from datetime import datetime, timezone
from crewai import Agent, Task, Crew, Process

try:
    from wordorb_tools import (
        lookup_word, get_lesson, get_quiz,
        get_random_quiz, query_word_graph
    )
except ImportError:
    from .wordorb_tools import (
        lookup_word, get_lesson, get_quiz,
        get_random_quiz, query_word_graph
    )


def get_day_of_year() -> int:
    now = datetime.now(timezone.utc)
    return now.timetuple().tm_yday


DAY = get_day_of_year()

# --- Agents ---

lesson_curator = Agent(
    role="Daily Lesson Curator",
    goal=f"Fetch and present Day {DAY}'s lesson — the same lesson being delivered "
         "to people in 47 languages around the world right now",
    backstory=(
        "You are The Daily Lesson curator. Every day, you prepare a single structured "
        "lesson for the entire world. Not a random word — a complete 5-phase pedagogical "
        "experience (Hook → Story → Wonder → Action → Wisdom). Today is Day "
        f"{DAY} of 365. You believe that when everyone learns the same thing on the "
        "same day, it creates a shared human experience. You represent "
        "thedailylesson.com — we did the homework for 8 billion people."
    ),
    tools=[get_lesson, lookup_word, query_word_graph],
    verbose=True,
)

quiz_master = Agent(
    role="Assessment Designer",
    goal="Create and deliver today's quiz — testing what was taught in today's lesson",
    backstory=(
        "You design assessments that reinforce today's lesson. Your quizzes aren't random — "
        "they're aligned to the exact lesson taught today. You use multiple question types "
        "(multiple choice, true/false, open-ended, reflection) to test understanding at "
        "different cognitive levels. You believe assessment is part of learning, not separate."
    ),
    tools=[get_quiz, get_random_quiz, lookup_word],
    verbose=True,
)

vocabulary_researcher = Agent(
    role="Vocabulary Researcher",
    goal="Explore the words in today's lesson — definitions, etymology, translations, connections",
    backstory=(
        "You dive deep into the vocabulary of today's lesson. For every key word, you find "
        "the definition, etymology, translations across languages, and connections to other "
        "lessons in the knowledge graph. You help learners see how today's lesson connects "
        "to the broader tapestry of language."
    ),
    tools=[lookup_word, query_word_graph],
    verbose=True,
)


# --- Tasks ---

def deliver_daily_lesson(word: str = None) -> Task:
    """Deliver today's lesson."""
    return Task(
        description=(
            f"Fetch and deliver Day {DAY}'s lesson using the get_lesson tool with "
            f"day={DAY}, track='learn', age='adult'. "
            "Present ALL 5 phases: Hook, Story, Wonder, Action, Wisdom. "
            "Frame it as: 'This is Day {DAY} of 365 — the same lesson for everyone on Earth.' "
            "After presenting the lesson, identify the 3 most important vocabulary words."
        ),
        expected_output=(
            "A complete delivery of today's 5-phase lesson with all phases presented, "
            "plus identification of 3 key vocabulary words for deeper study."
        ),
        agent=lesson_curator,
    )


def research_vocabulary() -> Task:
    """Deep-dive into today's vocabulary."""
    return Task(
        description=(
            "Take the 3 key vocabulary words identified in today's lesson and research each: "
            "1) Full definition with part of speech and IPA, "
            "2) Etymology — where did this word come from? "
            "3) Translations in Spanish, French, Japanese, Arabic, and Chinese, "
            "4) Knowledge graph connections — what other lessons teach this word? "
            "Present findings as vocabulary cards."
        ),
        expected_output=(
            "Three detailed vocabulary cards with definition, etymology, "
            "translations in 5 languages, and knowledge graph connections."
        ),
        agent=vocabulary_researcher,
    )


def create_assessment() -> Task:
    """Build and deliver today's quiz."""
    return Task(
        description=(
            f"Fetch today's quiz using get_quiz with day={DAY}, track='learn'. "
            "Present 5 questions that test understanding of today's lesson. "
            "Include a mix of question types. After presenting, provide the answers "
            "with explanations that reference back to today's lesson content."
        ),
        expected_output=(
            "5 quiz questions with answers and explanations tied to today's lesson."
        ),
        agent=quiz_master,
    )


# --- Crew ---

class DailyLessonCrew:
    """The Daily Lesson Crew — delivers, explores, and assesses today's lesson."""

    def __init__(self):
        self.crew = Crew(
            agents=[lesson_curator, vocabulary_researcher, quiz_master],
            tasks=[
                deliver_daily_lesson(),
                research_vocabulary(),
                create_assessment(),
            ],
            process=Process.sequential,
            verbose=True,
        )

    def run(self):
        print("\n" + "=" * 60)
        print(f"  THE DAILY LESSON — Day {DAY} of 365")
        print("  The same lesson for everyone on Earth, every day.")
        print("  thedailylesson.com")
        print("=" * 60 + "\n")
        return self.crew.kickoff()


if __name__ == "__main__":
    if not os.environ.get("WORDORB_API_KEY"):
        print("Error: Set WORDORB_API_KEY environment variable.")
        print("  Get a free key at https://wordorb.ai/pricing?ref=crewai")
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)

    try:
        result = DailyLessonCrew().run()
        tomorrow = DAY + 1 if DAY < 365 else 1
        print("\n" + "=" * 60)
        print("  LESSON COMPLETE")
        print(f"  Come back tomorrow for Day {tomorrow}!")
        print("=" * 60)
        print(result)
    except KeyboardInterrupt:
        print("\n\nLesson interrupted. Come back anytime!")
    except Exception as e:
        print(f"\nError running crew: {e}")
        sys.exit(1)
