"""
Complete CrewAI example: Vocabulary Tutor Crew using WordOrb.

Three specialized agents collaborate to deliver a personalized
vocabulary learning experience:

  1. Vocabulary Researcher  -- looks up words and finds connections
  2. Lesson Planner         -- creates personalized learning paths
  3. Quiz Master            -- generates and evaluates assessments

Run::

    export OPENAI_API_KEY="sk-..."
    export WORDORB_API_KEY="wo_..."
    python vocabulary_crew.py

Requires: crewai, httpx
"""

from __future__ import annotations

import os
import sys

from crewai import Agent, Crew, Process, Task

from wordorb_tools import (
    get_lesson,
    get_quiz,
    get_random_quiz,
    lookup_word,
    query_word_graph,
)

# ---------------------------------------------------------------------------
# Validate environment
# ---------------------------------------------------------------------------

if not os.environ.get("WORDORB_API_KEY"):
    print("Error: Set the WORDORB_API_KEY environment variable.")
    print("  Get a free key at https://wordorb.ai/pricing")
    sys.exit(1)

if not os.environ.get("OPENAI_API_KEY"):
    print("Error: Set the OPENAI_API_KEY environment variable.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

vocabulary_researcher = Agent(
    role="Vocabulary Researcher",
    goal=(
        "Thoroughly research words by looking up their definitions, etymology, "
        "pronunciation, and discovering how they connect to other words and "
        "lessons in the WordOrb curriculum."
    ),
    backstory=(
        "You are a linguist and etymologist with decades of experience tracing "
        "word origins across languages. You use the WordOrb dictionary (162K words, "
        "47 languages) and knowledge graph (30K connections) as your primary "
        "research tools. You provide precise, well-sourced word data."
    ),
    tools=[lookup_word, query_word_graph],
    verbose=True,
    allow_delegation=False,
)

lesson_planner = Agent(
    role="Lesson Planner",
    goal=(
        "Create personalized, engaging learning paths by selecting and sequencing "
        "WordOrb lessons based on the student's interests, level, and the "
        "thematic connections between words."
    ),
    backstory=(
        "You are a curriculum designer who specializes in vocabulary acquisition. "
        "You have access to the WordOrb lesson library (365 days x 4 tracks x 3 age "
        "groups = 4,380 lessons), each with a 5-phase structure: hook, story, wonder, "
        "action, wisdom. You combine research insights with pedagogical best practices "
        "to build learning paths that maximize retention."
    ),
    tools=[get_lesson, query_word_graph],
    verbose=True,
    allow_delegation=False,
)

quiz_master = Agent(
    role="Quiz Master",
    goal=(
        "Assess vocabulary knowledge by selecting appropriate quiz questions, "
        "presenting them clearly, evaluating answers, and providing constructive "
        "feedback that reinforces learning."
    ),
    backstory=(
        "You are an assessment specialist who believes testing is a powerful "
        "learning tool, not just a measurement. You draw from 21K quiz interactions "
        "across six question types (multiple choice, true/false, A/B comparison, "
        "open-ended, reflection) to create balanced assessments that are challenging "
        "but fair. You always explain the correct answer."
    ),
    tools=[get_quiz, get_random_quiz, lookup_word],
    verbose=True,
    allow_delegation=False,
)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def research_word(word: str) -> Task:
    """Create a task to research a word in depth."""
    return Task(
        description=(
            f"Research the word '{word}' thoroughly:\n"
            f"1. Look up '{word}' using the dictionary tool to get its definition, "
            f"   IPA pronunciation, etymology, and translations.\n"
            f"2. Query the knowledge graph to find all lessons where '{word}' appears "
            f"   and discover related words.\n"
            f"3. Compile a research brief that includes:\n"
            f"   - Full definition and pronunciation\n"
            f"   - Etymology and word history\n"
            f"   - List of related words from the knowledge graph\n"
            f"   - Which lessons feature this word (day numbers and tracks)\n"
        ),
        expected_output=(
            "A comprehensive research brief about the word, including its definition, "
            "pronunciation (IPA), etymology, related words, and a list of lessons "
            "where it appears in the WordOrb curriculum."
        ),
        agent=vocabulary_researcher,
    )


def create_lesson_plan(word: str, age_group: str = "adult") -> Task:
    """Create a task to build a lesson plan around a word."""
    return Task(
        description=(
            f"Create a personalized lesson plan for the word '{word}' "
            f"targeted at the '{age_group}' age group:\n"
            f"1. Use the research brief from the previous task to identify which "
            f"   lesson days feature '{word}'.\n"
            f"2. Retrieve at least one full lesson using the lesson tool.\n"
            f"3. Query the knowledge graph for related words and find additional "
            f"   connecting lessons.\n"
            f"4. Build a structured learning path with:\n"
            f"   - A hook to capture interest\n"
            f"   - The core lesson content\n"
            f"   - Connections to related words\n"
            f"   - Suggested follow-up lessons\n"
        ),
        expected_output=(
            "A structured lesson plan with: (1) an engaging hook, (2) core vocabulary "
            "content from the WordOrb lesson, (3) connections to related words, and "
            "(4) a list of recommended follow-up lesson days."
        ),
        agent=lesson_planner,
    )


def assess_knowledge(word: str, track: str = "learn") -> Task:
    """Create a task to build an assessment for a word."""
    return Task(
        description=(
            f"Create a vocabulary assessment for the word '{word}':\n"
            f"1. Get quiz questions aligned to the lesson day from the previous plan.\n"
            f"2. Also get 3 random questions from the '{track}' track for variety.\n"
            f"3. If needed, look up the word to verify answer accuracy.\n"
            f"4. Present a balanced 5-question assessment with:\n"
            f"   - At least one multiple-choice question\n"
            f"   - At least one open-ended question\n"
            f"   - Clear answer keys with explanations\n"
            f"5. Provide a scoring rubric and learning recommendations.\n"
        ),
        expected_output=(
            "A 5-question assessment with a mix of question types, each with "
            "an answer key and explanation, plus a scoring rubric and recommendations "
            "for further study based on performance."
        ),
        agent=quiz_master,
    )


# ---------------------------------------------------------------------------
# Crew
# ---------------------------------------------------------------------------


class VocabularyTutorCrew:
    """A CrewAI crew that researches a word, plans a lesson, and builds a quiz.

    Usage::

        crew = VocabularyTutorCrew(word="resilience", age_group="adult")
        result = crew.run()
        print(result)
    """

    def __init__(
        self,
        word: str,
        age_group: str = "adult",
        track: str = "learn",
    ) -> None:
        self.word = word
        self.age_group = age_group
        self.track = track

    def run(self) -> str:
        """Execute the crew and return the combined output."""
        task_research = research_word(self.word)
        task_lesson = create_lesson_plan(self.word, self.age_group)
        task_quiz = assess_knowledge(self.word, self.track)

        crew = Crew(
            agents=[vocabulary_researcher, lesson_planner, quiz_master],
            tasks=[task_research, task_lesson, task_quiz],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()
        return str(result)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the vocabulary tutor crew from the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="WordOrb Vocabulary Tutor Crew -- research, plan, and quiz"
    )
    parser.add_argument(
        "word",
        help="The word to study (e.g. 'resilience', 'courage')",
    )
    parser.add_argument(
        "--age",
        default="adult",
        choices=["kid", "adult", "elder"],
        help="Age group for lesson content (default: adult)",
    )
    parser.add_argument(
        "--track",
        default="learn",
        choices=["learn", "grow", "teach", "trivia"],
        help="Learning track (default: learn)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"  WordOrb Vocabulary Tutor Crew")
    print(f"  Word: {args.word}  |  Age: {args.age}  |  Track: {args.track}")
    print("=" * 60)
    print()

    crew = VocabularyTutorCrew(
        word=args.word,
        age_group=args.age,
        track=args.track,
    )
    result = crew.run()

    print("\n" + "=" * 60)
    print("  FINAL OUTPUT")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
