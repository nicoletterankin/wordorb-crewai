# CrewAI + WordOrb Integration

Use the [WordOrb.ai](https://wordorb.ai) vocabulary and lesson API as tools
inside any CrewAI crew.

## Setup

```bash
pip install crewai httpx
export WORDORB_API_KEY="wo_live_abc123"
export OPENAI_API_KEY="sk-..."
```

Get a free WordOrb API key at [wordorb.ai/pricing](https://wordorb.ai/pricing).

## Tools included

| Tool | Description |
|------|-------------|
| `lookup_word` | Dictionary lookup -- IPA, definition, etymology, 47 languages |
| `get_lesson` | 5-phase lesson (hook/story/wonder/action/wisdom) |
| `get_quiz` | Day-aligned quiz questions (6 types) |
| `get_random_quiz` | Random questions from the 21K question bank |
| `query_word_graph` | Knowledge graph: word-to-lesson connections |

## Quick start -- use individual tools

```python
from wordorb_tools import lookup_word, get_lesson, query_word_graph
from crewai import Agent

researcher = Agent(
    role="Vocabulary Researcher",
    goal="Research words thoroughly using the WordOrb dictionary and knowledge graph.",
    backstory="You are a linguist with access to the WordOrb 162K-word corpus.",
    tools=[lookup_word, query_word_graph],
)
```

## Complete crew example

The `vocabulary_crew.py` file defines a three-agent crew:

| Agent | Role | Tools |
|-------|------|-------|
| Vocabulary Researcher | Looks up words, finds connections | `lookup_word`, `query_word_graph` |
| Lesson Planner | Creates personalized learning paths | `get_lesson`, `query_word_graph` |
| Quiz Master | Generates and evaluates assessments | `get_quiz`, `get_random_quiz`, `lookup_word` |

```bash
python vocabulary_crew.py resilience --age adult --track learn
```

Or use it programmatically:

```python
from vocabulary_crew import VocabularyTutorCrew

crew = VocabularyTutorCrew(word="courage", age_group="kid", track="learn")
result = crew.run()
print(result)
```

## CrewAI tools directory listing

To list these tools in the [CrewAI tools directory](https://docs.crewai.com/tools/):

1. Publish as a package (e.g. `crewai-wordorb` on PyPI).
2. Open a PR to `crewai/crewai` with documentation under `docs/tools/wordorb.mdx`.
3. Include:
   - Tool names and descriptions
   - Setup instructions (API key, env vars)
   - A working example showing at least one agent using the tools
   - Link to WordOrb API documentation at `https://wordorb.ai/docs`

## License

MIT
