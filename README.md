# LLM Output Arbitration System

A multi-agent pipeline that takes any LLM-generated output, routes it to multiple competing critic models that independently evaluate it for accuracy, consistency, and completeness, then synthesizes their critiques into a single confidence-scored verdict with actionable callouts.

Live demo: https://mansimengde17.github.io/LLM-Output-Arbitration-System/

## The idea

Instead of another system that generates answers, this one catches bad answers. Three specialized critics evaluate every output in parallel, each backed by a different model on purpose: models sharing weights share blind spots, and the disagreements between heterogeneous critics are the most valuable signal in the system. An adjudicator weighs the evidence, resolves conflicts, and produces the final verdict.

## Architecture

```
                     output under evaluation
                              |
                              v
                     +----------------+
                     |  input parsing  |
                     +----------------+
              parallel fan-out (asyncio.gather)
              /               |               \
             v                v                v
 +------------------+ +------------------+ +--------------------+
 | Factual Accuracy | |     Logical      | |   Completeness      |
 |      Critic       | |   Consistency    | |      Critic         |
 |   (GPT-4o slot)   | |  (Claude slot)   | |  (local Llama slot) |
 +------------------+ +------------------+ +--------------------+
              \               |               /
               v              v              v
                  +------------------------+
                  |  critique collection    |
                  |  disagreement detector  |
                  +------------------------+
                              |
             all agree it's clean?  -- yes --> short-circuit: pass
                              | no
                              v
                  +------------------------+
                  |      Adjudicator        |
                  |  evidence-based conflict |
                  |  resolution              |
                  +------------------------+
                              |
                              v
              verdict: score 1-10, confidence,
              confirmed issues, dismissed flags,
              one-paragraph assessment
```

## Quick start (offline)

The critics ship with deterministic offline evaluators (fact database, logic pattern checks, question-coverage analysis), so the full arbitration graph runs with no API keys:

```bash
pip install -r requirements.txt
python demo.py
```

The demo arbitrates four canonical cases: a factually wrong response, a logically flawed argument, a response that misses half the question, and a genuinely good response. Each produces a full verdict with the critics' individual reports.

To serve the API:

```bash
uvicorn src.arbitration.api:app --reload
# POST /v1/arbitrate         evaluate a single output
# POST /v1/arbitrate/batch   evaluate multiple outputs
# GET  /v1/arbitrations/{id} retrieve a past verdict
# GET  /v1/analytics         critic behavior meta-analysis
```

With `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / a local Ollama set, each critic slot calls its assigned live model with structured-output prompts instead of the offline evaluator.

## Structured critique format

Every critic returns the same typed structure (Pydantic): dimension, score 1 to 5, list of issues (each with a quote from the original, the problem, and severity 1 to 3), and the critic's own confidence. The disagreement detector flags cases where critics disagree on whether something is an issue, severity ratings differ by more than 2 points, or one critic found issues the others missed.

## Fault tolerance

If a critic fails (API error, timeout), the graph degrades gracefully: the verdict is produced from the remaining critics with the missing dimension marked low-confidence. If all critics return clean reports, the adjudicator is short-circuited and a high-confidence pass is returned without the extra model call.

## Analytics

Across arbitrations the system tracks which critic finds the most issues, which critic gets overruled by the adjudicator most often, the distribution of failure types, and critic agreement rates. `GET /v1/analytics` exposes the aggregates.

## Repository layout

```
src/arbitration/critics.py     three critic agents with offline evaluators
src/arbitration/graph.py       parallel dispatch, disagreement detection
src/arbitration/adjudicator.py evidence-based conflict resolution
src/arbitration/store.py       SQLite verdict store and analytics
src/arbitration/api.py         FastAPI service
demo.py                        four canonical arbitration cases
tests/                         unit tests
```
