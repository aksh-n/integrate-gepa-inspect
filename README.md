# GEPA × Inspect AI Integration

Automatic system prompt optimization for LLM evaluation tasks using [GEPA](https://github.com/gepa-ai/gepa) (Genetic-Pareto evolutionary search) and [Inspect AI](https://inspect.aisi.org.uk/).

## Results

| Benchmark | Seed Prompt Accuracy | Optimized Prompt Accuracy |
|-----------|---------------------|--------------------------|
| AIME      | 20.0%               | 40.0%                     |
| APPS      | 66.7%               | 73.3%                     |

**Task model**: GPT-4.1 mini · **Reflection model**: GPT-5

## Architecture

```
gepa_inspect/
  adapter.py         # Generic GEPA↔Inspect bridge (task-agnostic)

aime/
  task.py            # AIME Inspect Task: dataset, solver, scorer
  main.py            # Runs GEPA optimization on AIME

apps/
  task.py            # APPS Inspect Task: dataset loader, sample converter
  main.py            # Runs GEPA optimization on APPS
```

### Key design decisions

**Generic adapter** (`gepa_inspect/adapter.py`): The `InspectGEPAAdapter` class bridges GEPA and Inspect without any task-specific logic. It is parameterised by two callables:
- `task_factory(system_prompt, samples) → Task` — builds an Inspect Task from a candidate prompt and samples
- `data_to_sample(data_dict, idx) → Sample` — converts a GEPA data dict to an Inspect Sample

Swapping in a new Inspect task requires only implementing these two functions and a dataset loader.

**AIME task** (`aime/task.py`): Custom Inspect Task with a dataset loaded from HuggingFace (`AI-MO/aimo-validation-aime`), structured to match the GEPA AIME example format. A custom scorer extracts answers from the `### N` format.

**APPS task** (`apps/task.py`): Reuses the existing `verify()` scorer and dataset utilities from `inspect_evals.apps`. Problems are subsampled (30–50 from interview-level) to stay within budget. Uses Docker sandbox for code execution.

**Reflective dataset**: The adapter's `make_reflective_dataset()` builds structured feedback (correct/incorrect, expected answer, scorer explanation) that GEPA's reflection model uses to propose improved prompts.

## Limitations

- **Budget-constrained runs**: With a $50 API budget and `max_metric_calls` capped at 50–80, the evolutionary search explores a limited number of generations. Longer runs would likely find better prompts.
- **Small validation sets**: 15-sample val sets introduce high variance in score estimates. A single flipped answer changes accuracy by ~6.7%.
- **APPS sandbox overhead**: Each APPS evaluation spins up Docker containers for code execution, which adds latency and limits parallelism.
- **Single-component optimization**: Only the system prompt is optimized. Jointly optimizing prompt + solver chain (e.g., chain-of-thought structure, few-shot examples) could yield larger gains.
- **No cross-task transfer**: Prompts are optimized independently per task. Domain-specific guidance learned on one task (e.g., parsing patterns for APPS) does not transfer.

## Future improvements

- **Larger budgets and populations**: More metric calls and a larger candidate pool would allow GEPA's Pareto front to diversify further.
- **Solver co-optimization**: Extend the adapter to let GEPA mutate the solver chain (e.g., toggle chain-of-thought, adjust temperature) alongside the prompt.
- **Ensemble scoring**: Use multiple scorers or majority-vote across temperatures to reduce noise in score estimates.
- **Curriculum subsampling**: Instead of random subsampling, prioritize harder examples that distinguish candidate prompts.

## AI usage

This project was built with extensive use of Claude Code (Claude Opus). Claude was used for:
- Designing the architecture and generic adapter pattern
- Implementing all source files (adapter, tasks, runners)
- Debugging integration issues between GEPA and Inspect AI

## Setup and usage

```bash
# Install dependencies
uv sync

# Run AIME optimization
uv run python -m aime.main 2>&1 | tee aime/best_prompt_trace.txt

# Run APPS optimization (requires Docker for sandbox)
uv run python -m apps.main 2>&1 | tee apps/best_prompt_trace.txt
```

Requires `OPENAI_API_KEY` in the environment (or `.env` file).
