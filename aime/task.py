"""AIME Inspect AI task and dataset loading for GEPA optimization."""

from __future__ import annotations

import random
from typing import Any

from datasets import load_dataset
from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState, generate, system_message


def load_aime_datasets() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load AIME datasets in GEPA's expected format.

    Returns (trainset, valset) where each item is:
        {"input": str, "additional_context": {"solution": str}, "answer": "### N"}
    """
    train_split = [
        {
            "input": x["problem"],
            "additional_context": {"solution": x["solution"]},
            "answer": "### " + str(x["answer"]),
        }
        for x in load_dataset("AI-MO/aimo-validation-aime")["train"]
    ]
    random.Random(0).shuffle(train_split)

    trainset = train_split[: len(train_split) // 2]
    valset = train_split[len(train_split) // 2 :]
    return trainset, valset


def aime_data_to_sample(data: dict[str, Any], idx: int) -> Sample:
    """Convert a GEPA DataInst dict to an Inspect Sample."""
    # Target is the numeric answer (strip "### " prefix)
    target = data["answer"].replace("### ", "")
    return Sample(
        input=data["input"],
        target=target,
        id=idx,
    )


@scorer(metrics=[accuracy(), stderr()])
def aime_scorer() -> "Scorer":
    """Score AIME responses by extracting the number after '###'."""

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion
        # Look for "### <number>" pattern in the response
        import re

        matches = re.findall(r"###\s*(\d+)", completion)
        if matches:
            model_answer = matches[-1]  # take the last match
        else:
            # Fallback: try to find a standalone number at the end
            num_matches = re.findall(r"\b(\d+)\b", completion)
            model_answer = num_matches[-1] if num_matches else ""

        expected = target.text.strip()
        correct = model_answer == expected

        return Score(
            value=CORRECT if correct else INCORRECT,
            answer=model_answer,
            explanation=f"Expected {expected}, got {model_answer}",
        )

    return score


def aime_task_factory(system_prompt: str, samples: list[Sample]) -> Task:
    """Create an Inspect Task for AIME evaluation."""
    return Task(
        dataset=MemoryDataset(samples),
        solver=[system_message(system_prompt.replace("{", "{{").replace("}", "}}")), generate()],
        scorer=aime_scorer(),
        config=GenerateConfig(temperature=0.0, max_tokens=4096),
    )
