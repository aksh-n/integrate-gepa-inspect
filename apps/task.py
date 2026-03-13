"""APPS Inspect AI task setup for GEPA optimization.

Reuses the scorer and dataset utilities from inspect_evals.apps.
"""

from __future__ import annotations

import random
import textwrap
from typing import Any

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate, prompt_template, system_message

from inspect_evals.apps.apps import (
    APPS_DATASET_REVISION,
    DATASET_PATH,
    record_to_sample,
    verify,
)
from inspect_evals.apps.huggingface_artifact.apps import AppsCode
from inspect_evals.constants import INSPECT_EVALS_CACHE_PATH
from inspect_evals.hf_dataset_script_helper import load_hf_dataset_with_script

APPS_PROMPT_TEMPLATE = textwrap.dedent(
    """
    # Now, complete the following task.

    ## Question:
    {question}

    ## Test Cases:
    ```python
    {test_list_str}
    ```

    ## Completion:
    """
)


def load_apps_datasets(
    n_samples: int = 50, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load and subsample APPS interview-level problems for GEPA.

    Returns (trainset, valset) where each item is a dict containing
    all fields needed to reconstruct an Inspect Sample.
    """
    dataset_location = INSPECT_EVALS_CACHE_PATH / "apps_dataset" / "apps"
    dataset = load_hf_dataset_with_script(
        repo_id=DATASET_PATH,
        record_to_sample=record_to_sample,
        builder_cls=AppsCode,
        cache_dir_fp=dataset_location,
        split="test",
        revision=APPS_DATASET_REVISION,
    )

    # Filter to interview-level problems (IDs 0-2999)
    dataset = dataset.filter(
        lambda s: s.metadata is not None
        and int(s.metadata["problem_id"]) in range(0, 3000)
    )

    # Convert Inspect Samples → GEPA DataInst dicts
    all_items = []
    for sample in dataset:
        all_items.append(
            {
                "input": sample.input if isinstance(sample.input, str) else str(sample.input),
                "answer": "passes_tests",
                "additional_context": {
                    "question": sample.metadata.get("question", "") if sample.metadata else "",
                    "test_list_str": sample.metadata.get("test_list_str", "") if sample.metadata else "",
                },
                # Preserve full sample data for reconstruction
                "_target": sample.target,
                "_metadata": sample.metadata,
                "_id": sample.id,
            }
        )

    # Subsample
    rng = random.Random(seed)
    rng.shuffle(all_items)
    all_items = all_items[:n_samples]

    # Split into train/val
    split = len(all_items) // 2
    return all_items[:split], all_items[split:]


def apps_data_to_sample(data: dict[str, Any], idx: int) -> Sample:
    """Reconstruct an Inspect Sample from a GEPA DataInst dict."""
    return Sample(
        input=data["_metadata"].get("question", data["input"]) if data.get("_metadata") else data["input"],
        target=data["_target"],
        id=idx,
        metadata=data.get("_metadata", {}),
    )


def apps_task_factory(system_prompt: str, samples: list[Sample]) -> Task:
    """Create an Inspect Task for APPS evaluation."""
    return Task(
        dataset=MemoryDataset(samples),
        solver=[
            system_message(system_prompt.replace("{", "{{").replace("}", "}}")),
            prompt_template(APPS_PROMPT_TEMPLATE),
            generate(),
        ],
        scorer=verify(),
        config=GenerateConfig(temperature=0.0),
        sandbox="docker",
    )
