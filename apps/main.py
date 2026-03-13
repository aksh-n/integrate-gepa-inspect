"""Run GEPA optimization on the APPS Inspect AI task."""

from pathlib import Path

import gepa

from apps.task import apps_data_to_sample, apps_task_factory, load_apps_datasets
from gepa_inspect import InspectGEPAAdapter

SEED_PROMPT = (
    "You are an expert Python programmer. You will be given a task and test cases. "
    "Be mindful of the test case format. Write the complete Python function to solve "
    "the task. Before answering, reason step-by-step to get the right answer. "
    "Then write your solution in a ```python``` code block."
)


def main() -> None:
    trainset, valset = load_apps_datasets(n_samples=30)
    print(f"APPS dataset: {len(trainset)} train, {len(valset)} val")

    adapter = InspectGEPAAdapter(
        task_factory=apps_task_factory,
        data_to_sample=apps_data_to_sample,
        model="openai/gpt-4.1-mini",
    )

    result = gepa.optimize(
        seed_candidate={"system_prompt": SEED_PROMPT},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm="openai/gpt-5",
        max_metric_calls=80,
        reflection_minibatch_size=3,
        run_dir="apps/runs",
        cache_evaluation=True,
        display_progress_bar=True,
    )

    best_prompt = result.best_candidate
    if isinstance(best_prompt, dict):
        best_prompt = best_prompt["system_prompt"]

    best_score = result.val_aggregate_scores[result.best_idx]
    print(f"\nOptimization complete!")
    print(f"Best validation score: {best_score:.4f}")
    print(f"Best prompt:\n{best_prompt}")

    output_dir = Path(__file__).parent
    (output_dir / "best_prompt.txt").write_text(best_prompt)
    print(f"\nBest prompt saved to {output_dir / 'best_prompt.txt'}")


if __name__ == "__main__":
    main()
