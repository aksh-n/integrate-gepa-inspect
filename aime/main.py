"""Run GEPA optimization on the AIME Inspect AI task."""

from pathlib import Path

import gepa

from aime.task import aime_data_to_sample, aime_task_factory, load_aime_datasets
from gepa_inspect import InspectGEPAAdapter

SEED_PROMPT = (
    "You are a helpful assistant. Answer the question. "
    "Put your final answer in the format '### <answer>'"
)


def main() -> None:
    trainset, valset = load_aime_datasets()
    # Keep small sets to finish within budget and time
    trainset = trainset[:15]
    valset = valset[:10]
    print(f"AIME dataset: {len(trainset)} train, {len(valset)} val")

    adapter = InspectGEPAAdapter(
        task_factory=aime_task_factory,
        data_to_sample=aime_data_to_sample,
        model="openai/gpt-4.1-mini",
    )

    result = gepa.optimize(
        seed_candidate={"system_prompt": SEED_PROMPT},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm="openai/gpt-5",
        max_metric_calls=50,
        reflection_minibatch_size=3,
        run_dir="aime/runs",
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
