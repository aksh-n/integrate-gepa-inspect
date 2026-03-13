"""Generic bridge between GEPA's optimization loop and Inspect AI's Task evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

from gepa.core.adapter import EvaluationBatch
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import CORRECT

# A TaskFactory receives a system prompt and a list of Samples,
# and returns a fully configured Inspect Task ready to evaluate.
TaskFactory = Callable[[str, list[Sample]], Task]

# Converts a GEPA DataInst dict + batch index into an Inspect Sample.
# The batch index should be used as sample.id for reliable matching.
DataToSample = Callable[[dict[str, Any], int], Sample]


def _score_to_float(value: Any) -> float:
    """Convert an Inspect Score value to a float for GEPA."""
    if isinstance(value, (int, float)):
        return float(value)
    if value == CORRECT:
        return 1.0
    return 0.0


class InspectGEPAAdapter:
    """Generic adapter bridging GEPA optimization with any Inspect AI Task.

    The adapter is parameterised by two callables that encapsulate all
    task-specific logic:
      - task_factory: builds an Inspect Task from a system prompt and samples
      - data_to_sample: converts a GEPA data dict to an Inspect Sample
    """

    def __init__(
        self,
        task_factory: TaskFactory,
        data_to_sample: DataToSample,
        model: str,
        max_connections: int = 200,
    ) -> None:
        self.task_factory = task_factory
        self.data_to_sample = data_to_sample
        self.model = model
        self.max_connections = max_connections
        self.propose_new_texts = None  # use GEPA's default LLM-based proposer

    # ---- GEPAAdapter protocol ------------------------------------------------

    def evaluate(
        self,
        batch: list[dict[str, Any]],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        system_prompt = candidate["system_prompt"]

        # Convert GEPA data dicts → Inspect Samples with index-based IDs.
        samples = [self.data_to_sample(item, idx) for idx, item in enumerate(batch)]

        # Build and run the Inspect task.
        task = self.task_factory(system_prompt, samples)
        logs = inspect_eval(
            tasks=task,
            model=self.model,
            display="none",
            log_samples=True,
            max_connections=self.max_connections,
        )
        eval_log = logs[0]

        # Map sample id → EvalSample for ordered score extraction.
        sample_map: dict[int | str, Any] = {}
        if eval_log.samples:
            for s in eval_log.samples:
                sample_map[s.id] = s

        scores: list[float] = []
        outputs: list[str] = []
        trajectories: list[dict[str, Any]] | None = [] if capture_traces else None

        for idx, item in enumerate(batch):
            eval_sample = sample_map.get(idx)
            if eval_sample is None:
                scores.append(0.0)
                outputs.append("")
                if trajectories is not None:
                    trajectories.append(
                        {
                            "input": item.get("input", ""),
                            "output": "",
                            "score": 0.0,
                            "explanation": "Sample not found in eval results",
                        }
                    )
                continue

            # Extract score from first scorer.
            score_val = 0.0
            explanation = ""
            if eval_sample.scores:
                first_score = next(iter(eval_sample.scores.values()))
                score_val = _score_to_float(first_score.value)
                explanation = first_score.explanation or ""

            model_output = eval_sample.output.completion if eval_sample.output else ""

            scores.append(score_val)
            outputs.append(model_output)

            if trajectories is not None:
                trajectories.append(
                    {
                        "input": item.get("input", ""),
                        "output": model_output,
                        "score": score_val,
                        "explanation": explanation,
                        "target": item.get("answer", ""),
                    }
                )

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        records: list[dict[str, Any]] = []

        if eval_batch.trajectories:
            for traj in eval_batch.trajectories:
                correct = "Correct" if traj["score"] >= 1.0 else "Incorrect"
                feedback = f"{correct}."
                if traj.get("target"):
                    feedback += f" Expected answer: {traj['target']}."
                if traj.get("explanation"):
                    feedback += f" {traj['explanation']}"

                records.append(
                    {
                        "Inputs": traj["input"],
                        "Generated Outputs": traj["output"],
                        "Feedback": feedback,
                    }
                )

        return {component: records for component in components_to_update}
