"""Reward system for the Phrasal Paraphraser."""

import difflib
from collections import Counter
from logging import Logger
from typing import Any, cast

import structlog
import torch
from rapidfuzz.distance.Levenshtein import distance

from training_wizard.specs.reward import RewardSpec

logger: Logger = structlog.get_logger()
torch.set_float32_matmul_precision("high")


def get_edits(source: str, target: str) -> list[tuple[str, str, str]]:
    """Extract non-equal edit operations between two strings.

    Args:
        source: The original string.
        target: The modified string.

    Returns:
        A list of tuples, where each tuple contains:
        - tag: The type of edit operation ('replace', 'delete', or 'insert')
        - source_segment: The segment from the source string involved in the edit
        - target_segment: The segment from the target string involved in the edit
    """
    matcher = difflib.SequenceMatcher(None, source, target)
    edits = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            edits.append((tag, source[i1:i2], target[j1:j2]))
    return edits


def get_metrics(source: str, hypothesis: str, target: str) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 score for the hypothesis relative to the ground truth target.

    Args:
        source: The original text.
        hypothesis: The model's predicted correction.
        target: The ground truth correction.

    Returns:
        A tuple containing:
        - precision: Fraction of hypothesis edits that are correct (i.e., also in the source→target edits).
        - recall: Fraction of ground-truth (source→target) edits that the hypothesis captured.
        - f1: Harmonic mean of precision and recall.
    """
    true_edits = get_edits(source, target)
    hyp_edits = get_edits(source, hypothesis)

    # If there are no edits to be made (source == target) and model made no edits (source == hypothesis)
    # then precision and recall should be 1
    if not true_edits and not hyp_edits:
        return 1.0, 1.0, 1.0

    true_counter = Counter(true_edits)
    hyp_counter = Counter(hyp_edits)

    correct = sum(min(true_counter[edit], hyp_counter[edit]) for edit in true_counter)

    precision = correct / sum(hyp_counter.values()) if hyp_counter else 0
    recall = correct / sum(true_counter.values()) if true_counter else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


class GECExampleRewardSpec(RewardSpec):
    """A specification for a binary reward model that uses multiple rewards."""

    def parse_prompts_and_completions(
        self,
        prompts: list[str] | list[list[dict[str, str]]],
        completions: list[str] | list[list[dict[str, str]]],
    ) -> tuple[list[str], list[str]]:
        """Parse the prompts and completions."""
        if isinstance(prompts[0], list):
            prompts = cast("list[list[dict[str, str]]]", prompts)
            completions = cast("list[list[dict[str, str]]]", completions)
            prompts_parsed = [prompt[-1]["content"] for prompt in prompts]
            completions_parsed = [completion[-1]["content"] for completion in completions]
        else:
            prompts = cast("list[str]", prompts)
            completions = cast("list[str]", completions)
            prompts_parsed = [prompt.replace("<sep>", "") for prompt in prompts]
            completions_parsed = completions
        return prompts_parsed, completions_parsed

    def _try_log(self, name: str, value: float):
        """Try to log the reward."""
        if self.trainer is None:
            return

        if (
            self.trainer.is_world_process_zero()
            and self.trainer.state.global_step % self.trainer.args.logging_steps == 0
        ):
            self.trainer._metrics["train"][f"gec_gkd/{name}"].append(value)  # type: ignore

    @torch.inference_mode()
    def compute_reward(
        self,
        prompts: list[str] | list[list[dict[str, str]]],
        completions: list[str] | list[list[dict[str, str]]],
        **kwargs: Any,
    ) -> list[float]:
        """Compute the reward for a set of prompts and completions."""
        prompts_parsed, completions_parsed = self.parse_prompts_and_completions(prompts, completions)

        assert "target" in kwargs, "target column must be provided for this example reward"
        target_completions = kwargs["target"]
        lev_distances = [
            distance(completion, target)
            for completion, target in zip(completions_parsed, target_completions, strict=True)
        ]
        self._try_log("completion_target_distance", sum(lev_distances) / len(lev_distances))
        metrics = [
            get_metrics(prompt, completion, target)
            for prompt, completion, target in zip(prompts_parsed, completions_parsed, target_completions, strict=True)
        ]
        mean_precision = sum(p for p, _, _ in metrics) / len(metrics)
        mean_recall = sum(r for _, r, _ in metrics) / len(metrics)
        mean_f1 = sum(f for _, _, f in metrics) / len(metrics)
        self._try_log("precision", mean_precision)
        self._try_log("recall", mean_recall)
        self._try_log("f1", mean_f1)
        f1_scores = [f for _, _, f in metrics]
        # Reward is the F1 score
        return f1_scores
