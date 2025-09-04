"""Judge related specs."""

from abc import ABC
from logging import Logger

import numpy as np
import structlog
import torch
from trl import BasePairwiseJudge

from .reward import RewardSpec
from .spec import Spec

logger: Logger = structlog.get_logger()


class JudgeSpec(Spec, BasePairwiseJudge, ABC):
    """Specification for a judge."""


class RewardJudgeSpec(JudgeSpec):
    """Judge using reward model."""

    reward_spec: RewardSpec
    """The reward spec to use for scoring."""

    @torch.no_grad()
    def judge(
        self, prompts: list[str], completions: list[list[str]], shuffle_order: bool = True, return_scores: bool = False
    ) -> list[int]:
        """Judge a set of prompts and completions."""
        scores = np.array(
            [
                self.reward_spec.compute_reward(prompts, [t[0] for t in completions]),
                self.reward_spec.compute_reward(prompts, [t[1] for t in completions]),
            ]
        ).T
        if return_scores:  # noqa: SIM108
            results = scores[:, 0].tolist()
        else:
            results = scores.argmax(-1).tolist()
        assert len(results) == len(prompts)
        # If the scores are the same, set the result to -1
        for i, (s1, s2) in enumerate(zip(scores[:, 0], scores[:, 1])):
            if s1 == s2:
                results[i] = -1
        return results
