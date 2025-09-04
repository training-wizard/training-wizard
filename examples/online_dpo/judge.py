"""Judge related specs."""

import math
import random
from functools import cached_property
from logging import Logger
from typing import cast

import structlog
import torch
import torch.distributed as dist
from pydantic import field_validator
from rapidfuzz.distance.Indel import normalized_similarity
from transformers import AutoModelForCausalLM, PreTrainedModel

from training_wizard.specs.judge import JudgeSpec
from training_wizard.specs.model import TransformerSpec

logger: Logger = structlog.get_logger()


class ParaphraserABJudgeSpec(JudgeSpec):
    """Specification for the judge."""

    transformer_spec: TransformerSpec
    """The specification for the transformer model to use as the judge."""

    messages_template: list[dict[str, str]]
    """Custom template to generate the conversational data from the dataset columns.
    Last user message should have the {input}, {A}, and {B} placeholders.

    Example:
    ```toml
    messages_template = [
        { role = "system", content = "Which is better? Answer with A or B." },
        { role = "user", content = "Input: {input}\nA: {A}\nB: {B}" },
    ]
    ```
    """

    peek_rate: int = 10
    """The rate at which to peek at the completions."""

    distance_reward: float | None = None
    """Reward multiplier for normalized jaccard distance."""

    first20_distance_reward: float | None = None
    """Reward multiplier for normalized levenshtein distance of the first 20 characters."""

    length_similarity_reward: float | None = None
    """Reward multiplier for length similarity."""

    def __init__(self, *args, **kwargs):
        """Initialize the judge."""
        super().__init__(*args, **kwargs)
        self._peek_steps = 0

    @cached_property
    def model(self) -> PreTrainedModel:
        """The model used in this module."""
        if dist.is_initialized():
            logger.info("DDP initialized")
            # Set device based on rank
            device = f"cuda:{dist.get_rank()}"
            logger.info(f"Overriding device_map in judge {self.transformer_spec.device_map} -> {device}")
            self.transformer_spec.device_map = device
        model = self.transformer_spec.create_model(AutoModelForCausalLM).eval()

        return cast("PreTrainedModel", model)

    @field_validator("transformer_spec")
    @classmethod
    def validate_transformer_spec(cls, v: TransformerSpec) -> TransformerSpec:
        """Get the tokenizer with validation."""
        if v.tokenizer.padding_side != "left":
            logger.warning("Setting tokenizer padding_side to 'left' for judge models")
            v.tokenizer.padding_side = "left"
        return v

    def compute_distance_reward(self, prompt: str, completion: str) -> float:
        """Compute the distance reward between the prompt and completion."""
        prompt_alpha = "".join(e for e in prompt if e.isalpha())
        completion_alpha = "".join(e for e in completion if e.isalpha())
        distance = 1 - normalized_similarity(prompt_alpha, completion_alpha)
        # Peak at 0.6, sharper falloff for low distances
        # This creates a stronger penalty for very similar texts
        # while still allowing them if the model is very confident
        return math.exp(-8 * (distance - 0.6) ** 2)

    def compute_first20_distance_reward(self, prompt: str, completion: str) -> float:
        """Compute the reward for the first 20 characters."""
        prompt_alpha = "".join(e for e in prompt if e.isalpha())
        completion_alpha = "".join(e for e in completion if e.isalpha())
        distance = 1 - normalized_similarity(prompt_alpha[:20], completion_alpha[:20])
        # Peak at 0.5, allowing for moderately different starts
        return math.exp(-6 * (distance - 0.5) ** 2)

    def compute_length_ratio_reward(self, prompt: str, completion: str) -> float:
        """Compute the length ratio reward between the prompt and completion.

        Returns:
            float: Reward between 0.0 (very different lengths) and 1.0 (similar lengths)
        """
        # Using word counts with small offset to handle very short texts
        length_ratio = (len(prompt.split()) + 3) / (len(completion.split()) + 3)
        # Convert to deviation from 1.0 (perfect ratio)
        deviation = abs(length_ratio - 1.0)
        # Exponential decay: 1.0 at deviation=0, approaching 0.0 as deviation increases
        return math.exp(-2 * deviation)

    def compute_rewards(self, prompt: str, completion: str) -> list[float]:
        """Compute the total reward for a completion."""
        rewards = []
        if self.distance_reward:
            rewards.append(self.distance_reward * self.compute_distance_reward(prompt, completion))
        if self.length_similarity_reward:
            rewards.append(self.length_similarity_reward * self.compute_length_ratio_reward(prompt, completion))
        if self.first20_distance_reward:
            rewards.append(self.first20_distance_reward * self.compute_first20_distance_reward(prompt, completion))
        return rewards

    @torch.no_grad()
    def judge(
        self, prompts: list[str], completions: list[list[str]], shuffle_order: bool = True, return_scores: bool = False
    ) -> list[int] | list[float]:
        """Judge the completion pairs for the given prompts.

        Args:
            prompts (`List[str]`): List of prompts.
            completions (`List[List[str]]`): List of completions pairs, where each element is a pair of completions for the corresponding prompt.
            shuffle_order (`bool`): Whether to shuffle the order of the completions to avoid positional bias.
            return_scores (`bool`): Whether to return the scores instead of the indices.

        Returns:
            List of idxs, where each idx is the rank of the best completion for the corresponding prompt.
            E.g., 1 means that the second completion (idx=1) is the best.
            Or list of scores if return_scores is True.

        Note:
            If the judge returns -1 for any prompt, it indicates that the inner process used to compute the preference has failed.
            For instance, this could occur if the underlying language model returned an invalid answer.
            In such cases, the caller should handle these invalid indices appropriately, possibly by implementing fallback logic or error handling.
        """  # noqa: E501
        self._peek_steps += 1
        prompts = [
            prompt.split("<pphr_input>")[-1].split("</pphr_input>", 1)[0].split("User:", maxsplit=1)[-1].lstrip()
            for prompt in prompts
        ]
        for i, completion_list in enumerate(completions):
            new_completion_list = []
            for completion in completion_list:
                completion = completion.strip()
                if completion.startswith("Assistant:"):
                    completion = completion[len("Assistant:") :].lstrip()
                if completion.find("<pphr_output>") != -1:
                    completion = completion.split("<pphr_output>")[-1].split("</pphr_output>", 1)[0]
                if "NO_PARAPHRASE" in completion:
                    completion = prompts[i]
                new_completion_list.append(completion)
            completions[i] = new_completion_list

        messages_batch = []
        swaps = []
        for prompt, completion in zip(prompts, completions):
            assert len(completion) == 2, "Expected 2 completions per prompt"
            template_copy = [d.copy() for d in self.messages_template]
            if random.random() < 0.5:
                swaps.append(True)
                a, b = completion[1], completion[0]
            else:
                swaps.append(False)
                a, b = completion[0], completion[1]
            template_copy[-1]["content"] = template_copy[-1]["content"].format(input=prompt, A=a, B=b)
            messages_batch.append(template_copy)

        enc = self.transformer_spec.tokenizer.apply_chat_template(
            messages_batch,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
            padding=True,
            **self.transformer_spec.tokenizer_call_kwargs,
        )
        enc = cast("dict[str, torch.Tensor]", enc)
        next_token_probs = (
            self.model(
                input_ids=enc["input_ids"].to(self.model.device),
                attention_mask=enc["attention_mask"].to(self.model.device),
            )
            .logits[:, -1]
            .softmax(dim=-1)
        )

        probs = []
        tokens_sorted = next_token_probs.argsort(dim=-1, descending=True).tolist()
        for i, tokens, swap_flag in zip(range(len(tokens_sorted)), tokens_sorted, swaps):
            probs_current = {}
            for token_id in tokens:
                token_str = self.transformer_spec.tokenizer.convert_ids_to_tokens(token_id)
                if "A" in token_str:
                    probs_current["A" if not swap_flag else "B"] = next_token_probs[i, token_id].item()
                elif "B" in token_str:
                    probs_current["B" if not swap_flag else "A"] = next_token_probs[i, token_id].item()
                if len(probs_current) == 2:
                    break
            probs.append(probs_current)

        if not all("A" in d and "B" in d for d in probs):
            logger.error("Judge: Missing A or B in probs. This should never happen.")
            return [-1] * len(prompts)

        for prompt, (a, b), prob_dict in zip(prompts, completions, probs):
            prob_dict["orig_A"] = prob_dict["A"]
            prob_dict["orig_B"] = prob_dict["B"]
            prob_dict["A"] = prob_dict["A"] + sum(self.compute_rewards(prompt, a))
            prob_dict["B"] = prob_dict["B"] + sum(self.compute_rewards(prompt, b))

        probs_pt = torch.tensor([[prob_dict["A"], prob_dict["B"]] for prob_dict in probs])
        probs_pt = probs_pt / probs_pt.sum(dim=-1, keepdim=True)
        if return_scores:  # noqa: SIM108
            results = probs_pt[:, 0].tolist()
        else:
            results = probs_pt.argmax(-1).tolist()

        if self._peek_steps % self.peek_rate == 0 and torch.cuda.current_device() == 0:
            logger.info(f"Peeking at {self._peek_steps} steps:")
            statement = ""
            for prompt, completion_list, result, prob_dict in zip(prompts, completions, results, probs):
                statement += f"\nPrompt:\t{prompt}\n"
                for i, completion in enumerate(completion_list):
                    statement += f"Out {i}:\t{completion}\n"
                statement += f"Choice:\t{result}\n"
                rewards_a = [prob_dict["orig_A"], *self.compute_rewards(prompt, completion_list[0])]
                rewards_b = [prob_dict["orig_B"], *self.compute_rewards(prompt, completion_list[1])]
                statement += f"A:\t{prob_dict['A']} ({rewards_a})\n"
                statement += f"B:\t{prob_dict['B']} ({rewards_b})\n"
            logger.info(statement)
        return results
