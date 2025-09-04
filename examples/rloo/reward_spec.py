"""A reward specification for training language models with multiple reward components.

This module provides a comprehensive reward function that combines multiple metrics to evaluate
the quality of model outputs. The metrics include:

- Model-based reward from a classifier
- Length ratio between input and output
- Proper end-of-sequence token usage
- Jaccard distance for vocabulary overlap
- Levenshtein distance for overall similarity
- Character type matching at string boundaries
- Preservation of quoted content
- Matching of uncommon/special characters
- Punctuation pattern matching

The rewards are weighted and combined into a final score between 0 and 1.
"""

import math
import re
import string
import unicodedata
from functools import cached_property
from logging import Logger

import numpy as np
import structlog
import torch
import torch.distributed as dist
from rapidfuzz.distance.Indel import normalized_similarity
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForSequenceClassification, PreTrainedModel

from training_wizard.specs.model import TransformerSpec
from training_wizard.specs.reward import RewardSpec

logger: Logger = structlog.get_logger()


def is_common_char(c: str) -> bool:
    """Check if a character is considered common.

    Common characters include:
    - ASCII letters (upper and lowercase)
    - Numbers
    - Basic punctuation
    - Regular spaces and non-breaking spaces

    Args:
        c: The character to check

    Returns:
        bool: True if the character is common, False otherwise
    """
    category = unicodedata.category(c)
    # Ll: lowercase letter, Lu: uppercase letter, Nd: decimal number
    if category in {"Ll", "Lu", "Nd"}:
        return True
    # Basic punctuation and space
    if c in string.punctuation + " ":
        return True
    # Common whitespace (but not newlines or tabs)
    if c in {" ", "\xa0"}:  # regular space and non-breaking space  # noqa: SIM103
        return True
    return False


class Seq2SeqClassifierRewardSpec(RewardSpec):
    """A reward specification that combines multiple metrics to evaluate sequence-to-sequence outputs.

    This class provides a comprehensive reward function that evaluates model outputs based on:
    - A classifier model's score
    - Length ratio between input/output
    - Proper end-of-sequence token usage
    - Text similarity metrics (Jaccard, Levenshtein)
    - Character-level features (edge characters, quotes, special chars)
    - Punctuation patterns

    The individual reward components are weighted and combined into a final score.
    """

    transformer_spec: TransformerSpec
    """The transformer model specification for the classifier."""

    source_capture_regex: str = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"
    """Regular expression to extract the source text from a query. Default pattern is for Qwen 2.5."""

    source_capture_regex_group: int = 1
    """The capture group index for the source regex."""

    target_capture_regex: str = r"(.*?)(?:<\|im_end\|>|$)"
    """Regular expression to extract the target text from a completion. Default pattern is for Qwen 2.5."""

    target_capture_regex_group: int = 1
    """The capture group index for the target regex."""

    input_template: str = "{source}\n\n{target}\n\nScore:"
    """Template string for formatting classifier model inputs."""

    noop_token: str | None = None
    """Token indicating a no-operation response."""

    print_samples_freq: int | None = None
    """How often to print sample outputs and their scores (in steps). None disables printing."""

    # Reward component weights (will be normalized to sum to 1)

    model_reward: float = 10
    """Weight for the classifier model's score."""

    length_ratio_reward: float = 1
    """Weight for the input/output length ratio score."""

    eos_token_reward: float = 1
    """Weight for proper end-of-sequence token usage."""

    jaccard_distance_reward: float = 2
    """Weight for Jaccard distance score."""

    levenshtein_distance_reward: float = 1
    """Weight for Levenshtein distance score."""

    edge_character_type_reward: float = 1
    """Weight for matching character types at string boundaries."""

    quotes_match_reward: float = 1
    """Weight for preserving quoted content."""

    uncommon_characters_match_reward: float = 1
    """Weight for matching special/uncommon characters."""

    punctuation_match_reward: float = 1
    """Weight for matching punctuation patterns."""

    def __init__(self, **kwargs):
        """Initialize the reward specification."""
        super().__init__(**kwargs)
        self._step = 0

    @cached_property
    def model(self) -> PreTrainedModel:
        """Get the classifier model.

        Returns:
            The initialized classifier model on the appropriate device.
        """
        if dist.is_initialized():
            logger.info("DDP initialized")
            # Set device based on rank
            device = f"cuda:{dist.get_rank()}"
            logger.info(f"Overriding device_map in judge {self.transformer_spec.device_map} -> {device}")
            self.transformer_spec.device_map = device
        model = self.transformer_spec.create_model(AutoModelForSequenceClassification)
        assert model.config.num_labels == 1, "Reward model must be a classifier with a single label."
        return model

    def parse_output(self, query: str, completion: str) -> tuple[str, str]:
        """Extract source and target text from model prompt/completion.

        Args:
            query: The input query text
            completion: The model's completion text

        Returns:
            A tuple of (source_text, target_text)
        """
        query_match = re.search(self.source_capture_regex, query, flags=re.DOTALL | re.MULTILINE)
        completion_match = re.search(self.target_capture_regex, completion, flags=re.DOTALL | re.MULTILINE)
        if not query_match:
            logger.warning(f"Could not find source pattern `{self.source_capture_regex}` in query: {query}")
            src_final = query
        else:
            src_final = query_match.group(self.source_capture_regex_group).strip()
            if src_final == "":
                logger.warning(f"Source is empty. Using full query. Query: {query}")
                src_final = query
        if not completion_match:
            if self.noop_token and self.noop_token in completion:
                tgt_final = src_final
            else:
                logger.warning(
                    f"Could not find target pattern `{self.target_capture_regex}` in completion: {completion}"
                )
                tgt_final = completion
        else:
            tgt_final = completion_match.group(self.target_capture_regex_group).strip()
            if tgt_final == "":
                logger.warning(f"Target is empty. Using full completion. Completion: {completion}")
                tgt_final = completion
        return src_final, tgt_final

    def compute_model_reward(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Get reward scores from the classifier model.

        Args:
            pairs: List of (source, target) text pairs to score

        Returns:
            List of classifier scores between 0 and 1
        """
        inputs = [self.input_template.format(source=src, target=tgt) for src, tgt in pairs]
        enc = self.transformer_spec.encode_batch(inputs, return_tensors="pt", padding=True).to(device=self.model.device)
        outputs = self.model(**enc)
        scores = outputs.logits.squeeze().sigmoid().to(device=torch.device("cpu"), dtype=torch.float32).tolist()
        return scores

    def compute_length_ratio_reward(self, source: str, prediction: str) -> float:
        """Calculate reward based on the length ratio between source and prediction.

        Gives higher scores when the lengths are similar, with a peak at 1:1 ratio.

        Args:
            source: Source text
            prediction: Predicted text

        Returns:
            Score between 0 and 1
        """
        length_ratio = len(prediction) / len(source)
        # Peak at 1.0 with a plateau, sharper falloff for ratios far from 1
        # This creates a stronger penalty for very different lengths
        return math.exp(-10 * (2 * length_ratio - 2) ** 4)

    def compute_eos_token_reward(self, completion: str) -> float:
        """Check for proper end-of-sequence token usage.

        Args:
            completion: The completion text

        Returns:
            1.0 if EOS token is present, 0.0 otherwise
        """
        return 1.0 if self.transformer_spec.tokenizer.eos_token in completion else 0.0  # type: ignore

    def compute_jaccard_distance_reward(self, source: str, prediction: str) -> float:
        """Calculate reward based on token overlap using Jaccard distance.

        Gives higher scores for moderate overlap - penalizes both too much and too little similarity.
        Only applies to longer texts (>50 chars).

        Args:
            source: Source text
            prediction: Predicted text

        Returns:
            Score between 0 and 1
        """
        if len(source) < 50:
            return 1.0  # Short source, ignore
        source_tokens = set(self.transformer_spec.tokenizer.tokenize(source))
        completion_tokens = set(self.transformer_spec.tokenizer.tokenize(prediction))
        intersection = source_tokens & completion_tokens
        union = source_tokens | completion_tokens
        jaccard_distance = len(intersection) / len(union)
        # Allow for 0.4 to 0.6 distance, with a sharper falloff for distances outside that range
        return math.exp(-10 * (2 * jaccard_distance - 1) ** 4)

    def compute_levenshtein_distance_reward(self, source: str, prediction: str) -> float:
        """Calculate reward based on edit distance between texts.

        Gives higher scores for moderate edit distances - penalizes both identical and very different texts.

        Args:
            source: Source text
            prediction: Predicted text

        Returns:
            Score between 0 and 1
        """
        levenshtein_distance = 1 - normalized_similarity(source, prediction)
        return math.exp(-10 * (2 * levenshtein_distance - 1) ** 4)

    def compute_punctuation_match_reward(self, source: str, prediction: str) -> float:
        """Calculate reward based on matching punctuation patterns.

        Compares the frequency of each punctuation character between source and prediction.

        Args:
            source: Source text
            prediction: Predicted text

        Returns:
            Score between 0 and 1 based on punctuation matching
        """
        # Handle empty strings
        if not source or not prediction:
            return 0.0  # Empty strings should be penalized

        # Count occurrences of each punctuation character
        source_counts = {c: source.count(c) for c in string.punctuation if c in source}
        pred_counts = {c: prediction.count(c) for c in string.punctuation if c in prediction}

        # Get union of all punctuation chars
        all_punct = set(source_counts) | set(pred_counts)

        if not all_punct:
            return 1.0  # No punctuation in either string

        # Compare counts for each punctuation char
        matches = sum(min(source_counts.get(p, 0), pred_counts.get(p, 0)) for p in all_punct)
        total = sum(max(source_counts.get(p, 0), pred_counts.get(p, 0)) for p in all_punct)

        return matches / total if total > 0 else 1.0

    def compute_edge_character_type_reward(self, source: str, prediction: str) -> float:
        """Calculate reward based on matching character types at string boundaries.

        Checks if the first and last characters match in terms of:
        - Letter vs non-letter
        - Punctuation vs non-punctuation
        - Case (upper/lower)

        Args:
            source: Source text
            prediction: Predicted text

        Returns:
            Score between 0 and 1
        """
        reward = 1.0

        for i in [0, -1]:
            if source[i].isalpha() != prediction[i].isalpha():
                reward *= 0.5
            if source[i] in string.punctuation and source[i] != prediction[i]:
                reward *= 0.5
            if source[i] in string.punctuation != prediction[i] in string.punctuation:
                reward *= 0.5
            if source[i].isupper() != prediction[i].isupper():
                reward *= 0.5

        return reward

    def compute_quotes_match_reward(self, source: str, prediction: str) -> float:
        """Calculate reward based on preservation of quoted content.

        Checks if text between various types of quotes is preserved or very similar.
        Handles multiple quote styles including curly quotes and guillemets.

        Args:
            source: Source text
            prediction: Predicted text

        Returns:
            Score between 0 and 1
        """
        reward = 1.0
        quote_pairs = [('"', '"'), ("'", "'"), (""", """), ("'", "'"), ("„", "“"), ("«", "»")]
        quotes_source = []
        quotes_prediction = []
        for q1, q2 in quote_pairs:
            quotes_source.extend(re.findall(q1 + r"(.*?)(?:" + q2 + "|$)", source))
            quotes_prediction.extend(re.findall(q1 + r"(.*?)(?:" + q2 + "|$)", prediction))
        if len(quotes_source) != len(quotes_prediction):
            return 0.0
        for quote_source, quote_prediction in zip(quotes_source, quotes_prediction):
            if quote_source == quote_prediction:
                pass
            elif normalized_similarity(quote_source, quote_prediction) < 0.9:
                return 0.0
            else:
                reward *= 0.75
        return reward

    def compute_uncommon_characters_match_reward(self, source: str, prediction: str) -> float:
        """Calculate reward based on preservation of special/uncommon characters.

        Checks for preservation of:
        - Whitespace characters (tabs, newlines)
        - Bullet points and list markers
        - Special symbols and emoji
        - Other uncommon Unicode characters

        Args:
            source: Source text
            prediction: Predicted text

        Returns:
            Score between 0 and 1 based on matching of uncommon characters
        """
        # Find uncommon characters in source
        source_uncommon = {c for c in source if not is_common_char(c)}
        if not source_uncommon:
            return 1.0  # No uncommon chars to match

        # Find uncommon characters in prediction
        pred_uncommon = {c for c in prediction if not is_common_char(c)}

        # Calculate character presence match
        char_match = len(source_uncommon & pred_uncommon) / len(source_uncommon)

        # Calculate frequency similarity for matched characters
        freq_similarity = 1.0
        for char in source_uncommon & pred_uncommon:
            source_freq = source.count(char)
            pred_freq = prediction.count(char)
            # Allow some variation in frequency
            freq_ratio = min(source_freq, pred_freq) / max(source_freq, pred_freq)
            freq_similarity *= freq_ratio

        return char_match * freq_similarity

    @cached_property
    def reward_weights(self) -> np.ndarray:
        """Get the normalized weights for each reward component.

        Returns:
            Array of weights that sum to 1
        """
        rewards_unnormalized = np.array([
            self.model_reward,
            self.length_ratio_reward,
            self.eos_token_reward,
            self.jaccard_distance_reward,
            self.levenshtein_distance_reward,
            self.edge_character_type_reward,
            self.quotes_match_reward,
            self.uncommon_characters_match_reward,
        ])
        return rewards_unnormalized / rewards_unnormalized.sum()

    @torch.no_grad()
    def compute_reward(self, queries: list[str], completions: list[str]) -> list[float]:
        """Compute the combined reward scores for a batch of queries and completions.

        Combines all reward components using their weights to produce final scores.
        Optionally prints sample outputs and their scores based on print_samples_freq.

        Args:
            queries: List of input queries
            completions: List of model completions

        Returns:
            List of final reward scores between 0 and 1
        """
        pairs = [self.parse_output(query, completion) for query, completion in zip(queries, completions)]
        scores = []
        model_rewards = self.compute_model_reward(pairs)
        for model_reward, completion, (src, pred) in zip(model_rewards, completions, pairs):
            scores.append([
                model_reward,
                self.compute_length_ratio_reward(src, pred),
                self.compute_eos_token_reward(completion),
                self.compute_jaccard_distance_reward(src, pred),
                self.compute_levenshtein_distance_reward(src, pred),
                self.compute_edge_character_type_reward(src, pred),
                self.compute_quotes_match_reward(src, pred),
                self.compute_uncommon_characters_match_reward(src, pred),
            ])
        scores = np.array(scores)  # shape (N, 7)
        final_scores: np.ndarray = (scores * self.reward_weights).sum(axis=1)

        if (
            self.print_samples_freq is not None
            and self._step % self.print_samples_freq == 0
            and (not dist.is_initialized() or dist.get_rank() == 0)
        ):
            console = Console(color_system="auto")
            table = Table(title="Sample Rewards", show_lines=True)
            table.add_column("Source", style="cyan", overflow="fold")
            table.add_column("Prediction", style="green", overflow="fold")
            table.add_column("Score", justify="right")

            # Only show up to 5 examples
            for (src, pred), score in list(zip(pairs, final_scores))[:5]:
                # Calculate color based on score from red (0) to green (1)
                if score < 0.5:  # noqa: SIM108
                    # More red as score approaches 0
                    color = f"color(red) dim({score * 2})"
                else:
                    # More bright green as score approaches 1
                    color = f"color(green) dim({2 - score * 2})"

                table.add_row(
                    src,
                    pred,
                    f"[{color}]{score:.3f}[/]",
                )

            console.print(table)
        self._step += 1
        return final_scores.tolist()
