"""!!!EXPERIMENTAL AND UNTESTED!!!

Approximate Likelihood Matching (ALM) distillation module (minimal unconstrained + binary CE).
"""

from functools import cached_property
from logging import Logger
from typing import Any, cast

import numpy as np
import structlog
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModelForCausalLM, PreTrainedTokenizer, TrainerCallback

from ...components.wizard_trainer import WizardTrainer
from ..model import TransformerSpec
from ..modules.module import WizardModule

logger: Logger = structlog.get_logger()


# Taken from https://github.com/bminixhofer/tokenkit (simplified for unconstrained alignment)
def get_alignment_indices(
    tokens_teacher: list[str],
    tokens_student: list[str],
    mask_teacher: np.ndarray,
    mask_student: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """Compute chunk alignment between teacher and student token sequences by matching cumulative lengths.

    This is a direct port of tokenkit's "unconstrained" alignment,
    using cumulative character counts to decide chunk boundaries.

    Args:
        tokens_teacher: List of tokens from the teacher tokenizer.
        tokens_student: List of tokens from the student tokenizer.
        mask_teacher: Boolean mask for teacher tokens, shape [seq_len].
        mask_student: Boolean mask for student tokens, shape [seq_len].

    Returns:
        A list of tuples (start_i, end_i, start_j, end_j) representing aligned chunks.
    """
    i = j = 0
    cum_t = cum_s = 0
    cum_t_dict = {}
    cum_s_dict = {}
    start_i = start_j = 0
    alignments = []
    num_tokens_teacher = len(tokens_teacher)
    num_tokens_student = len(tokens_student)
    while i < num_tokens_teacher and j < num_tokens_student:
        if not mask_teacher[i]:
            i += 1
            continue
        if not mask_student[j]:
            j += 1
            continue
        cum_t = cum_t_dict.get(i - 1, 0) + len(tokens_teacher[i])
        cum_s = cum_s_dict.get(j - 1, 0) + len(tokens_student[j])
        cum_t_dict[i] = cum_t
        cum_s_dict[j] = cum_s
        if cum_t == cum_s:
            alignments.append((start_i, i + 1, start_j, j + 1))
            start_i = i + 1
            start_j = j + 1
            cum_t = cum_s = 0
            cum_t_dict.clear()
            cum_s_dict.clear()
        if cum_t <= cum_s:
            i += 1
        if cum_s <= cum_t:
            j += 1
    return alignments


def get_unconstrained_alignments(
    input_ids_teacher: np.ndarray,
    input_ids_student: np.ndarray,
    attention_mask_teacher: np.ndarray,
    attention_mask_student: np.ndarray,
    tokenizer_teacher: PreTrainedTokenizer,
    tokenizer_student: PreTrainedTokenizer,
) -> tuple[np.ndarray, np.ndarray]:
    """Build boolean alignment matrices of shape [batch, seq_len, num_chunks].

    Args:
        input_ids_teacher: Teacher input IDs (batch, seq_len).
        input_ids_student: Student input IDs (batch, seq_len).
        attention_mask_teacher: Teacher attention mask (batch, seq_len).
        attention_mask_student: Student attention mask (batch, seq_len).
        tokenizer_teacher: Teacher tokenizer for id→token mapping.
        tokenizer_student: Student tokenizer for id→token mapping.

    Returns:
        A tuple (align_s, align_t) each of shape [batch, seq_len, num_chunks].
    """
    bsz = input_ids_teacher.shape[0]
    # estimate chunk count per example dynamic, collect all then pad
    student_mats = []
    teacher_mats = []
    for batch_idx in range(bsz):
        toks_t = tokenizer_teacher.convert_ids_to_tokens(input_ids_teacher[batch_idx])
        toks_s = tokenizer_student.convert_ids_to_tokens(input_ids_student[batch_idx])
        mask_t = attention_mask_teacher[batch_idx].astype(bool)
        mask_s = attention_mask_student[batch_idx].astype(bool)
        align_idx = get_alignment_indices(toks_t, toks_s, mask_t, mask_s)
        nch = len(align_idx)
        mat_t = np.zeros((input_ids_teacher.shape[1], nch), dtype=bool)
        mat_s = np.zeros((input_ids_student.shape[1], nch), dtype=bool)
        for c, (si, ei, sj, ej) in enumerate(align_idx):
            mat_t[si:ei, c] = True
            mat_s[sj:ej, c] = True
        teacher_mats.append(mat_t)
        student_mats.append(mat_s)
    # pad matrices to common chunk dim
    max_c = max(m.shape[1] for m in student_mats)
    t_stack = np.zeros((bsz, input_ids_teacher.shape[1], max_c), dtype=bool)
    s_stack = np.zeros((bsz, input_ids_student.shape[1], max_c), dtype=bool)
    for i, (mt, ms) in enumerate(zip(teacher_mats, student_mats)):
        t_stack[i, :, : mt.shape[1]] = mt
        s_stack[i, :, : ms.shape[1]] = ms
    return s_stack, t_stack


def binary_cross_entropy_diff(
    t_logp: torch.Tensor,
    s_logp: torch.Tensor,
) -> torch.Tensor:
    """Compute binary cross-entropy between teacher and student aligned chunk log-probabilities.

    Args:
        t_logp: Teacher log-probabilities, shape (batch, chunks).
        s_logp: Student log-probabilities, shape (batch, chunks).

    Returns:
        The mean binary cross-entropy loss (scalar).
    """
    p_t = t_logp.exp()
    p_s = s_logp.exp()
    return F.binary_cross_entropy(p_s, p_t)


class ALMDistillModule(WizardModule):
    """WizardModule for Approximate Likelihood Matching (ALM) distillation.

    Handles data collation and loss computation for unconstrained-alignment ALM distillation.
    """

    teacher_spec: TransformerSpec
    """The teacher model specification."""

    student_spec: TransformerSpec
    """The student model specification."""

    @cached_property
    def callbacks(self) -> list[TrainerCallback]:
        """Return training callbacks. ALM does not use any special callbacks."""
        return []

    @torch.no_grad()
    def batch_metrics(self, inputs: dict[str, Any], outputs: Any) -> dict[str, float]:
        """Compute batch-level metrics (placeholder)."""
        return {}

    @torch.no_grad()
    def eval_metrics(self, eval_dataset: Dataset, batch_size: int) -> dict[str, float]:
        """Compute evaluation metrics over the validation split (placeholder)."""
        return {}

    def validate_dataset(self, ds: Dataset):
        """Validate that the dataset has the required 'text' column.

        Args:
            ds: A HuggingFace Dataset for training.

        Raises:
            AssertionError: If 'text' column is missing.
        """
        if "text" not in ds.column_names:
            raise AssertionError("Dataset must have a 'text' column")

    @cached_property
    def teacher(self) -> torch.nn.Module:
        """The teacher model."""
        teacher = self.teacher_spec.create_model(AutoModelForCausalLM).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        return teacher

    @cached_property
    def model(self) -> torch.nn.Module:
        """Instantiate & freeze the teacher, then return the student model.

        Returns:
            The student causal-LM model to be trained.
        """
        return self.student_spec.create_model(AutoModelForCausalLM)

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Return the student's tokenizer."""
        return self.student_spec.tokenizer

    def data_collator(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate text-only batch into model inputs & alignment matrices.

        Args:
            batch: List of examples with a 'text' field.

        Returns:
            A dict containing:
              - 'input_ids', 'attention_mask'             (student)
              - 'teacher_input_ids', 'teacher_attention_mask'
              - 'align_s', 'align_t'                     (alignment mats)
        """
        texts = [row["text"] for row in batch]
        # student encoding
        s_enc = self.student_spec.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        # teacher encoding
        t_enc = self.teacher_spec.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        # alignment
        input_ids_teacher = cast("torch.Tensor", t_enc["input_ids"])
        input_ids_student = cast("torch.Tensor", s_enc["input_ids"])
        attention_mask_teacher = cast("torch.Tensor", t_enc["attention_mask"])
        attention_mask_student = cast("torch.Tensor", s_enc["attention_mask"])
        s_mat, t_mat = get_unconstrained_alignments(
            input_ids_teacher.numpy(),
            input_ids_student.numpy(),
            attention_mask_teacher.numpy(),
            attention_mask_student.numpy(),
            self.teacher_spec.tokenizer,
            self.student_spec.tokenizer,
        )
        return {
            "input_ids": input_ids_student,
            "attention_mask": attention_mask_student,
            "teacher_input_ids": input_ids_teacher,
            "teacher_attention_mask": attention_mask_teacher,
            "align_s": torch.from_numpy(s_mat),
            "align_t": torch.from_numpy(t_mat),
        }

    def compute_loss(
        self,
        trainer: WizardTrainer,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the ALM distillation loss.

        Args:
            trainer: The active WizardTrainer.
            inputs: Batch from data_collator with alignments.
            return_outputs: If True, also return raw chunk-logits.

        Returns:
            The loss tensor, or (loss, outputs) if return_outputs=True.
        """
        # student forward
        assert trainer.model is not None, "Trainer model is None"

        s_out = trainer.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        s_logits = s_out.logits
        # teacher forward
        with torch.no_grad():
            t_out = self.teacher(input_ids=inputs["teacher_input_ids"], attention_mask=inputs["teacher_attention_mask"])
        t_logits = t_out.logits
        # main path logprobs
        t_logp = F.log_softmax(t_logits, dim=-1)
        s_logp = F.log_softmax(s_logits, dim=-1)
        # shift to next-token
        t_labels = inputs["teacher_input_ids"][:, 1:]
        s_labels = inputs["input_ids"][:, 1:]
        t_main = t_logp[:, :-1, :].gather(-1, t_labels.unsqueeze(-1)).squeeze(-1)
        s_main = s_logp[:, :-1, :].gather(-1, s_labels.unsqueeze(-1)).squeeze(-1)
        # alignment mats
        a_s = inputs["align_s"][:, :-1, :].float()
        a_t = inputs["align_t"][:, :-1, :].float()
        # aggregate per-chunk
        # shape [batch, chunks]
        s_chunk = torch.einsum("bsi,bs->bi", a_s, s_main)
        t_chunk = torch.einsum("bti,bt->bi", a_t, t_main)
        # binary CE
        loss = binary_cross_entropy_diff(t_chunk, s_chunk)
        if return_outputs:
            return loss, {"s_chunk": s_chunk, "t_chunk": t_chunk}
        return loss
