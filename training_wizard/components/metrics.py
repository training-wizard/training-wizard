"""Common metric bundles."""

from logging import Logger
from statistics import mean

import seaborn as sns
import structlog
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from rapidfuzz.distance.Indel import normalized_similarity
from torchmetrics import MetricCollection, PrecisionRecallCurve
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryFBetaScore,
    BinaryMatthewsCorrCoef,
    BinaryPrecision,
    BinaryRecall,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassFBetaScore,
    MulticlassMatthewsCorrCoef,
    MulticlassPrecision,
    MulticlassPrecisionRecallCurve,
    MulticlassRecall,
)
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    PearsonCorrCoef,
    SpearmanCorrCoef,
)
from torchmetrics.text import BLEUScore

logger: Logger = structlog.get_logger()


@torch.no_grad()
def llm_logit_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss: torch.Tensor | None = None,
    return_perplexity: bool = True,
    return_token_accuracy: bool = True,
    return_logit_std: bool = True,
    return_mean_top_5_logit_diff: bool = True,
    return_mean_target_sequence_log_prob: bool = True,
) -> dict[str, float]:
    """Compute various metrics based on model logits and labels.

    Compute perplexity, token accuracy, logit standard deviation, entropy, mean top-5 logit difference,
    and mean log-probability of the full target sequence.

    Args:
        logits: Model output logits of shape (batch, seq_len, vocab).
        labels: Ground-truth ids with padding set to -100.
        loss: Optional loss tensor from the forward pass.
        return_perplexity: Log perplexity of the batch.
        return_token_accuracy: Exact-match next-token accuracy.
        return_logit_std: Standard deviation of logits.
        return_mean_top_5_logit_diff: Gap between highest and 5-th highest logit.
        return_mean_target_sequence_log_prob: Mean log-probability of the entire
            target sequence (masked by padding).

    Returns:
        Dictionary with the requested metrics.
    """
    metrics = {}
    labels = labels.to(logits.device)

    if loss is not None and return_perplexity:
        try:
            perplexity = torch.exp(loss.mean()).item()
            metrics["perplexity"] = perplexity
        except OverflowError:
            logger.warning("Perplexity calculation resulted in overflow, logging infinity.")
            metrics["perplexity"] = float("inf")

    mask = labels != -100
    num_non_padding_tokens = mask.sum().item()

    if num_non_padding_tokens == 0:
        logger.debug("No non-padding tokens found in batch, skipping logit metrics.")
        return metrics

    if return_token_accuracy:
        preds = torch.argmax(logits, dim=-1)
        correct_predictions = ((preds == labels) & mask).sum().item()
        token_accuracy = correct_predictions / num_non_padding_tokens
        metrics["token_accuracy"] = token_accuracy

    if return_logit_std:
        logit_std = torch.std(logits, dim=-1)
        masked_logit_std = logit_std[mask]
        avg_logit_std = masked_logit_std.mean().item()
        metrics["logit_std"] = avg_logit_std

    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    masked_entropy = entropy[mask]
    avg_entropy = masked_entropy.mean().item()
    metrics["entropy"] = avg_entropy

    vocab_size = logits.shape[-1]
    if return_mean_top_5_logit_diff:
        if vocab_size >= 5:
            top_5_logits, _ = torch.topk(logits, 5, dim=-1)
            top_5_diff = top_5_logits[..., 0] - top_5_logits[..., 4]
            masked_top_5_diff = top_5_diff[mask]
            avg_top_5_diff = masked_top_5_diff.mean().item()
            metrics["mean_top_5_logit_diff"] = avg_top_5_diff
        else:
            logger.warning(f"Vocab size ({vocab_size}) is less than 5, cannot compute top-5 logit difference.")

    if return_mean_target_sequence_log_prob:
        log_probs = torch.log_softmax(logits, dim=-1)
        safe_labels = labels.clone()
        safe_labels[~mask] = 0
        true_token_log_probs = torch.gather(log_probs, 2, safe_labels.unsqueeze(-1).long()).squeeze(-1)
        true_token_log_probs = true_token_log_probs * mask
        sequence_log_probs = true_token_log_probs.sum(dim=1)
        valid_seq_mask = mask.sum(dim=1) > 0
        if valid_seq_mask.any():
            mean_log_prob = sequence_log_probs[valid_seq_mask].mean().item()
            metrics["mean_target_sequence_log_prob"] = mean_log_prob

    return metrics


@torch.no_grad()
def multiclass_classification_metrics(
    y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int, plots: bool = True
) -> dict[str, float | Figure]:
    """Compute multiclass classification metrics.

    Args:
        y_true: True labels (integer tensor with class indices).
        y_pred: Predicted labels (class probabilities or logits).
        num_classes: The number of classes.
        plots: Whether to create plots.

    Returns:
        A dictionary containing the following metrics:
        - accuracy: Accuracy.
        - precision: Precision (macro average).
        - recall: Recall (macro average).
        - f1_score: F1 score (macro average).
        - f_half_score: F0.5 score (macro average).
        - auroc: Area under the ROC curve (macro average).
        - matthews_corr_coef: Matthews correlation coefficient.

        If `plots` is True, also:
        - precision_recall_curve: Precision-recall curve plot for each class.
        - density_plot: Joint density plot of labels and predicted probabilities.
    """
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()

    # Initialize metric collection for multiclass metrics
    metrics = MetricCollection({
        "accuracy": MulticlassAccuracy(num_classes=num_classes, average="macro"),
        "precision": MulticlassPrecision(num_classes=num_classes, average="macro"),
        "recall": MulticlassRecall(num_classes=num_classes, average="macro"),
        "f1_score": MulticlassF1Score(num_classes=num_classes, average="macro"),
        "f_half_score": MulticlassFBetaScore(num_classes=num_classes, beta=0.5, average="macro"),
        "auroc": MulticlassAUROC(num_classes=num_classes),
        "matthews_corr_coef": MulticlassMatthewsCorrCoef(num_classes=num_classes),
    })

    # Compute metrics
    metrics = {k: v.item() for k, v in metrics(preds=y_pred, target=y_true).items()}

    if plots:
        pr_curve = MulticlassPrecisionRecallCurve(num_classes=num_classes)
        pr_curve.update(preds=y_pred, target=y_true)
        plot: Figure = pr_curve.plot()[0]
        plot.axes[0].set_ylim(0, 1)
        metrics["precision_recall_curve"] = plot

        # Create a density figure with seaborn
        y_true_np, y_pred_np = y_true.numpy(), y_pred.softmax(dim=1).numpy().max(axis=1)
        density_plot = sns.displot(
            data={"y_true": y_true_np, "y_pred": y_pred_np},
            kind="kde",
            fill=True,
            clip=(y_true_np.min(), y_pred_np.max()),
        ).figure
        metrics["density_plot"] = density_plot

    return metrics


@torch.no_grad()
def binary_classification_metrics(
    y_true: torch.Tensor, y_pred: torch.Tensor, plots: bool = True
) -> dict[str, float | Figure]:
    """Compute binary classification metrics.

    Args:
        y_true: True labels (0 or 1). Int tensor.
        y_pred: Predicted labels (between 0 and 1).
        plots: Whether to create plots.

    Returns:
        A dictionary containing the following metrics:
        - accuracy: Accuracy.
        - precision: Precision.
        - recall: Recall.
        - f1_score: F1 score.
        - f_half_score: F0.5 score.
        - auroc: Area under the ROC curve.
        - matthews_corr_coef: Matthews correlation coefficient.

        If `plots` is True, also:
        - precision_recall_curve: Precision-recall curve plot.
        - density_plot: Joint density plot of labels and predicted probabilities.
    """
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()

    metrics = MetricCollection({
        "accuracy": BinaryAccuracy(),
        "precision": BinaryPrecision(),
        "recall": BinaryRecall(),
        "f1_score": BinaryF1Score(),
        "f_half_score": BinaryFBetaScore(beta=0.5),
        "auroc": BinaryAUROC(),
        "matthews_corr_coef": BinaryMatthewsCorrCoef(),
    })
    metrics = {k: v.item() for k, v in metrics(preds=y_pred, target=y_true).items()}

    if plots:
        if y_pred.ndim == 1:
            pr_curve = PrecisionRecallCurve(task="binary")
        else:
            pr_curve = PrecisionRecallCurve(task="multilabel", num_labels=y_pred.shape[1])
        pr_curve.update(preds=y_pred, target=y_true)
        plot: Figure = pr_curve.plot(score=True)[0]
        plot.axes[0].set_ylim(0, 1)
        metrics["precision_recall_curve"] = plot

        # Create a density figure with seaborn
        y_true_np, y_pred_np = y_true.cpu().numpy(), y_pred.cpu().numpy()

        # Handle multilabel data
        if y_pred_np.ndim > 1:
            # For multilabel, create separate distributions for each label
            fig, axes = plt.subplots(y_pred_np.shape[1], 1, figsize=(8, 4 * y_pred_np.shape[1]))
            axes = axes if y_pred_np.shape[1] > 1 else [axes]

            for i, ax in enumerate(axes):
                sns.kdeplot(
                    data={f"true_label_{i}": y_true_np[:, i], f"pred_label_{i}": y_pred_np[:, i]},
                    ax=ax,
                    fill=True,
                )
                ax.set_title(f"Label {i}")

            plt.tight_layout()
            density_plot = fig
        else:
            # Original single-label code
            density_plot = sns.displot(
                data={"y_true": y_true_np, "y_pred": y_pred_np},
                kind="kde",
                fill=True,
                clip=(y_true_np.min(), y_pred_np.max()),
            ).figure
        metrics["density_plot"] = density_plot

    return metrics


@torch.no_grad()
def regression_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, plots: bool = True) -> dict[str, float | Figure]:
    """Compute regression metrics.

    Args:
        y_true: True values.
        y_pred: Predicted values.
        plots: Whether to create plots.

    Returns:
        A dictionary containing the following metrics:
        - mse: Mean squared error.
        - mae: Mean absolute error.
        - mape: Mean absolute percentage error.
        - pearson_corr: Pearson correlation coefficient.
        - spearman_corr: Spearman correlation coefficient.
    """
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()

    metrics = MetricCollection({
        "mse": MeanSquaredError(),
        "mae": MeanAbsoluteError(),
        "mape": MeanAbsolutePercentageError(),
        "pearson_corr": PearsonCorrCoef(),
        "spearman_corr": SpearmanCorrCoef(),
    })
    metrics = {k: v.item() for k, v in metrics(preds=y_pred, target=y_true).items()}

    if plots:
        # Create a density figure with seaborn
        y_true_np, y_pred_np = y_true.numpy(), y_pred.numpy()
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.hist2d(y_true_np, y_pred_np, bins=30, cmap="Blues")

        # Add labels and title to the plot
        ax.set_title("True vs Predicted", fontsize=14, weight="bold")
        ax.set_xlabel("True Values", fontsize=12)
        ax.set_ylabel("Predicted Values", fontsize=12)
        metrics["regression_plot"] = fig

    return metrics


@torch.no_grad()
def seq2seq_metrics(
    target: list[str] | None,
    hypothesis: list[str],
    source: list[str],
) -> dict[str, float | Figure]:
    """Compute Seq2Seq metrics.

    Args:
        target: Corrected sentences. Optional.
        hypothesis: Predicted sentences.
        source: Original sentences.

    Returns:
        A dictionary containing the following metrics:
        - bleu: BLEU score.
        - src_hyp_ratio: Edit distance ratio between source and hypothesis.

        If `target` is provided:
        - hyp_tgt_ratio: Edit distance ratio between hypothesis and target.
    """
    result: dict[str, float | Figure] = {}
    bleu = BLEUScore(smooth=True)
    if target:
        # BLEUScore expects target to be a sequence of sequences, so wrap each target in a list
        target_sequences = [[t] for t in target]
        bleu_score: float = bleu(preds=hypothesis, target=target_sequences).item()
        result["bleu"] = bleu_score

    src_hyp_ratio = mean(normalized_similarity(h, t) for h, t in zip(source, hypothesis, strict=True))
    result["src_hyp_ratio"] = src_hyp_ratio

    if target:
        hyp_tgt_ratio = mean(normalized_similarity(h, t) for h, t in zip(hypothesis, target, strict=True))
        result["hyp_tgt_ratio"] = hyp_tgt_ratio
    return result
