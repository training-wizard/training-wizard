"""HL-Gauss method from the Google paper.

Stop Regressing: Training Value Functions via Classification for Scalable Deep RL
https://arxiv.org/abs/2403.03950
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HLGauss(nn.Module):
    """Hard-to-Learn Gaussian distribution module."""

    def __init__(self, min_value: float, max_value: float, num_buckets: int, sigma: float | None = None):
        """Initialize the HL-Gauss module."""
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.num_buckets = num_buckets

        # Calculate bucket width (𝜍)
        bucket_width = (max_value - min_value) / num_buckets

        # Default sigma based on the paper recommendation: sigma/𝜍 = 0.75
        self.sigma = sigma or 0.75 * bucket_width

        # Define bucket boundaries and centers
        self.register_buffer("bucket_boundaries", torch.linspace(min_value, max_value, num_buckets + 1))
        self.bucket_boundaries: torch.Tensor
        self.register_buffer("bucket_centers", (self.bucket_boundaries[:-1] + self.bucket_boundaries[1:]) / 2)
        self.bucket_centers: torch.Tensor

    def to(self, *args, **kwargs) -> "HLGauss":
        """Move the module to a specific device."""
        self = super().to(*args, **kwargs)
        self.bucket_boundaries = self.bucket_boundaries.to(*args, **kwargs)
        self.bucket_centers = self.bucket_centers.to(*args, **kwargs)
        return self

    def score_to_distribution(self, scores: torch.Tensor) -> torch.Tensor:
        """Convert scores to a distribution over buckets."""
        scores = scores.unsqueeze(-1)

        # Numerical stability: avoid overflow issues with very large or small inputs
        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=scores.device))

        cdf_evals = torch.erf((self.bucket_boundaries - scores) / (self.sigma * sqrt_2).clamp(min=1e-8))

        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        return bin_probs / z.unsqueeze(-1).clamp(min=1e-8)

    def distribution_to_score(self, distribution: torch.Tensor) -> torch.Tensor:
        """Convert distribution over buckets to scores."""
        return (distribution * self.bucket_centers).sum(dim=-1).clip(self.min_value, self.max_value)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """Forward pass of the HL-Gauss module."""
        return self.score_to_distribution(scores)

    def loss(self, logits: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        """Compute the loss between logits and target scores."""
        return F.cross_entropy(logits, self.score_to_distribution(target_scores))


def run_tests():
    """Run tests for the HLGauss module."""
    print("Running tests for HLGauss module...")

    # Test 1: Initialization
    hl_gauss = HLGauss(min_value=-1.0, max_value=1.0, num_buckets=21, sigma=0.1)
    assert (
        hl_gauss.min_value == -1.0
        and hl_gauss.max_value == 1.0
        and hl_gauss.num_buckets == 21
        and hl_gauss.sigma == 0.1
    ), "Initialization test failed"
    print("Test 1: Initialization - Passed")

    # Test 2: Score to distribution
    scores = torch.tensor([-0.5, 0.0, 0.5])
    distributions = hl_gauss.score_to_distribution(scores)
    assert distributions.shape == (3, 21), f"Expected shape (3, 21), got {distributions.shape}"
    assert torch.allclose(distributions.sum(dim=-1), torch.ones(3)), "Distributions do not sum to 1"
    print("Test 2: Score to distribution - Passed")

    # Test 3: Distribution to score
    reconstructed_scores = hl_gauss.distribution_to_score(distributions)
    assert torch.allclose(scores, reconstructed_scores, atol=1e-2), f"Expected {scores}, got {reconstructed_scores}"
    print("Test 3: Distribution to score - Passed")

    # Test 4: Forward pass
    forward_distributions = hl_gauss(scores)
    assert torch.allclose(distributions, forward_distributions), "Forward pass doesn't match score_to_distribution"
    print("Test 4: Forward pass - Passed")

    # Test 5: Loss computation
    logits = torch.randn(3, 21)
    loss = hl_gauss.loss(logits, scores)
    assert loss.ndim == 0 and loss >= 0, f"Expected scalar non-negative loss, got {loss}"
    print("Test 5: Loss computation - Passed")

    # Test 6: Distribution peak
    peak_indices = distributions.argmax(dim=-1)
    expected_peaks = torch.tensor([5, 10, 15])  # Assuming 21 buckets from -1 to 1
    assert torch.all(peak_indices == expected_peaks), f"Expected peaks at {expected_peaks}, got {peak_indices}"
    print("Test 6: Distribution peak - Passed")

    # Test 7: Sigma effect
    hl_gauss_small = HLGauss(min_value=-1.0, max_value=1.0, num_buckets=21, sigma=0.05)
    hl_gauss_large = HLGauss(min_value=-1.0, max_value=1.0, num_buckets=21, sigma=0.2)
    score = torch.tensor([0.0])
    dist_small = hl_gauss_small.score_to_distribution(score)
    dist_large = hl_gauss_large.score_to_distribution(score)
    assert dist_small.max() > dist_large.max(), "Smaller sigma should result in higher peak"
    print("Test 7: Sigma effect - Passed")

    # Test 8: Monotonicity
    monotonic_scores = torch.linspace(-0.9, 0.9, 10)
    monotonic_distributions = hl_gauss.score_to_distribution(monotonic_scores)
    monotonic_reconstructed = hl_gauss.distribution_to_score(monotonic_distributions)
    assert torch.all(monotonic_reconstructed.diff() > 0), "Reconstructed scores should be monotonically increasing"
    print("Test 8: Monotonicity - Passed")

    # Test 9: Edge cases
    edge_scores = torch.tensor([-1.0, 1.0])
    edge_distributions = hl_gauss.score_to_distribution(edge_scores)
    assert edge_distributions[0, 0] == edge_distributions[0].max(), (
        "Leftmost bucket should have highest probability for min value"
    )
    assert edge_distributions[1, -1] == edge_distributions[1].max(), (
        "Rightmost bucket should have highest probability for max value"
    )
    print("Test 9: Edge cases - Passed")

    print("All tests passed successfully!")


if __name__ == "__main__":
    run_tests()
