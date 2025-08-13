import numpy as np
import pytest
from scipy.stats import entropy as scipy_entropy

from infopy.functional import discrete_entropy, kozachenko_leonenko_entropy


class TestDiscreteEntropy:
    """Test suite for discrete_entropy function."""

    def test_discrete_entropy_uniform(self):
        """Test entropy of uniform discrete distribution."""
        # Uniform distribution over 4 values
        X = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        expected = np.log2(4)  # log2(n) for uniform distribution
        result = discrete_entropy(X)
        assert np.isclose(result, expected, rtol=1e-10)

    def test_discrete_entropy_deterministic(self):
        """Test entropy of deterministic (single value) distribution."""
        X = np.array([1, 1, 1, 1, 1])
        result = discrete_entropy(X)
        assert np.isclose(result, 0.0, atol=1e-10)

    def test_discrete_entropy_binary(self):
        """Test entropy of binary distribution."""
        # Equal probability binary
        X = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        expected = 1.0  # log2(2) for uniform binary
        result = discrete_entropy(X)
        assert np.isclose(result, expected, rtol=1e-10)

        # Biased binary (0.25, 0.75)
        X = np.array([0, 1, 1, 1])
        p = np.array([0.25, 0.75])
        expected = -np.sum(p * np.log2(p))
        result = discrete_entropy(X)
        assert np.isclose(result, expected, rtol=1e-10)

    def test_discrete_entropy_pointwise(self):
        """Test pointwise entropy calculation."""
        X = np.array([0, 1, 2, 0, 1, 2])
        pointwise_result = discrete_entropy(X, pointwise=True)

        # Should have same length as input
        assert len(pointwise_result) == len(X)

        # All values should be positive (information content)
        assert np.all(pointwise_result >= 0)

        # Average pointwise should equal total entropy
        total_entropy = discrete_entropy(X, pointwise=False)
        avg_pointwise = np.mean(pointwise_result)
        assert np.isclose(avg_pointwise, total_entropy, rtol=1e-10)

    def test_discrete_entropy_comparison_with_scipy(self):
        """Test against scipy.stats.entropy for validation."""
        X = np.array([0, 1, 2, 0, 1, 1, 2, 2, 2])

        # Calculate using our function
        result = discrete_entropy(X)

        # Calculate using scipy
        _, counts = np.unique(X, return_counts=True)
        prob = counts / len(X)
        expected = scipy_entropy(prob, base=2)

        assert np.isclose(result, expected, rtol=1e-10)

    def test_discrete_entropy_single_element(self):
        """Test entropy calculation with single element."""
        X = np.array([5])
        result = discrete_entropy(X)
        assert np.isclose(result, 0.0, atol=1e-10)

    def test_discrete_entropy_string_labels(self):
        """Test entropy with string/object labels."""
        X = np.array(["a", "b", "a", "b", "c", "c"])
        result = discrete_entropy(X)

        # Manual calculation
        counts = np.array([2, 2, 2])  # a:2, b:2, c:2
        prob = counts / 6
        expected = -np.sum(prob * np.log2(prob))

        assert np.isclose(result, expected, rtol=1e-10)

    def test_discrete_entropy_empty_array(self):
        """Test behavior with empty array."""
        X = np.array([])
        with pytest.raises((ValueError, IndexError)):
            discrete_entropy(X)


class TestKozachenkoLeonenkoEntropy:
    """Test suite for kozachenko_leonenko_entropy function."""

    def test_kl_entropy_basic_functionality(self):
        """Test basic functionality of K-L entropy estimator."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 2))

        result = kozachenko_leonenko_entropy(X)

        # Should return a positive value for continuous data
        assert result > 0
        assert isinstance(result, float)

    def test_kl_entropy_pointwise(self):
        """Test pointwise entropy calculation."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (50, 2))

        pointwise_result = kozachenko_leonenko_entropy(X, pointwise=True)
        total_result = kozachenko_leonenko_entropy(X, pointwise=False)

        # Should have same length as input
        assert len(pointwise_result) == len(X)

        # Average pointwise should equal total entropy
        assert np.isclose(np.mean(pointwise_result), total_result, rtol=1e-10)

    def test_kl_entropy_different_metrics(self):
        """Test with different distance metrics."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (30, 2))

        result_euclidean = kozachenko_leonenko_entropy(X, metric="euclidean")
        result_manhattan = kozachenko_leonenko_entropy(X, metric="manhattan")
        result_chebyshev = kozachenko_leonenko_entropy(X, metric="chebyshev")

        # All should be positive and finite
        assert result_euclidean > 0 and np.isfinite(result_euclidean)
        assert result_manhattan > 0 and np.isfinite(result_manhattan)
        assert result_chebyshev > 0 and np.isfinite(result_chebyshev)

    def test_kl_entropy_different_neighbors(self):
        """Test with different number of neighbors."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 2))

        result_k3 = kozachenko_leonenko_entropy(X, n_neighbors=3)
        result_k5 = kozachenko_leonenko_entropy(X, n_neighbors=5)
        result_k10 = kozachenko_leonenko_entropy(X, n_neighbors=10)

        # All should be positive
        assert result_k3 > 0
        assert result_k5 > 0
        assert result_k10 > 0

    def test_kl_entropy_1d_data(self):
        """Test with 1D data."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 1))

        result = kozachenko_leonenko_entropy(X)
        assert result > 0
        assert isinstance(result, float)

    def test_kl_entropy_high_dimensional(self):
        """Test with higher dimensional data."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (50, 5))

        result = kozachenko_leonenko_entropy(X)
        assert result > 0
        assert isinstance(result, float)

    def test_kl_entropy_identical_points(self):
        """Test behavior with identical points."""
        X = np.ones((10, 2))

        # Should handle identical points gracefully
        result = kozachenko_leonenko_entropy(X)
        # Entropy of constant distribution should be low
        assert result >= 0

    def test_kl_entropy_small_sample(self):
        """Test with very small sample size."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (5, 2))

        result = kozachenko_leonenko_entropy(X)
        assert result >= 0
        assert np.isfinite(result)

    def test_kl_entropy_parameter_bounds(self):
        """Test parameter boundary conditions."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (20, 2))

        # Test with n_neighbors at boundary (should be reduced internally)
        result = kozachenko_leonenko_entropy(X, n_neighbors=50)  # > n_samples
        assert result >= 0
        assert np.isfinite(result)

    def test_kl_entropy_deterministic_sequence(self):
        """Test with deterministic but varying sequence."""
        # Create a deterministic but non-constant sequence
        X = np.array([[i, i**2] for i in range(20)], dtype=float)

        result = kozachenko_leonenko_entropy(X)
        assert result > 0  # Should have positive entropy
        assert np.isfinite(result)

    def test_kl_entropy_non_negativity(self):
        """Test that entropy estimates are non-negative."""
        np.random.seed(42)
        for _ in range(10):  # Multiple random trials
            X = np.random.normal(0, 1, (50, 2))
            result = kozachenko_leonenko_entropy(X)
            assert result >= 0, f"Entropy should be non-negative, got {result}"
