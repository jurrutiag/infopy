import warnings

import numpy as np
import pytest

from infopy.estimators import (
    CCMIEstimator,
    CDMIEntropyBasedEstimator,
    CDMIRossEstimator,
    ContinuousEntropyEstimator,
    DDMIEstimator,
    DiscreteEntropyEstimator,
    MixedMIEstimator,
)


class TestPropertyBased:
    """Property-based tests for mutual information estimators."""

    def test_mi_symmetry_property(self):
        """Test MI symmetry: MI(X,Y) = MI(Y,X) across different estimators."""
        np.random.seed(42)

        test_cases = [
            # Continuous-Continuous
            (CCMIEstimator(), np.random.normal(0, 1, (100, 2)), np.random.normal(0, 1, (100, 1))),
            # Discrete-Discrete
            (DDMIEstimator(), np.random.randint(0, 3, (100, 1)), np.random.randint(0, 2, (100, 1))),
        ]

        for estimator, X, y in test_cases:
            mi_xy = estimator.estimate(X, y)
            mi_yx = estimator.estimate(y, X)

            # Allow some tolerance for estimation errors
            tolerance = 0.15
            assert abs(mi_xy - mi_yx) < tolerance, (
                f"Symmetry violated: MI(X,Y)={mi_xy:.3f}, MI(Y,X)={mi_yx:.3f}"
            )

    def test_mi_non_negativity_property(self):
        """Test that MI is always non-negative."""
        np.random.seed(42)

        estimators_and_data = [
            (DDMIEstimator(), np.random.randint(0, 4, (150, 1)), np.random.randint(0, 3, (150, 1))),
            (CCMIEstimator(), np.random.normal(0, 1, (150, 2)), np.random.normal(0, 1, (150, 1))),
            (
                CDMIRossEstimator(),
                np.random.normal(0, 1, (150, 2)),
                np.random.randint(0, 3, (150, 1)),
            ),
            (
                CDMIEntropyBasedEstimator(),
                np.random.normal(0, 1, (150, 2)),
                np.random.randint(0, 3, (150, 1)),
            ),
            (
                MixedMIEstimator(),
                np.random.normal(0, 1, (150, 2)),
                np.random.normal(0, 1, (150, 1)),
            ),
        ]

        for estimator, X, y in estimators_and_data:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress warnings for this test
                result = estimator.estimate(X, y)
                assert result >= 0, (
                    f"MI should be non-negative, got {result} for {type(estimator).__name__}"
                )

    def test_mi_scale_invariance_continuous(self):
        """Test MI invariance under scaling for continuous variables."""
        np.random.seed(42)

        # Original data
        X = np.random.normal(0, 1, (200, 2))
        y = X[:, 0].reshape(-1, 1) + 0.1 * np.random.normal(0, 1, (200, 1))

        # Scaled data
        scale_factor = 5.0
        X_scaled = X * scale_factor
        y_scaled = y * scale_factor

        estimator = CCMIEstimator(n_neighbors=5)
        mi_original = estimator.estimate(X, y)
        mi_scaled = estimator.estimate(X_scaled, y_scaled)

        # MI should be approximately invariant under scaling
        tolerance = 0.3
        assert abs(mi_original - mi_scaled) < tolerance, (
            f"MI not scale-invariant: {mi_original:.3f} vs {mi_scaled:.3f}"
        )

    def test_mi_translation_invariance_continuous(self):
        """Test MI invariance under translation for continuous variables."""
        np.random.seed(42)

        # Original data
        X = np.random.normal(0, 1, (200, 2))
        y = X[:, 0].reshape(-1, 1) + 0.1 * np.random.normal(0, 1, (200, 1))

        # Translated data
        translation = 10.0
        X_translated = X + translation
        y_translated = y + translation

        estimator = CCMIEstimator(n_neighbors=5)
        mi_original = estimator.estimate(X, y)
        mi_translated = estimator.estimate(X_translated, y_translated)

        # MI should be approximately invariant under translation
        tolerance = 0.3
        assert abs(mi_original - mi_translated) < tolerance, (
            f"MI not translation-invariant: {mi_original:.3f} vs {mi_translated:.3f}"
        )

    def test_mi_permutation_invariance(self):
        """Test MI invariance under data permutation."""
        np.random.seed(42)

        # Original data
        X = np.random.normal(0, 1, (100, 2))
        y = np.random.randint(0, 3, (100, 1))

        # Permuted data
        perm = np.random.permutation(len(X))
        X_perm = X[perm]
        y_perm = y[perm]

        estimator = CDMIRossEstimator()
        mi_original = estimator.estimate(X, y)
        mi_permuted = estimator.estimate(X_perm, y_perm)

        # MI should be exactly the same for permuted data
        tolerance = 1e-10
        assert abs(mi_original - mi_permuted) < tolerance, (
            f"MI not permutation-invariant: {mi_original:.6f} vs {mi_permuted:.6f}"
        )

    def test_mi_data_ordering_independence(self):
        """Test that MI doesn't depend on the ordering of categories for discrete data."""
        np.random.seed(42)

        # Original discrete data
        X = np.random.randint(0, 3, (200, 1))
        y = np.random.randint(0, 2, (200, 1))

        # Relabel categories
        X_relabeled = X.copy()
        X_relabeled[X == 0] = 10
        X_relabeled[X == 1] = 11
        X_relabeled[X == 2] = 12  # noqa: PLR2004

        y_relabeled = y.copy()
        y_relabeled[y == 0] = 20
        y_relabeled[y == 1] = 21

        estimator = DDMIEstimator()
        mi_original = estimator.estimate(X, y)
        mi_relabeled = estimator.estimate(X_relabeled, y_relabeled)

        # MI should be the same regardless of label values
        tolerance = 1e-10
        assert abs(mi_original - mi_relabeled) < tolerance, (
            f"MI depends on label values: {mi_original:.6f} vs {mi_relabeled:.6f}"
        )

    def test_entropy_non_negativity(self):
        """Test that entropy estimates are non-negative."""
        np.random.seed(42)

        # Test discrete entropy
        discrete_estimator = DiscreteEntropyEstimator()
        X_discrete = np.random.randint(0, 5, (100, 1))
        entropy_discrete = discrete_estimator.estimate(X_discrete)
        assert entropy_discrete >= 0, (
            f"Discrete entropy should be non-negative, got {entropy_discrete}"
        )

        # Test continuous entropy
        continuous_estimator = ContinuousEntropyEstimator()
        X_continuous = np.random.normal(0, 1, (100, 2))
        entropy_continuous = continuous_estimator.estimate(X_continuous)
        assert entropy_continuous >= 0, (
            f"Continuous entropy should be non-negative, got {entropy_continuous}"
        )

    def test_pointwise_mi_consistency(self):
        """Test that average pointwise MI equals total MI."""
        np.random.seed(42)

        test_cases = [
            (DDMIEstimator(), np.random.randint(0, 3, (100, 1)), np.random.randint(0, 2, (100, 1))),
            (CCMIEstimator(), np.random.normal(0, 1, (100, 2)), np.random.normal(0, 1, (100, 1))),
            (
                CDMIEntropyBasedEstimator(),
                np.random.normal(0, 1, (100, 2)),
                np.random.randint(0, 3, (100, 1)),
            ),
            (
                MixedMIEstimator(),
                np.random.normal(0, 1, (100, 2)),
                np.random.normal(0, 1, (100, 1)),
            ),
        ]

        for estimator, X, y in test_cases:
            pointwise_mi = estimator.estimate(X, y, pointwise=True)
            total_mi = estimator.estimate(X, y, pointwise=False)
            avg_pointwise = np.mean(pointwise_mi)

            # For continuous estimators, account for the clamping behavior
            # where total MI is clamped to non-negative but pointwise MI can be negative
            if isinstance(
                estimator,
                (CCMIEstimator, CDMIRossEstimator, CDMIEntropyBasedEstimator, MixedMIEstimator),
            ):
                expected_total = max(0.0, avg_pointwise)
                tolerance = 1e-10
                assert abs(expected_total - total_mi) < tolerance, (
                    f"Expected total MI {expected_total:.6f} != actual total MI {total_mi:.6f} "
                    f"for {type(estimator).__name__}"
                )
            else:
                # For discrete estimators, exact equality should hold
                tolerance = 1e-10
                assert abs(avg_pointwise - total_mi) < tolerance, (
                    f"Average pointwise MI {avg_pointwise:.6f} != total MI {total_mi:.6f} "
                    f"for {type(estimator).__name__}"
                )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_data_handling(self):
        """Test behavior with empty datasets."""
        estimators = [
            DDMIEstimator(),
            CCMIEstimator(),
            CDMIRossEstimator(),
            CDMIEntropyBasedEstimator(),
            MixedMIEstimator(),
        ]

        for estimator in estimators:
            X_empty = np.array([]).reshape(0, 1)
            y_empty = np.array([]).reshape(0, 1)

            with pytest.raises((ValueError, IndexError)):
                estimator.estimate(X_empty, y_empty)

    def test_single_sample(self):
        """Test behavior with single sample."""
        estimators_and_data = [
            (DDMIEstimator(), np.array([[1]]), np.array([[0]])),
            (CCMIEstimator(), np.array([[1.0, 2.0]]), np.array([[0.5]])),
        ]

        for estimator, X, y in estimators_and_data:
            # Should either work or raise appropriate error
            try:
                result = estimator.estimate(X, y)
                assert np.isfinite(result), f"Single sample result should be finite, got {result}"
            except (ValueError, IndexError):
                # It's acceptable to raise an error for single sample
                pass

    def test_identical_samples(self):
        """Test behavior when all samples are identical."""
        # Discrete case
        X_discrete = np.ones((10, 1), dtype=int)
        y_discrete = np.ones((10, 1), dtype=int)

        estimator_discrete = DDMIEstimator()
        result_discrete = estimator_discrete.estimate(X_discrete, y_discrete)
        assert np.isfinite(result_discrete), (
            "Result should be finite for identical discrete samples"
        )

        # Continuous case (with small noise to avoid numerical issues)
        X_continuous = np.ones((10, 2)) + 1e-10 * np.random.randn(10, 2)
        y_continuous = np.ones((10, 1)) + 1e-10 * np.random.randn(10, 1)

        estimator_continuous = CCMIEstimator()
        result_continuous = estimator_continuous.estimate(X_continuous, y_continuous)
        assert np.isfinite(result_continuous), (
            "Result should be finite for nearly identical continuous samples"
        )

    def test_very_small_datasets(self):
        """Test behavior with very small datasets."""
        np.random.seed(42)

        small_sizes = [2, 3, 5]

        for n in small_sizes:
            # Discrete case
            X_discrete = np.random.randint(0, 2, (n, 1))
            y_discrete = np.random.randint(0, 2, (n, 1))

            estimator_discrete = DDMIEstimator()
            try:
                result_discrete = estimator_discrete.estimate(X_discrete, y_discrete)
                assert np.isfinite(result_discrete), (
                    f"Result should be finite for small discrete dataset (n={n})"
                )
                assert result_discrete >= 0, f"MI should be non-negative for small dataset (n={n})"
            except (ValueError, IndexError):
                # Acceptable to fail for very small datasets
                pass

            # Continuous case
            X_continuous = np.random.normal(0, 1, (n, 2))
            y_continuous = np.random.normal(0, 1, (n, 1))

            estimator_continuous = CCMIEstimator()
            try:
                result_continuous = estimator_continuous.estimate(X_continuous, y_continuous)
                assert np.isfinite(result_continuous), (
                    f"Result should be finite for small continuous dataset (n={n})"
                )
                assert result_continuous >= 0, (
                    f"MI should be non-negative for small dataset (n={n})"
                )
            except (ValueError, IndexError):
                # Acceptable to fail for very small datasets
                pass

    def test_high_dimensional_data(self):
        """Test behavior with high-dimensional data."""
        np.random.seed(42)
        n_samples = 200
        high_dims = [10, 20]

        for dim in high_dims:
            X = np.random.normal(0, 1, (n_samples, dim))
            y = np.random.normal(0, 1, (n_samples, 1))

            estimator = CCMIEstimator()
            result = estimator.estimate(X, y)

            assert np.isfinite(result), (
                f"Result should be finite for high-dimensional data (dim={dim})"
            )
            assert result >= 0, f"MI should be non-negative for high-dimensional data (dim={dim})"

    def test_extreme_parameter_values(self):
        """Test estimators with extreme parameter values."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 2))
        y = np.random.randint(0, 3, (100, 1))

        # Test with very small n_neighbors
        estimator_small_k = CDMIRossEstimator(n_neighbors=1)
        result_small_k = estimator_small_k.estimate(X, y)
        assert np.isfinite(result_small_k) and result_small_k >= 0

        # Test with large n_neighbors (should be capped internally)
        estimator_large_k = CDMIRossEstimator(n_neighbors=1000)
        result_large_k = estimator_large_k.estimate(X, y)
        assert np.isfinite(result_large_k) and result_large_k >= 0

    def test_mixed_data_types(self):
        """Test handling of mixed data types in discrete estimators."""
        # String labels
        X_str = np.array([["a"], ["b"], ["a"], ["c"], ["b"]])
        y_str = np.array([["x"], ["y"], ["x"], ["z"], ["y"]])

        estimator = DDMIEstimator()
        result = estimator.estimate(X_str, y_str)
        assert np.isfinite(result) and result >= 0

        # Float labels that represent discrete categories
        X_float = np.array([[1.0], [2.0], [1.0], [3.0], [2.0]])
        y_float = np.array([[10.0], [20.0], [10.0], [30.0], [20.0]])

        result_float = estimator.estimate(X_float, y_float)
        assert np.isfinite(result_float) and result_float >= 0

    def test_nan_inf_handling(self):
        """Test behavior when data contains NaN or Inf values."""
        np.random.seed(42)

        # Data with NaN
        X_nan = np.random.normal(0, 1, (50, 2))
        X_nan[0, 0] = np.nan
        y_nan = np.random.normal(0, 1, (50, 1))

        estimator = CCMIEstimator()

        # Should either handle gracefully or raise appropriate error
        try:
            result = estimator.estimate(X_nan, y_nan)
            # If it doesn't raise an error, result should still be valid
            assert np.isfinite(result), (
                "Result with NaN data should be finite if processing succeeds"
            )
        except (ValueError, RuntimeError):
            # It's acceptable to raise an error for NaN data
            pass

        # Data with Inf
        X_inf = np.random.normal(0, 1, (50, 2))
        X_inf[0, 0] = np.inf
        y_inf = np.random.normal(0, 1, (50, 1))

        try:
            result = estimator.estimate(X_inf, y_inf)
            assert np.isfinite(result), (
                "Result with Inf data should be finite if processing succeeds"
            )
        except (ValueError, RuntimeError):
            # It's acceptable to raise an error for Inf data
            pass

    def test_mismatched_sample_sizes(self):
        """Test error handling for mismatched sample sizes."""
        estimator = DDMIEstimator()

        X = np.random.randint(0, 3, (100, 1))
        y = np.random.randint(0, 2, (50, 1))  # Different size

        with pytest.raises((ValueError, IndexError)):
            estimator.estimate(X, y)

    def test_wrong_dimensional_input(self):
        """Test error handling for wrong dimensional inputs."""
        estimator = DDMIEstimator()

        # 3D input (should be 2D)
        X_3d = np.random.randint(0, 3, (50, 2, 3))
        y_2d = np.random.randint(0, 2, (50, 1))

        with pytest.raises((ValueError, IndexError)):
            estimator.estimate(X_3d, y_2d)
