import queue
import threading
import time

import numpy as np
import pytest

from infopy.estimators import (
    CCMIEstimator,
    CDMIEntropyBasedEstimator,
    CDMIRossEstimator,
    ContinuousEntropyEstimator,
    DDMIEstimator,
    DiscreteEntropyEstimator,
)

# Performance test constants
MAX_TIME_CONTINUOUS = 60
MAX_TIME_DISCRETE = 10
MAX_TIME_HIGH_DIM = 30
MAX_TIME_NEIGHBORS = 20
MAX_TIME_DISCRETE_ENTROPY = 5
MAX_TIME_CONTINUOUS_ENTROPY = 20
MAX_GROWTH_FACTOR = 10
MAX_NEIGHBOR_GROWTH = 5
MAX_THREAD_TIME_RATIO = 10
CONSISTENCY_TOLERANCE = 1e-10
THREAD_TOLERANCE = 0.1
EXPECTED_THREAD_COUNT = 3


class TestPerformance:
    """Performance and stress tests for infopy estimators."""

    @pytest.mark.slow
    def test_large_dataset_performance_continuous(self):
        """Test performance with large continuous datasets."""
        np.random.seed(42)
        large_sizes = [1000, 5000]

        for n_samples in large_sizes:
            print(f"\nTesting continuous MI with {n_samples} samples...")

            X = np.random.normal(0, 1, (n_samples, 3))
            y = np.random.normal(0, 1, (n_samples, 1))

            estimator = CCMIEstimator(n_neighbors=5)

            start_time = time.time()
            result = estimator.estimate(X, y)
            end_time = time.time()

            elapsed = end_time - start_time
            print(f"  Time: {elapsed:.2f}s, Result: {result:.3f}")

            # Basic sanity checks
            assert np.isfinite(result), f"Result should be finite for large dataset (n={n_samples})"
            assert result >= 0, f"MI should be non-negative for large dataset (n={n_samples})"

            # Performance check: should complete within reasonable time
            assert elapsed < MAX_TIME_CONTINUOUS, (
                f"Estimation took too long: {elapsed:.2f}s for {n_samples} samples"
            )

    @pytest.mark.slow
    def test_large_dataset_performance_discrete(self):
        """Test performance with large discrete datasets."""
        np.random.seed(42)
        large_sizes = [1000, 10000]

        for n_samples in large_sizes:
            print(f"\nTesting discrete MI with {n_samples} samples...")

            X = np.random.randint(0, 10, (n_samples, 1))
            y = np.random.randint(0, 5, (n_samples, 1))

            estimator = DDMIEstimator()

            start_time = time.time()
            result = estimator.estimate(X, y)
            end_time = time.time()

            elapsed = end_time - start_time
            print(f"  Time: {elapsed:.2f}s, Result: {result:.3f}")

            # Basic sanity checks
            assert np.isfinite(result), f"Result should be finite for large dataset (n={n_samples})"
            assert result >= 0, f"MI should be non-negative for large dataset (n={n_samples})"

            # Discrete MI should be very fast
            assert elapsed < MAX_TIME_DISCRETE, (
                f"Discrete estimation took too long: {elapsed:.2f}s for {n_samples} samples"
            )

    @pytest.mark.slow
    def test_large_dataset_performance_mixed(self):
        """Test performance with large mixed-type datasets."""
        np.random.seed(42)
        large_sizes = [1000, 2000]

        for n_samples in large_sizes:
            print(f"\nTesting continuous-discrete MI with {n_samples} samples...")

            X = np.random.normal(0, 1, (n_samples, 3))
            y = np.random.randint(0, 4, (n_samples, 1))

            # Test Ross estimator
            estimator_ross = CDMIRossEstimator(n_neighbors=5)
            start_time = time.time()
            result_ross = estimator_ross.estimate(X, y)
            elapsed_ross = time.time() - start_time

            print(f"  Ross estimator - Time: {elapsed_ross:.2f}s, Result: {result_ross:.3f}")

            # Test entropy-based estimator
            estimator_entropy = CDMIEntropyBasedEstimator(n_neighbors=5)
            start_time = time.time()
            result_entropy = estimator_entropy.estimate(X, y)
            elapsed_entropy = time.time() - start_time

            print(
                f"  Entropy estimator - Time: {elapsed_entropy:.2f}s, Result: {result_entropy:.3f}"
            )

            # Basic sanity checks
            assert np.isfinite(result_ross) and result_ross >= 0
            assert np.isfinite(result_entropy) and result_entropy >= 0

            # Performance checks
            assert elapsed_ross < MAX_TIME_CONTINUOUS, (
                f"Ross estimation took too long: {elapsed_ross:.2f}s"
            )
            assert elapsed_entropy < MAX_TIME_CONTINUOUS, (
                f"Entropy estimation took too long: {elapsed_entropy:.2f}s"
            )

    def test_high_dimensional_scaling(self):
        """Test performance scaling with dimensionality."""
        np.random.seed(42)
        n_samples = 500
        dimensions = [2, 5, 10, 20]

        times = []
        results = []

        for dim in dimensions:
            print(f"\nTesting with {dim} dimensions...")

            X = np.random.normal(0, 1, (n_samples, dim))
            y = np.random.normal(0, 1, (n_samples, 1))

            estimator = CCMIEstimator(n_neighbors=4)

            start_time = time.time()
            result = estimator.estimate(X, y)
            elapsed = time.time() - start_time

            times.append(elapsed)
            results.append(result)

            print(f"  Time: {elapsed:.3f}s, Result: {result:.3f}")

            assert np.isfinite(result) and result >= 0
            assert elapsed < MAX_TIME_HIGH_DIM, (
                f"High-dimensional estimation took too long: {elapsed:.2f}s for dim={dim}"
            )

        # Time should increase with dimensionality, but not exponentially
        for i in range(1, len(times)):
            growth_factor = times[i] / times[i - 1]
            assert growth_factor < MAX_GROWTH_FACTOR, (
                f"Time growth too large from dim {dimensions[i - 1]} to "
                f"{dimensions[i]}: {growth_factor:.2f}x"
            )

    def test_neighbor_parameter_scaling(self):
        """Test how performance scales with number of neighbors."""
        np.random.seed(42)
        n_samples = 1000
        X = np.random.normal(0, 1, (n_samples, 3))
        y = np.random.normal(0, 1, (n_samples, 1))

        neighbor_counts = [3, 5, 10, 20]
        times = []

        for k in neighbor_counts:
            estimator = CCMIEstimator(n_neighbors=k)

            start_time = time.time()
            result = estimator.estimate(X, y)
            elapsed = time.time() - start_time

            times.append(elapsed)

            print(f"k={k}: Time={elapsed:.3f}s, Result={result:.3f}")

            assert np.isfinite(result) and result >= 0
            assert elapsed < MAX_TIME_NEIGHBORS, (
                f"Estimation with k={k} took too long: {elapsed:.2f}s"
            )

        # Time should not increase dramatically with k
        max_time_ratio = max(times) / min(times)
        assert max_time_ratio < MAX_NEIGHBOR_GROWTH, (
            f"Time scaling with neighbors too large: {max_time_ratio:.2f}x"
        )

    @pytest.mark.slow
    def test_pointwise_mi_performance(self):
        """Test performance of pointwise MI calculation."""
        np.random.seed(42)
        n_samples = 2000

        X = np.random.normal(0, 1, (n_samples, 2))
        y = np.random.randint(0, 3, (n_samples, 1))

        estimator = CDMIEntropyBasedEstimator(n_neighbors=5)

        # Test total MI
        start_time = time.time()
        total_mi = estimator.estimate(X, y, pointwise=False)
        total_time = time.time() - start_time

        # Test pointwise MI
        start_time = time.time()
        pointwise_mi = estimator.estimate(X, y, pointwise=True)
        pointwise_time = time.time() - start_time

        print(f"Total MI: {total_mi:.3f} (time: {total_time:.3f}s)")
        print(f"Pointwise MI: mean={np.mean(pointwise_mi):.3f} (time: {pointwise_time:.3f}s)")

        # Sanity checks
        assert np.isfinite(total_mi) and total_mi >= 0
        assert len(pointwise_mi) == n_samples
        assert np.all(np.isfinite(pointwise_mi))

        # Consistency check
        assert abs(np.mean(pointwise_mi) - total_mi) < CONSISTENCY_TOLERANCE

        # Performance check - pointwise should not be dramatically slower
        time_ratio = pointwise_time / total_time
        assert time_ratio < MAX_THREAD_TIME_RATIO, (
            f"Pointwise calculation too slow: {time_ratio:.2f}x slower"
        )

    def test_memory_usage_large_dataset(self):
        """Test that large datasets don't cause memory issues."""
        np.random.seed(42)
        n_samples = 5000

        # Test with continuous data
        X = np.random.normal(0, 1, (n_samples, 4))
        y = np.random.normal(0, 1, (n_samples, 1))

        estimator = CCMIEstimator(n_neighbors=5)

        try:
            result = estimator.estimate(X, y)
            assert np.isfinite(result) and result >= 0
        except MemoryError:
            pytest.fail("Memory error with large continuous dataset")

        # Test with discrete data
        X_discrete = np.random.randint(0, 20, (n_samples, 1))
        y_discrete = np.random.randint(0, 10, (n_samples, 1))

        estimator_discrete = DDMIEstimator()

        try:
            result_discrete = estimator_discrete.estimate(X_discrete, y_discrete)
            assert np.isfinite(result_discrete) and result_discrete >= 0
        except MemoryError:
            pytest.fail("Memory error with large discrete dataset")

    def test_entropy_estimator_performance(self):
        """Test performance of entropy estimators."""
        np.random.seed(42)
        n_samples = 3000

        # Test discrete entropy
        X_discrete = np.random.randint(0, 10, (n_samples, 1))
        discrete_estimator = DiscreteEntropyEstimator()

        start_time = time.time()
        entropy_discrete = discrete_estimator.estimate(X_discrete)
        discrete_time = time.time() - start_time

        # Test continuous entropy
        X_continuous = np.random.normal(0, 1, (n_samples, 3))
        continuous_estimator = ContinuousEntropyEstimator(n_neighbors=5)

        start_time = time.time()
        entropy_continuous = continuous_estimator.estimate(X_continuous)
        continuous_time = time.time() - start_time

        print(f"Discrete entropy: {entropy_discrete:.3f} (time: {discrete_time:.3f}s)")
        print(f"Continuous entropy: {entropy_continuous:.3f} (time: {continuous_time:.3f}s)")

        # Sanity checks
        assert np.isfinite(entropy_discrete) and entropy_discrete >= 0
        assert np.isfinite(entropy_continuous) and entropy_continuous >= 0

        # Performance checks
        assert discrete_time < MAX_TIME_DISCRETE_ENTROPY, (
            f"Discrete entropy too slow: {discrete_time:.2f}s"
        )
        assert continuous_time < MAX_TIME_CONTINUOUS_ENTROPY, (
            f"Continuous entropy too slow: {continuous_time:.2f}s"
        )

    def test_batch_estimation_consistency(self):
        """Test that results are consistent across multiple estimations."""
        np.random.seed(42)
        n_samples = 500

        X = np.random.normal(0, 1, (n_samples, 2))
        y = np.random.randint(0, 3, (n_samples, 1))

        estimator = CDMIRossEstimator(n_neighbors=5)

        # Run multiple estimations
        results = []
        for _ in range(5):
            result = estimator.estimate(X, y)
            results.append(result)

        # Results should be identical (deterministic algorithm)
        for i in range(1, len(results)):
            assert abs(results[i] - results[0]) < CONSISTENCY_TOLERANCE, (
                f"Results not consistent: {results[0]:.6f} vs {results[i]:.6f}"
            )

    @pytest.mark.slow
    def test_stress_test_random_data(self):
        """Stress test with various random data configurations."""
        np.random.seed(42)

        configs = [
            # (n_samples, x_dim, y_dim, x_type, y_type)
            (2000, 1, 1, "discrete", "discrete"),
            (1500, 3, 1, "continuous", "continuous"),
            (1000, 4, 1, "continuous", "discrete"),
            (800, 2, 1, "discrete", "continuous"),
        ]

        for n_samples, x_dim, y_dim, x_type, y_type in configs:
            print(f"\nStress test: {n_samples} samples, X:{x_dim}D {x_type}, Y:{y_dim}D {y_type}")

            # Generate data based on type
            if x_type == "discrete":
                X = np.random.randint(0, 5, (n_samples, x_dim))
            else:
                X = np.random.normal(0, 1, (n_samples, x_dim))

            if y_type == "discrete":
                y = np.random.randint(0, 3, (n_samples, y_dim))
            else:
                y = np.random.normal(0, 1, (n_samples, y_dim))

            # Select appropriate estimator
            if x_type == "discrete" and y_type == "discrete":
                estimator = DDMIEstimator()
            elif x_type == "continuous" and y_type == "continuous":
                estimator = CCMIEstimator(n_neighbors=5)
            else:
                estimator = CDMIRossEstimator(n_neighbors=5)

            start_time = time.time()
            try:
                result = estimator.estimate(X, y)
                elapsed = time.time() - start_time

                print(f"  Result: {result:.3f}, Time: {elapsed:.3f}s")

                # Basic checks
                assert np.isfinite(result), f"Non-finite result: {result}"
                assert result >= 0, f"Negative MI: {result}"
                assert elapsed < MAX_TIME_HIGH_DIM, f"Too slow: {elapsed:.2f}s"

            except Exception as e:
                pytest.fail(f"Failed on config {configs}: {e}")

    def test_concurrent_estimation_safety(self):
        """Test that estimators are safe for concurrent use."""

        np.random.seed(42)
        n_samples = 500
        X = np.random.normal(0, 1, (n_samples, 2))
        y = np.random.normal(0, 1, (n_samples, 1))

        results = queue.Queue()

        def estimate_mi():
            estimator = CCMIEstimator(n_neighbors=5)
            result = estimator.estimate(X, y)
            results.put(result)

        # Run multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=estimate_mi)
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check results
        thread_results = []
        while not results.empty():
            thread_results.append(results.get())

        assert len(thread_results) == EXPECTED_THREAD_COUNT
        for result in thread_results:
            assert np.isfinite(result) and result >= 0

        # Results should be similar (using same data and random seed)
        mean_result = np.mean(thread_results)
        for result in thread_results:
            assert abs(result - mean_result) < THREAD_TOLERANCE, (
                f"Thread results too different: {thread_results}"
            )
