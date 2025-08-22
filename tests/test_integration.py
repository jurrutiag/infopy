import numpy as np

from infopy.estimators import (
    CCMIEstimator,
    CDMIEntropyBasedEstimator,
    CDMIRossEstimator,
    DDMIEstimator,
    get_mi_estimator,
)

# Test constants
INDEPENDENCE_TOLERANCE = 0.1
SMALL_TOLERANCE = 0.1
MEDIUM_TOLERANCE = 0.2
LARGE_TOLERANCE = 0.4
EXTREME_TOLERANCE = 0.5


class TestKnownMIValues:
    """Integration tests using synthetic data with known theoretical MI values."""

    def test_discrete_uniform_independence(self):
        """Test MI=0 for independent uniform discrete variables."""
        np.random.seed(42)
        n_samples = 1000

        # Create independent uniform discrete variables
        X = np.random.randint(0, 4, (n_samples, 1))
        y = np.random.randint(0, 3, (n_samples, 1))

        estimator = DDMIEstimator()
        result = estimator.estimate(X, y)

        # MI should be close to 0 for independent variables
        assert abs(result) < INDEPENDENCE_TOLERANCE, f"Expected MI ≈ 0, got {result}"

    def test_discrete_deterministic_relationship(self):
        """Test MI for deterministic discrete relationship."""
        # Create deterministic relationship: y = X % 3
        X = np.arange(300).reshape(-1, 1)
        y = (X % 3).reshape(-1, 1)

        estimator = DDMIEstimator()
        result = estimator.estimate(X, y)

        # For deterministic relationship, MI = H(Y)
        # H(Y) = log2(3) ≈ 1.585 for uniform distribution over 3 values
        expected_mi = np.log2(3)
        assert abs(result - expected_mi) < SMALL_TOLERANCE, (
            f"Expected MI ≈ {expected_mi}, got {result}"
        )

    def test_discrete_identical_variables(self):
        """Test MI for identical discrete variables."""
        np.random.seed(42)
        X = np.random.randint(0, 4, (500, 1))
        y = X.copy()

        estimator = DDMIEstimator()
        result = estimator.estimate(X, y)

        # For identical variables, MI = H(X) = H(Y)
        # Expected entropy for uniform distribution over 4 values
        expected_mi = np.log2(4)  # = 2.0
        assert abs(result - expected_mi) < MEDIUM_TOLERANCE, (
            f"Expected MI ≈ {expected_mi}, got {result}"
        )

    def test_bivariate_gaussian_known_mi(self):
        """Test MI for bivariate Gaussian with known correlation."""
        np.random.seed(42)
        n_samples = 1000

        # Create correlated bivariate Gaussian
        rho = 0.8  # correlation coefficient
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]

        # Generate samples
        samples = np.random.multivariate_normal(mean, cov, n_samples)
        X = samples[:, 0].reshape(-1, 1)
        y = samples[:, 1].reshape(-1, 1)

        # Theoretical MI for bivariate Gaussian: -0.5 * log(1 - rho^2)
        theoretical_mi = -0.5 * np.log(1 - rho**2) / np.log(2)  # Convert to bits

        estimator = CCMIEstimator(n_neighbors=5)
        estimated_mi = estimator.estimate(X, y)

        # Allow some tolerance due to finite sample effects
        tolerance = 0.3
        assert abs(estimated_mi - theoretical_mi) < tolerance, (
            f"Expected MI ≈ {theoretical_mi:.3f}, got {estimated_mi:.3f}"
        )

    def test_gaussian_independence(self):
        """Test MI=0 for independent Gaussian variables."""
        np.random.seed(42)
        n_samples = 1000

        X = np.random.normal(0, 1, (n_samples, 2))
        y = np.random.normal(0, 1, (n_samples, 1))

        estimator = CCMIEstimator(n_neighbors=5)
        result = estimator.estimate(X, y)

        # MI should be close to 0 for independent variables
        assert abs(result) < MEDIUM_TOLERANCE, f"Expected MI ≈ 0, got {result}"

    def test_continuous_discrete_relationship(self):
        """Test MI for continuous-discrete with known relationship."""
        np.random.seed(42)
        n_samples = 1000

        # Create continuous variable
        X = np.random.normal(0, 1, (n_samples, 1))

        # Create discrete variable based on X
        y = (X > 0).astype(int).reshape(-1, 1)  # Binary based on sign

        estimator = CDMIRossEstimator(n_neighbors=5)
        result = estimator.estimate(X, y)

        # Should have positive MI since y depends on X
        assert result > MEDIUM_TOLERANCE, f"Expected positive MI, got {result}"

    def test_mixed_gaussian_correlation_levels(self):
        """Test MI estimation across different correlation levels."""
        np.random.seed(42)
        n_samples = 800
        correlations = [0.0, 0.3, 0.6, 0.9]
        estimated_mis = []
        theoretical_mis = []

        for rho in correlations:
            # Generate correlated bivariate Gaussian
            mean = [0, 0]
            cov = [[1, rho], [rho, 1]]
            samples = np.random.multivariate_normal(mean, cov, n_samples)
            X = samples[:, 0].reshape(-1, 1)
            y = samples[:, 1].reshape(-1, 1)

            # Theoretical MI
            theoretical_mi = -0.5 * np.log(1 - rho**2) / np.log(2)
            theoretical_mis.append(theoretical_mi)

            # Estimated MI
            estimator = CCMIEstimator(n_neighbors=4)
            estimated_mi = estimator.estimate(X, y)
            estimated_mis.append(estimated_mi)

        # Check that MI increases with correlation
        for i in range(1, len(estimated_mis)):
            assert estimated_mis[i] >= estimated_mis[i - 1] - 0.1, (
                "MI should increase with correlation"
            )

        # Check that estimates are reasonably close to theoretical values
        for est, theo in zip(estimated_mis[1:], theoretical_mis[1:]):  # Skip rho=0 case
            assert abs(est - theo) < LARGE_TOLERANCE, (
                f"Estimated MI {est:.3f} too far from theoretical {theo:.3f}"
            )


class TestDataGenerationMI:
    """Tests using specific data generation patterns with known MI properties."""

    def test_xor_relationship(self):
        """Test MI for XOR relationship between discrete variables."""
        np.random.seed(42)
        n_samples = 1000

        # Generate random binary variables
        X1 = np.random.randint(0, 2, n_samples)
        X2 = np.random.randint(0, 2, n_samples)

        # Create XOR relationship
        y = (X1 ^ X2).reshape(-1, 1)
        X = np.column_stack([X1, X2])

        estimator = DDMIEstimator()
        result = estimator.estimate(X, y)

        # XOR creates strong dependence, MI should be positive
        assert result > EXTREME_TOLERANCE, f"Expected high MI for XOR relationship, got {result}"

    def test_linear_continuous_relationship(self):
        """Test MI for linear continuous relationship."""
        np.random.seed(42)
        n_samples = 800

        # Create linear relationship with noise
        X = np.random.normal(0, 1, (n_samples, 1))
        noise = np.random.normal(0, 0.1, (n_samples, 1))
        y = 2 * X + noise  # Linear relationship

        estimator = CCMIEstimator(n_neighbors=5)
        result = estimator.estimate(X, y)

        # Strong linear relationship should have high MI
        assert result > 1.0, f"Expected high MI for linear relationship, got {result}"

    def test_nonlinear_continuous_relationship(self):
        """Test MI for nonlinear continuous relationship."""
        np.random.seed(42)
        n_samples = 600

        # Create nonlinear relationship
        X = np.random.uniform(-2, 2, (n_samples, 1))
        noise = np.random.normal(0, 0.1, (n_samples, 1))
        y = np.sin(X) + noise  # Nonlinear relationship

        estimator = CCMIEstimator(n_neighbors=5)
        result = estimator.estimate(X, y)

        # Nonlinear relationship should have positive MI
        assert result > MEDIUM_TOLERANCE + SMALL_TOLERANCE, (
            f"Expected positive MI for nonlinear relationship, got {result}"
        )

    def test_multimodal_distribution(self):
        """Test MI estimation with multimodal distributions."""
        np.random.seed(42)
        n_samples = 800

        # Create multimodal distribution
        mode1 = np.random.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], n_samples // 2)
        mode2 = np.random.multivariate_normal([3, 3], [[1, 0.8], [0.8, 1]], n_samples // 2)

        samples = np.vstack([mode1, mode2])
        np.random.shuffle(samples)

        X = samples[:, 0].reshape(-1, 1)
        y = samples[:, 1].reshape(-1, 1)

        estimator = CCMIEstimator(n_neighbors=5)
        result = estimator.estimate(X, y)

        # Correlated multimodal distribution should have positive MI
        assert result > EXTREME_TOLERANCE, (
            f"Expected positive MI for multimodal distribution, got {result}"
        )


class TestEstimatorConsistency:
    """Tests to ensure different estimators give consistent results where applicable."""

    def test_continuous_discrete_estimator_consistency(self):
        """Test consistency between different continuous-discrete estimators."""
        np.random.seed(42)
        n_samples = 400

        # Create continuous-discrete relationship
        X = np.random.normal(0, 1, (n_samples, 2))
        y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)

        # Test both estimators
        ross_estimator = CDMIRossEstimator(n_neighbors=5)
        entropy_estimator = CDMIEntropyBasedEstimator(n_neighbors=5)

        ross_result = ross_estimator.estimate(X, y)
        entropy_result = entropy_estimator.estimate(X, y)

        # Results should be in the same ballpark
        assert abs(ross_result - entropy_result) < EXTREME_TOLERANCE, (
            f"Estimators gave very different results: {ross_result:.3f} vs {entropy_result:.3f}"
        )

        # Both should be positive for this dependent relationship
        assert ross_result > 0 and entropy_result > 0

    def test_estimator_selection_function(self):
        """Test that get_mi_estimator returns appropriate estimators."""
        np.random.seed(42)

        # Test discrete-discrete
        X_dd = np.random.randint(0, 3, (100, 1))
        y_dd = np.random.randint(0, 2, (100, 1))
        estimator_dd = get_mi_estimator("discrete", "discrete")
        result_dd = estimator_dd.estimate(X_dd, y_dd)
        assert result_dd >= 0

        # Test continuous-continuous
        X_cc = np.random.normal(0, 1, (100, 2))
        y_cc = np.random.normal(0, 1, (100, 1))
        estimator_cc = get_mi_estimator("continuous", "continuous")
        result_cc = estimator_cc.estimate(X_cc, y_cc)
        assert result_cc >= 0

        # Test continuous-discrete
        X_cd = np.random.normal(0, 1, (100, 2))
        y_cd = np.random.randint(0, 3, (100, 1))
        estimator_cd = get_mi_estimator("continuous", "discrete")
        result_cd = estimator_cd.estimate(X_cd, y_cd)
        assert result_cd >= 0


class TestTheoreticalProperties:
    """Tests to verify theoretical properties of mutual information."""

    def test_mi_symmetry(self):
        """Test that MI(X,Y) = MI(Y,X) for appropriate estimators."""
        np.random.seed(42)

        # Test with continuous variables
        X = np.random.normal(0, 1, (200, 1))
        y = np.random.normal(0, 1, (200, 1))

        estimator = CCMIEstimator(n_neighbors=5)
        mi_xy = estimator.estimate(X, y)
        mi_yx = estimator.estimate(y, X)

        # Should be approximately equal
        assert abs(mi_xy - mi_yx) < SMALL_TOLERANCE, f"MI not symmetric: {mi_xy:.3f} vs {mi_yx:.3f}"

    def test_mi_non_negativity(self):
        """Test that MI estimates are non-negative."""
        np.random.seed(42)

        estimators_and_data = [
            (DDMIEstimator(), np.random.randint(0, 3, (100, 1)), np.random.randint(0, 2, (100, 1))),
            (CCMIEstimator(), np.random.normal(0, 1, (100, 2)), np.random.normal(0, 1, (100, 1))),
            (
                CDMIRossEstimator(),
                np.random.normal(0, 1, (100, 2)),
                np.random.randint(0, 3, (100, 1)),
            ),
        ]

        for estimator, X, y in estimators_and_data:
            result = estimator.estimate(X, y)
            assert result >= 0, f"MI should be non-negative, got {result}"

    def test_mi_self_information(self):
        """Test that MI(X,X) = H(X) for discrete variables."""
        np.random.seed(42)
        X = np.random.randint(0, 4, (200, 1))

        estimator = DDMIEstimator()
        mi_xx = estimator.estimate(X, X)

        # Calculate entropy manually
        _, counts = np.unique(X, return_counts=True)
        prob = counts / len(X)
        entropy = -np.sum(prob * np.log2(prob))

        # MI(X,X) should equal H(X)
        assert abs(mi_xx - entropy) < SMALL_TOLERANCE, (
            f"MI(X,X) = {mi_xx:.3f} should equal H(X) = {entropy:.3f}"
        )
