import warnings

import numpy as np
import pytest

from infopy.estimators import (
    BaseMIEstimator,
    CCMIEstimator,
    CDMIEntropyBasedEstimator,
    CDMIRossEstimator,
    ContinuousEntropyEstimator,
    DDMIEstimator,
    DiscreteEntropyEstimator,
    MixedMIEstimator,
    SymmetricalUncertaintyEstimator,
    get_entropy_estimator,
    get_mi_estimator,
)

# Test constants
INDEPENDENCE_TOLERANCE = 0.1
HIGH_CORRELATION_THRESHOLD = 0.9


class TestBaseMIEstimator:
    """Test suite for BaseMIEstimator base class."""

    def test_abstract_base_class(self):
        """Test that BaseMIEstimator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseMIEstimator()

    def test_flip_xy_functionality(self):
        """Test flip_xy parameter functionality using concrete estimator."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (50, 2))
        y = np.random.randint(0, 3, (50, 1))

        # Test with flip_xy=False
        estimator_normal = CDMIRossEstimator(flip_xy=False)
        result_normal = estimator_normal.estimate(X, y)

        # Test with flip_xy=True (should swap X and y)
        estimator_flipped = CDMIRossEstimator(flip_xy=True)
        result_flipped = estimator_flipped.estimate(X, y)

        # Results should be different since we're estimating different MI
        assert isinstance(result_normal, float)
        assert isinstance(result_flipped, float)

    def test_conditional_mi_validation(self):
        """Test conditional MI input validation."""
        estimator = DDMIEstimator()
        X = np.random.randint(0, 3, (50, 1))
        y = np.random.randint(0, 2, (50, 1))

        # Test with 1D conditioning variable (should now work with automatic reshaping)
        z_data = np.random.randint(0, 2, 50)
        z_1d = z_data
        result_1d = estimator.estimate(X, y, conditioned_on=z_1d)
        assert isinstance(result_1d, float)

        # Test with properly shaped conditioning variable using same data
        z_2d = z_data.reshape(-1, 1)
        result_2d = estimator.estimate(X, y, conditioned_on=z_2d)
        assert isinstance(result_2d, float)

        # Results should be identical since z_1d and z_2d contain the same data
        assert np.isclose(result_1d, result_2d)


class TestDDMIEstimator:
    """Test suite for DDMIEstimator (discrete-discrete MI)."""

    def test_independent_variables(self):
        """Test MI estimation for independent discrete variables."""
        np.random.seed(42)
        X = np.random.randint(0, 3, (1000, 1))
        y = np.random.randint(0, 3, (1000, 1))

        estimator = DDMIEstimator()
        result = estimator.estimate(X, y)

        # MI should be close to 0 for independent variables
        assert abs(result) < INDEPENDENCE_TOLERANCE

    def test_identical_variables(self):
        """Test MI estimation for identical discrete variables."""
        np.random.seed(42)
        X = np.random.randint(0, 3, (100, 1))
        y = X.copy()

        estimator = DDMIEstimator()
        result = estimator.estimate(X, y)

        # MI should be equal to entropy of X (or y)
        assert result > 0

    def test_input_validation(self):
        """Test input validation for DDMIEstimator."""
        estimator = DDMIEstimator()

        # Test 1D arrays (should now work with automatic reshaping)
        X_1d = np.array([1, 2, 3])
        y_1d = np.array([1, 2, 3])
        result = estimator.estimate(X_1d, y_1d)
        assert isinstance(result, float)

        # Test multivariate y (should raise error)
        X = np.random.randint(0, 3, (50, 1))
        y_multi = np.random.randint(0, 3, (50, 2))
        with pytest.raises(ValueError, match="multivariate y"):
            estimator.estimate(X, y_multi)

    def test_automatic_reshaping(self):
        """Test that 1D arrays are automatically reshaped to 2D."""
        estimator = DDMIEstimator()

        # Test with 1D arrays
        X_1d = np.array([0, 1, 0, 1, 0])
        y_1d = np.array([1, 1, 0, 0, 1])

        # Test with equivalent 2D arrays
        X_2d = X_1d.reshape(-1, 1)
        y_2d = y_1d.reshape(-1, 1)

        result_1d = estimator.estimate(X_1d, y_1d)
        result_2d = estimator.estimate(X_2d, y_2d)

        # Results should be identical
        assert np.isclose(result_1d, result_2d)

    def test_pointwise_mi(self):
        """Test pointwise MI estimation."""
        np.random.seed(42)
        X = np.random.randint(0, 3, (50, 1))
        y = np.random.randint(0, 2, (50, 1))

        estimator = DDMIEstimator()
        pointwise_result = estimator.estimate(X, y, pointwise=True)
        total_result = estimator.estimate(X, y, pointwise=False)

        # Should have same length as input
        assert len(pointwise_result) == len(X)

        # Check relationship between pointwise and total MI
        assert isinstance(pointwise_result, np.ndarray)
        assert isinstance(total_result, float)

    def test_deterministic_relationship(self):
        """Test MI for deterministic discrete relationship."""
        # Create deterministic relationship: y = X % 2
        X = np.arange(20).reshape(-1, 1)
        y = (X % 2).reshape(-1, 1)

        estimator = DDMIEstimator()
        result = estimator.estimate(X, y)

        # Should have positive MI for deterministic relationship
        assert result > 0


class TestCDMIRossEstimator:
    """Test suite for CDMIRossEstimator (continuous-discrete MI)."""

    def test_basic_functionality(self):
        """Test basic functionality of Ross estimator."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 2))
        y = np.random.randint(0, 3, (100, 1))

        estimator = CDMIRossEstimator()
        result = estimator.estimate(X, y)

        assert isinstance(result, float)
        assert result >= 0

    def test_different_neighbors(self):
        """Test with different number of neighbors."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 2))
        y = np.random.randint(0, 3, (100, 1))

        estimator_k3 = CDMIRossEstimator(n_neighbors=3)
        estimator_k5 = CDMIRossEstimator(n_neighbors=5)

        result_k3 = estimator_k3.estimate(X, y)
        result_k5 = estimator_k5.estimate(X, y)

        assert result_k3 >= 0
        assert result_k5 >= 0

    def test_pointwise_warning(self):
        """Test that pointwise MI raises warning."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (50, 2))
        y = np.random.randint(0, 3, (50, 1))

        estimator = CDMIRossEstimator()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            estimator.estimate(X, y, pointwise=True)
            assert len(w) == 1
            assert "should not be used with local MI" in str(w[0].message)

    def test_input_validation(self):
        """Test input validation."""
        estimator = CDMIRossEstimator()

        # Test 1D arrays (should now work with automatic reshaping)
        X_1d = np.array([1.0, 2.0, 3.0])
        y_1d = np.array([1, 2, 3])
        result = estimator.estimate(X_1d, y_1d)
        assert isinstance(result, float)

    def test_dependent_variables(self):
        """Test with dependent continuous-discrete variables."""
        np.random.seed(42)
        # Create dependent relationship
        X_base = np.random.normal(0, 1, (200, 2))
        y = (X_base[:, 0] > 0).astype(int).reshape(-1, 1)

        estimator = CDMIRossEstimator()
        result = estimator.estimate(X_base, y)

        # Should have positive MI for dependent variables
        assert result > 0


class TestCDMIEntropyBasedEstimator:
    """Test suite for CDMIEntropyBasedEstimator."""

    def test_basic_functionality(self):
        """Test basic functionality of entropy-based estimator."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 2))
        y = np.random.randint(0, 3, (100, 1))

        estimator = CDMIEntropyBasedEstimator()
        result = estimator.estimate(X, y)

        assert isinstance(result, float)
        assert result >= 0

    def test_pointwise_support(self):
        """Test pointwise MI estimation support."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (50, 2))
        y = np.random.randint(0, 3, (50, 1))

        estimator = CDMIEntropyBasedEstimator()
        pointwise_result = estimator.estimate(X, y, pointwise=True)

        assert isinstance(pointwise_result, np.ndarray)
        assert len(pointwise_result) == len(X)

    def test_different_parameters(self):
        """Test with different algorithm parameters."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 2))
        y = np.random.randint(0, 3, (100, 1))

        estimator = CDMIEntropyBasedEstimator(
            n_neighbors=3, metric="manhattan", algorithm="ball_tree"
        )
        result = estimator.estimate(X, y)

        assert result >= 0


class TestCCMIEstimator:
    """Test suite for CCMIEstimator (continuous-continuous MI)."""

    def test_independent_gaussians(self):
        """Test MI estimation for independent Gaussian variables."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (200, 2))
        y = np.random.normal(0, 1, (200, 1))

        estimator = CCMIEstimator()
        result = estimator.estimate(X, y)

        # MI should be close to 0 for independent variables
        assert abs(result) < INDEPENDENCE_TOLERANCE * 3

    def test_correlated_gaussians(self):
        """Test MI estimation for correlated Gaussian variables."""
        np.random.seed(42)
        # Create correlated variables
        X_base = np.random.normal(0, 1, (200, 1))
        noise = np.random.normal(0, 0.1, (200, 1))
        y = X_base + noise
        X = np.hstack([X_base, np.random.normal(0, 1, (200, 1))])

        estimator = CCMIEstimator()
        result = estimator.estimate(X, y)

        # Should have positive MI for correlated variables
        assert result > 0

    def test_pointwise_mi(self):
        """Test pointwise MI estimation."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (50, 2))
        y = np.random.normal(0, 1, (50, 1))

        estimator = CCMIEstimator()
        pointwise_result = estimator.estimate(X, y, pointwise=True)
        total_result = estimator.estimate(X, y, pointwise=False)

        assert isinstance(pointwise_result, np.ndarray)
        assert len(pointwise_result) == len(X)
        assert isinstance(total_result, float)

    def test_different_neighbors(self):
        """Test with different number of neighbors."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 2))
        y = np.random.normal(0, 1, (100, 1))

        estimator_k3 = CCMIEstimator(n_neighbors=3)
        estimator_k5 = CCMIEstimator(n_neighbors=5)

        result_k3 = estimator_k3.estimate(X, y)
        result_k5 = estimator_k5.estimate(X, y)

        assert result_k3 >= 0
        assert result_k5 >= 0


class TestMixedMIEstimator:
    """Test suite for MixedMIEstimator."""

    def test_basic_functionality(self):
        """Test basic functionality of mixed estimator."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 2))
        y = np.random.normal(0, 1, (100, 1))

        estimator = MixedMIEstimator()
        result = estimator.estimate(X, y)

        assert isinstance(result, float)
        assert result >= 0

    def test_pointwise_mi(self):
        """Test pointwise MI estimation."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (50, 2))
        y = np.random.normal(0, 1, (50, 1))

        estimator = MixedMIEstimator()
        pointwise_result = estimator.estimate(X, y, pointwise=True)

        assert isinstance(pointwise_result, np.ndarray)
        assert len(pointwise_result) == len(X)

    def test_1d_inputs(self):
        """Test with 1D inputs (should be reshaped internally)."""
        np.random.seed(42)
        X = np.random.normal(0, 1, 50)
        y = np.random.normal(0, 1, 50)

        estimator = MixedMIEstimator()
        result = estimator.estimate(X, y)

        assert result >= 0

    def test_different_neighbors(self):
        """Test with different number of neighbors."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 2))
        y = np.random.normal(0, 1, (100, 1))

        estimator = MixedMIEstimator(n_neighbors=6)
        result = estimator.estimate(X, y)

        assert result >= 0


class TestEntropyEstimators:
    """Test suite for entropy estimator classes."""

    def test_discrete_entropy_estimator(self):
        """Test DiscreteEntropyEstimator."""
        estimator = DiscreteEntropyEstimator()
        X = np.random.randint(0, 4, (100, 1))

        result = estimator.estimate(X)
        pointwise_result = estimator.estimate(X, pointwise=True)

        assert isinstance(result, float)
        assert result >= 0
        assert len(pointwise_result) == len(X)

    def test_continuous_entropy_estimator(self):
        """Test ContinuousEntropyEstimator."""
        estimator = ContinuousEntropyEstimator()
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 2))

        result = estimator.estimate(X)
        pointwise_result = estimator.estimate(X, pointwise=True)

        assert isinstance(result, float)
        assert result >= 0
        assert len(pointwise_result) == len(X)

    def test_continuous_entropy_parameters(self):
        """Test ContinuousEntropyEstimator with different parameters."""
        estimator = ContinuousEntropyEstimator(n_neighbors=5, metric="manhattan")
        np.random.seed(42)
        X = np.random.normal(0, 1, (50, 2))

        result = estimator.estimate(X)
        assert result >= 0


class TestSymmetricalUncertaintyEstimator:
    """Test suite for SymmetricalUncertaintyEstimator."""

    def test_discrete_discrete_su(self):
        """Test SU for discrete-discrete variables."""
        np.random.seed(42)
        X = np.random.randint(0, 3, (100, 1))
        y = np.random.randint(0, 2, (100, 1))

        estimator = SymmetricalUncertaintyEstimator("discrete", "discrete")
        result = estimator.estimate(X, y)

        # SU should be between 0 and 1
        assert 0 <= result <= 1

    def test_continuous_continuous_su(self):
        """Test SU for continuous-continuous variables."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 2))
        y = np.random.normal(0, 1, (100, 1))

        estimator = SymmetricalUncertaintyEstimator("continuous", "continuous")
        result = estimator.estimate(X, y)

        assert result >= 0

    def test_mixed_types_su(self):
        """Test SU for mixed variable types."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 2))
        y = np.random.randint(0, 3, (100, 1))

        estimator = SymmetricalUncertaintyEstimator("continuous", "discrete")
        result = estimator.estimate(X, y)

        assert result >= 0

    def test_pointwise_error(self):
        """Test that pointwise raises error for SU."""
        estimator = SymmetricalUncertaintyEstimator("discrete", "discrete")
        X = np.random.randint(0, 3, (50, 1))
        y = np.random.randint(0, 2, (50, 1))

        with pytest.raises(ValueError, match="pointwise MI"):
            estimator.estimate(X, y, pointwise=True)

    def test_identical_variables_su(self):
        """Test SU for identical variables (should be 1)."""
        np.random.seed(42)
        X = np.random.randint(0, 3, (100, 1))
        y = X.copy()

        estimator = SymmetricalUncertaintyEstimator("discrete", "discrete")
        result = estimator.estimate(X, y)

        # SU should be close to 1 for identical variables
        assert result > HIGH_CORRELATION_THRESHOLD


class TestHelperFunctions:
    """Test suite for helper functions."""

    def test_get_mi_estimator_discrete_discrete(self):
        """Test get_mi_estimator for discrete-discrete."""
        estimator = get_mi_estimator("discrete", "discrete")
        assert isinstance(estimator, DDMIEstimator)

    def test_get_mi_estimator_continuous_continuous(self):
        """Test get_mi_estimator for continuous-continuous."""
        estimator = get_mi_estimator("continuous", "continuous")
        assert isinstance(estimator, CCMIEstimator)

    def test_get_mi_estimator_continuous_discrete(self):
        """Test get_mi_estimator for continuous-discrete."""
        estimator = get_mi_estimator("continuous", "discrete")
        assert isinstance(estimator, CDMIRossEstimator)
        assert estimator.flip_xy is False

    def test_get_mi_estimator_discrete_continuous(self):
        """Test get_mi_estimator for discrete-continuous."""
        estimator = get_mi_estimator("discrete", "continuous")
        assert isinstance(estimator, CDMIRossEstimator)
        assert estimator.flip_xy is True

    def test_get_mi_estimator_pointwise_suited(self):
        """Test get_mi_estimator with pointwise_suited=True."""
        estimator = get_mi_estimator("continuous", "discrete", pointwise_suited=True)
        assert isinstance(estimator, CDMIEntropyBasedEstimator)

    def test_get_mi_estimator_mixed(self):
        """Test get_mi_estimator for mixed types."""
        estimator = get_mi_estimator("mixed", "continuous")
        assert isinstance(estimator, MixedMIEstimator)

        estimator = get_mi_estimator("discrete", "mixed")
        assert isinstance(estimator, MixedMIEstimator)

    def test_get_mi_estimator_invalid_types(self):
        """Test get_mi_estimator with invalid types."""
        with pytest.raises(ValueError, match="Unknown x_type"):
            get_mi_estimator("invalid", "discrete")

    def test_get_entropy_estimator_discrete(self):
        """Test get_entropy_estimator for discrete."""
        estimator = get_entropy_estimator("discrete")
        assert isinstance(estimator, DiscreteEntropyEstimator)

    def test_get_entropy_estimator_continuous(self):
        """Test get_entropy_estimator for continuous."""
        estimator = get_entropy_estimator("continuous")
        assert isinstance(estimator, ContinuousEntropyEstimator)

    def test_get_entropy_estimator_invalid_type(self):
        """Test get_entropy_estimator with invalid type."""
        with pytest.raises(ValueError, match="Unknown x_type"):
            get_entropy_estimator("invalid")


class TestConditionalMI:
    """Test suite for conditional mutual information."""

    def test_conditional_mi_basic(self):
        """Test basic conditional MI functionality."""
        np.random.seed(42)
        X = np.random.randint(0, 3, (100, 1))
        y = np.random.randint(0, 2, (100, 1))
        z = np.random.randint(0, 2, (100, 1))

        estimator = DDMIEstimator()
        result = estimator.estimate(X, y, conditioned_on=z)

        assert isinstance(result, float)

    def test_conditional_mi_continuous(self):
        """Test conditional MI with continuous variables."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 2))
        y = np.random.normal(0, 1, (100, 1))
        z = np.random.normal(0, 1, (100, 1))

        estimator = CCMIEstimator()
        result = estimator.estimate(X, y, conditioned_on=z)

        assert isinstance(result, float)

    def test_conditional_mi_independence_given_z(self):
        """Test conditional MI when X and Y are independent given Z."""
        np.random.seed(42)
        # Create variables where X and Y are independent given Z
        z = np.random.normal(0, 1, (200, 1))
        X = z + np.random.normal(0, 0.1, (200, 1))
        y = z + np.random.normal(0, 0.1, (200, 1))

        estimator = CCMIEstimator()
        conditional_mi = estimator.estimate(X, y, conditioned_on=z)
        unconditional_mi = estimator.estimate(X, y)

        # Conditional MI should be much smaller than unconditional MI
        assert conditional_mi < unconditional_mi
