# infopy-estimators

[![Python](https://img.shields.io/pypi/pyversions/infopy-estimators)](https://pypi.org/project/infopy-estimators/)
[![PyPI](https://img.shields.io/pypi/v/infopy-estimators)](https://pypi.org/project/infopy-estimators/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python library for estimating mutual information (MI) and entropy in discrete, continuous, and mixed random variables. Built with a focus on performance and ease of use, `infopy-estimators` provides implementations of various information-theoretic estimators.

## Features

- **Multiple MI Estimators**: Support for discrete-discrete, continuous-discrete, continuous-continuous, and mixed variable types
- **Multidimensional Support**: All estimators handle multidimensional random vectors (X, Y ∈ ℝⁿ)
- **Pointwise MI**: Calculate mutual information per sample, not just averages
- **Conditional MI**: Compute conditional mutual information I(X;Y|Z)
- **Automatic Estimator Selection**: Let the library choose the best estimator based on your data types
- **Entropy Estimation**: Dedicated entropy estimators for continuous and discrete variables
- **Type-Safe**: Full type hints and mypy compliance

## Installation

### Using pip
```bash
pip install infopy-estimators
```

### Using uv (recommended for development)
```bash
uv add infopy-estimators
```

### From source
```bash
git clone https://github.com/jurrutiag/infopy.git
cd infopy
uv sync  # or pip install -e .
```

## Quick Start

### Basic Usage

```python
import numpy as np
from infopy import get_mi_estimator

# Generate sample data
X = np.random.randn(1000, 2)  # Continuous 2D variable
Y = np.random.randn(1000, 1)  # Continuous 1D variable

# Automatically select appropriate estimator
estimator = get_mi_estimator(x_type="continuous", y_type="continuous")

# Estimate mutual information
mi = estimator.estimate(X, Y)
print(f"Mutual Information: {mi:.4f}")
```

### Pointwise Mutual Information

```python
# Get MI for each sample instead of average
estimator = get_mi_estimator(x_type="continuous", y_type="continuous", pointwise_suited=True)
pointwise_mi = estimator.estimate(X, Y, pointwise=True)
print(f"Pointwise MI shape: {pointwise_mi.shape}")  # (1000,)
```

### Mixed Variable Types

```python
# Continuous and discrete variables
X_continuous = np.random.randn(1000, 3)
Y_discrete = np.random.randint(0, 5, size=(1000, 1))

estimator = get_mi_estimator(x_type="continuous", y_type="discrete")
mi = estimator.estimate(X_continuous, Y_discrete)
print(f"Continuous-Discrete MI: {mi:.4f}")
```

### Conditional Mutual Information

```python
# Compute I(X;Y|Z)
X = np.random.randn(1000, 2)
Y = np.random.randn(1000, 2)
Z = np.random.randn(1000, 1)

estimator = get_mi_estimator(x_type="continuous", y_type="continuous")
cmi = estimator.estimate_conditional(X, Y, Z)
print(f"Conditional MI: {cmi:.4f}")
```

### Entropy Estimation

```python
from infopy import ContinuousEntropyEstimator, DiscreteEntropyEstimator

# Continuous entropy
X_continuous = np.random.randn(1000, 3)
cont_estimator = ContinuousEntropyEstimator()
entropy = cont_estimator.estimate(X_continuous)
print(f"Continuous Entropy: {entropy:.4f}")

# Discrete entropy
X_discrete = np.random.randint(0, 10, size=(1000, 2))
disc_estimator = DiscreteEntropyEstimator()
entropy = disc_estimator.estimate(X_discrete)
print(f"Discrete Entropy: {entropy:.4f}")
```

## Available Estimators

### Mutual Information Estimators

| Estimator | Variable Types | Method | Reference |
|-----------|---------------|---------|-----------|
| `DDMIEstimator` | Discrete-Discrete | Maximum likelihood PMF estimation | - |
| `CDMIRossEstimator` | Continuous-Discrete | Ross method | Ross (2014) [1] |
| `CDMIEntropyBasedEstimator` | Continuous-Discrete | Kozachenko-Leonenko entropy | - |
| `CCMIEstimator` | Continuous-Continuous | Kraskov estimator | Kraskov et al. (2004) [2] |
| `MixedMIEstimator` | Mixed types | Gao estimator (experimental) | Gao et al. (2018) [3] |

### Entropy Estimators

| Estimator | Variable Type | Method |
|-----------|--------------|---------|
| `ContinuousEntropyEstimator` | Continuous | Kozachenko-Leonenko k-NN |
| `DiscreteEntropyEstimator` | Discrete | Maximum likelihood |

### Automatic Estimator Selection

The `get_mi_estimator()` function automatically selects the appropriate estimator:

```python
from infopy import get_mi_estimator

# For continuous-continuous variables
estimator = get_mi_estimator(x_type="continuous", y_type="continuous")

# For discrete-discrete variables
estimator = get_mi_estimator(x_type="discrete", y_type="discrete")

# For mixed types
estimator = get_mi_estimator(x_type="continuous", y_type="discrete")

# For pointwise MI calculation
estimator = get_mi_estimator(x_type="continuous", y_type="continuous", pointwise_suited=True)
```

## API Reference

### BaseMIEstimator

All MI estimators inherit from `BaseMIEstimator` and implement:

```python
class BaseMIEstimator:
    def estimate(self, X: np.ndarray, Y: np.ndarray, pointwise: bool = False) -> Union[float, np.ndarray]:
        """Estimate mutual information between X and Y."""

    def estimate_conditional(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> float:
        """Estimate conditional mutual information I(X;Y|Z)."""
```

### Parameters

- `X, Y`: Input arrays of shape `(n_samples, n_features)`
- `pointwise`: If True, return MI for each sample; if False, return average MI
- `Z`: Conditioning variable for conditional MI

### Returns

- `estimate()`: Float (average MI) or array of shape `(n_samples,)` (pointwise MI)
- `estimate_conditional()`: Float (conditional MI)

## Advanced Usage

### Custom k-NN Parameters

```python
from infopy import CCMIEstimator

# Use custom k for k-NN estimation
estimator = CCMIEstimator(k=5)  # Default is k=3
mi = estimator.estimate(X, Y)
```

### Handling High-Dimensional Data

```python
# For high-dimensional continuous data
X = np.random.randn(1000, 50)  # 50-dimensional
Y = np.random.randn(1000, 30)  # 30-dimensional

# CCMIEstimator handles high dimensions well
estimator = CCMIEstimator(k=10)  # Increase k for stability
mi = estimator.estimate(X, Y)
```

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/jurrutiag/infopy.git
cd infopy

# Install with development dependencies
uv sync --extra testing
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=infopy

# Run specific test
uv run pytest tests/test_estimators.py::TestCCMIEstimator
```

### Code Quality

```bash
# Format code
uv run ruff format

# Lint code
uv run ruff check

# Type checking
uv run mypy src/
```

## Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Ensure all tests pass and code is formatted
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Standards

- **Formatter**: ruff format (100 char line length)
- **Linter**: ruff with comprehensive rules
- **Type Checker**: mypy with strict settings
- **Test Framework**: pytest with coverage
- **Package Manager**: uv

## Citation

If you use this library in your research, please cite:

```bibtex
@software{infopy_estimators,
  author = {Urrutia, Juan},
  title = {infopy-estimators: Information Theory Estimators for Python},
  url = {https://github.com/jurrutiag/infopy},
  year = {2024}
}
```

## References

1. B. C. Ross "Mutual Information between Discrete and Continuous Data Sets". PLoS ONE 9(2), 2014.
2. A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual information". Phys. Rev. E 69, 2004.
3. Gao, Weihao, et al. "Estimating Mutual Information for Discrete-Continuous Mixtures". NeurIPS, 2018.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/jurrutiag/infopy/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jurrutiag/infopy/discussions)
- **Email**: juan.urrutia.gandolfo@gmail.com
