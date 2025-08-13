from typing import Union

import numpy as np
from scipy.special import gamma, psi
from sklearn.neighbors import NearestNeighbors  # type: ignore


def kozachenko_leonenko_entropy(
    X: np.ndarray, pointwise: bool = False, n_neighbors: int = 4, metric: str = "euclidean"
) -> Union[float, np.ndarray]:
    n_neighbors = min(n_neighbors, X.shape[0] - 2)

    nn = NearestNeighbors(metric=metric)
    nn.set_params(n_neighbors=n_neighbors)
    nn.fit(X)
    distances = nn.kneighbors()[0]

    r = distances[:, -1]

    n, m = np.shape(X)
    m = 1
    Vm = (np.pi ** (0.5 * m)) / gamma(0.5 * m + 1)

    rterm = np.log(r + np.finfo(X.dtype).eps)
    second = np.log(Vm)
    third = np.log((n - 1) * np.exp(-psi(n_neighbors)))

    ent = rterm + second + third

    if pointwise:
        pointwise_ent: np.ndarray = ent
        return pointwise_ent
    else:
        mean_ent: float = float(np.mean(ent))
        return max(0.0, mean_ent)


def discrete_entropy(X: np.ndarray, pointwise: bool = False) -> Union[float, np.ndarray]:
    if X.size == 0:
        raise ValueError("Input array cannot be empty")

    _, inverse, counts = np.unique(X, return_counts=True, return_inverse=True)
    p_X = counts / X.shape[0]

    log2 = -np.log2(p_X)

    if pointwise:
        pointwise_entropy: np.ndarray = log2[inverse]
        return pointwise_entropy
    else:
        return float(np.nansum(p_X * log2))
