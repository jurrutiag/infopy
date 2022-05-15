import numpy as np
from scipy.special import gamma, psi
from sklearn.neighbors import NearestNeighbors


def kozachenko_leonenko_entropy(X, local=False, n_neighbors=4, metric="euclidean"):
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

    if local:
        return ent

    else:
        return max(0, np.mean(ent))


def discrete_entropy(X, local=False):
    _, inverse, counts = np.unique(X, return_counts=True, return_inverse=True)
    p_X = counts / X.shape[0]

    log2 = -np.log2(p_X)

    if local:
        return log2[inverse]

    else:
        return np.nansum(p_X * log2)
