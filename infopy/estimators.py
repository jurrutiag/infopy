import warnings

import numpy as np
from scipy.special import digamma
from sklearn.neighbors import KDTree, NearestNeighbors

from functional import kraskov_entropy


class CDMIRossEstimator:
    def __init__(self, n_neighbors=4):
        self.n_neighbors = n_neighbors

    def estimate(self, X, c, local=False):
        if local:
            warnings.warn(
                "CDMIRossEstimator should not be used with local MI. Use CDMIEntropyBasedEstimator instead."
            )

        if len(X.shape) != 2 or len(c.shape) != 2:
            raise ValueError(
                "X and c must be 2D arrays, if they are vectors, reshape them with .reshape(-1, 1)"
            )

        X = X + np.random.randn(*X.shape) * 1e-10
        n_samples = X.shape[0]

        if len(X.shape) == 1:
            X = X.reshape((-1, 1))

        radius = np.empty(n_samples)
        label_counts = np.empty(n_samples)
        k_all = np.empty(n_samples)

        nn = NearestNeighbors()
        for label in np.unique(c, axis=0):
            mask = (c == label).all(axis=1)
            count = np.sum(mask)
            if count > 1:
                k = min(self.n_neighbors, count - 1)

                nn.set_params(n_neighbors=k)
                nn.fit(X[mask, :])
                r = nn.kneighbors()[0]
                print(r)
                radius[mask] = np.nextafter(r[:, -1], 0)

                k_all[mask] = k

            label_counts[mask] = count

        # Ignore points with unique labels.
        mask = label_counts > 1
        n_samples = np.sum(mask)
        label_counts = label_counts[mask]
        k_all = k_all[mask]
        X = X[mask, :]
        radius = radius[mask]

        kd = KDTree(X)
        m_all = kd.query_radius(X, radius, count_only=True, return_distance=False)
        m_all = np.array(m_all) - 1.0

        mis = (
            digamma(n_samples)
            + digamma(k_all)
            - digamma(label_counts)
            - digamma(m_all + 1)
        )

        if local:
            return mis

        else:
            return max(0, np.mean(mis))


class CDMIEntropyBasedEstimator:
    def __init__(self, n_neighbors=5, algorithm="auto", metric="euclidean", n_jobs=-1):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric
        self.n_jobs = n_jobs

    def estimate(self, X, c, local=False):
        H = kraskov_entropy(
            X, local=True, n_neighbors=self.n_neighbors, metric=self.metric
        )
        for unique_c in np.unique(c, axis=0):
            mask = (unique_c == c).all(axis=1)
            Hc = kraskov_entropy(
                X[mask, :], local=True, n_neighbors=self.n_neighbors, metric=self.metric
            )
            H[mask] -= Hc

        if local:
            return H

        else:
            return np.mean(H)


class CCMIEstimator:
    def __init__(self, n_neighbors=4):
        self.n_neighbors = n_neighbors

    def estimate(self, X, y, local=False):
        if len(X.shape) != 2 or len(y.shape) != 2:
            raise ValueError(
                "X and c must be 2D arrays, if they are vectors, reshape them with .reshape(-1, 1)"
            )

        X = X + np.random.randn(*X.shape) * 1e-8
        y = y + np.random.randn(*y.shape) * 1e-8

        n_samples = X.shape[0]

        xy = np.hstack((X, y))

        nn = NearestNeighbors(metric="chebyshev", n_neighbors=self.n_neighbors)

        nn.fit(xy)
        radius = nn.kneighbors()[0]
        radius = np.nextafter(radius[:, -1], 0)

        kd = KDTree(X, metric="chebyshev")
        nx = kd.query_radius(X, radius, count_only=True, return_distance=False)
        nx = np.array(nx) - 1.0

        kd = KDTree(y, metric="chebyshev")
        ny = kd.query_radius(y, radius, count_only=True, return_distance=False)
        ny = np.array(ny) - 1.0

        mis = (
            digamma(n_samples)
            + digamma(self.n_neighbors)
            - digamma(nx + 1)
            - digamma(ny + 1)
        )

        if local:
            return mis

        else:
            return max(0, np.mean(mis))
