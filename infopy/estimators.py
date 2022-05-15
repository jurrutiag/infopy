import warnings

import numpy as np
from scipy.special import digamma
from sklearn.neighbors import KDTree, NearestNeighbors
from abc import ABC, abstractmethod

from .functional import kozachenko_leonenko_entropy, discrete_entropy


class BaseMIEstimator(ABC):
    def __init__(self, flip_xy=False):
        self.flip_xy = flip_xy
    
    def estimate(self, X, y, local=False, conditioned_on=None):
        if self.flip_xy:
            X, y = y, X
        
        if conditioned_on is not None:
            z = conditioned_on
            if len(z.shape) != 2:
                raise ValueError(
                    "Condition variable must be a 2D array, if it's a vector, reshape it with .reshape(-1, 1)"
                )
                
            mi_xz_y = self._estimate(np.hstack((X, z)), y)
            mi_z_y = self._estimate(z, y)
            
            return mi_xz_y - mi_z_y

        return self._estimate(X, y, local)
            
    
    @abstractmethod
    def _estimate(self, X, y, local=False):
        pass


class DDMIEstimator(BaseMIEstimator):
    def _estimate(self, X, y, local=False):
        y = y.reshape(-1)
        unique_y = np.unique(y)
        unique_x, inverse_x = np.unique(X, axis=0, return_inverse=True)

        counts = np.zeros((unique_x.shape[0], unique_y.shape[0]))

        for i, uy in enumerate(unique_y):
            y_index = y == uy
            y_number_indices = np.arange(X.shape[0])[y_index]
            X_uy = X[y_index]
            unique_xuy, index_xuy, count_xuy = np.unique(
                X_uy, axis=0, return_index=True, return_counts=True
            )
            converted_indices = inverse_x[y_number_indices[index_xuy]]
            counts[converted_indices, i] = count_xuy

        probs = counts / counts.sum()
        p_x = probs.sum(axis=1, keepdims=True)
        p_c = probs.sum(axis=0, keepdims=True)

        if local:
            # Obtain this unique_x original index
            unique_x_indices = inverse_x.astype(int)
            
            # Obtain the corresponding y indices
            unique_y_indices = y.astype(int)
            
            # Filter the information to be only of selected samples and normalize probabilities before expectation
            mis = np.log2(probs / (p_x * p_c))[unique_x_indices, unique_y_indices]
            return mis

        else:
            IM = np.nansum(probs * np.log2(probs / (p_x * p_c)))

        return IM


class CDMIRossEstimator(BaseMIEstimator):
    def __init__(self, *args, n_neighbors=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_neighbors = n_neighbors

    def _estimate(self, X, c, local=False):
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


class CDMIEntropyBasedEstimator(BaseMIEstimator):
    def __init__(self, *args, n_neighbors=5, algorithm="auto", metric="euclidean", n_jobs=-1, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric
        self.n_jobs = n_jobs

    def _estimate(self, X, c, local=False):
        H = kozachenko_leonenko_entropy(
            X, local=True, n_neighbors=self.n_neighbors, metric=self.metric
        )
        for unique_c in np.unique(c, axis=0):
            mask = (unique_c == c).all(axis=1)
            Hc = kozachenko_leonenko_entropy(
                X[mask, :], local=True, n_neighbors=self.n_neighbors, metric=self.metric
            )
            H[mask] -= Hc

        if local:
            return H

        else:
            return np.mean(H)


class CCMIEstimator(BaseMIEstimator):
    def __init__(self, *args, n_neighbors=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_neighbors = n_neighbors

    def _estimate(self, X, y, local=False):
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


class ContinuousEntropyEstimator:
    def __init__(self, n_neighbors=4, metric="euclidean"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        
    def estimate(self, X, local=False):
        return kozachenko_leonenko_entropy(X, local=local, n_neighbors=self.n_neighbors, metric=self.metric)


class DiscreteEntropyEstimator:
    def estimate(self, X, local=False):
        return discrete_entropy(X, local=local)


def get_mi_estimator(x_type, y_type, local=False):
    if x_type == "discrete" and y_type == "discrete":
        return DDMIEstimator()

    elif x_type == "continuous" and y_type == "continuous":
        return CCMIEstimator()

    elif x_type in ["continuous", "discrete"] and y_type in ["continuous", "discrete"]:
        flip_xy = x_type == "discrete" and y_type == "continuous"
        if local:
            return CDMIEntropyBasedEstimator(flip_xy=flip_xy)

        else:
            return CDMIRossEstimator(flip_xy=flip_xy)

    else:
        raise ValueError(f"Unknown x_type: {x_type} or y_type: {y_type}")


def get_entropy_estimator(x_type, local=False):
    if x_type == "discrete":
        return DiscreteEntropyEstimator()

    elif x_type == "continuous":
        return ContinuousEntropyEstimator()

    else:
        raise ValueError(f"Unknown x_type: {x_type}")
