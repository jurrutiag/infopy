import numpy as np
from scipy.special import digamma
from sklearn.neighbors import KDTree, NearestNeighbors


class CDMIEstimator:
    def __init__(self, *args, n_neighbors=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_neighbors = n_neighbors
        self.cache_for_sample_selection = None
        self.device = "cpu"

    def estimate(self, X, c, local=False):
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
