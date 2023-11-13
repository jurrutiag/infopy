# infopy
Information Theory Library for Python containing implementations of mutual information and entropy estimators.

# Usage

## MI Estimators:

Currently, there are 6 MI estimators implemented in `infopy`. These estimators are used to estimate $I(X; Y)$, with different implementations depending on whether $X$ or $Y$ are continuous or discrete. All of them support multidimensional $X$ and $Y$, i.e., $X, Y \in \mathbb{R}^{n}$, treating them as random vectors instead of scalar random variables. The available estimators are:

* `estimators.DDMIEstimator`: For discrete $X$ and discrete $Y$, based on maximum likelihood estimation of the PMF of X, Y and (X, Y).
* `estimators.CDMIRossEstimator`: For continuous $X$ and discrete $Y$ (interchangeable), based on Ross MI estimation [1]
* `estimators.CDMIEntropyBasedEstimator`: For continuous $X$ and discrete $Y$ (interchangeable), based on Kozachenko-Leonenko entropy estimation.
* `estimators.CCMIEstimator`: For continuous $X$ and continuous $Y$, based on Kraskov MI estimator [2].
* `estimators.MixedMIEstimator`: For mixed $X$ and $Y$. It uses the Gao MI estimator [3]. Note: This estimator has not yet been successfully tested.
* `estimators.EDGEMIEstimator`: This estimator is based on the method described in [4]. It is believed to be applicable for any variable type, but successful results have not yet been obtained.

To automatically select an appropriate estimator based on the types of $X$ and $Y$, use the `estimators.get_mi_estimator` function. The `pointwise` parameter in this function specifies whether to obtain an estimator that provides an estimation per sample (pointwise mutual information) instead of averaging.

# References

1. B. C. Ross “Mutual Information between Discrete and Continuous Data Sets”. PLoS ONE 9(2), 2014. <br/>
2. A. Kraskov, H. Stogbauer and P. Grassberger, “Estimating mutual information”. Phys. Rev. E 69, 2004. <br/>
3. Gao, Weihao, et al. Estimating Mutual Information for Discrete-Continuous Mixtures. 2018. <br/>
4. Noshad, Morteza, et al. Scalable Mutual Information Estimation Using Dependence Graphs. 2018. <br/>
