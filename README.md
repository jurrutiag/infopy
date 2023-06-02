# infopy
Information Theory Toolbox for Python containing implementations of mutual information and entropy estimators.

# Usage

## MI Estimators:

Currently there are 6 MI estimators implemented. Given $I(X; Y)$ the mutual information to estimate, we have:

* `estimators.DDMIEstimator`: Used for discrete $X$ and discrete $Y$, based on maximum likelihood estimation of the PMF of S, C and (S, C).
* `estimators.CDMIRossEstimator`: Used for continuous $X$ and discrete $Y$ (interchangeable), based on Ross MI estimation [1]
* `estimators.CDMIEntropyBasedEstimator`: Used for continuous $X$ and discrete $Y$ (interchangeable), based on Kozachenko-Leonenko entropy estimation.
* `estimators.CCMIEstimator`: Used for continuous $X$ and continuous $Y$, based on Kraskov MI estimator [2].
* `estimators.MixedMIEstimator`: Used for mixed $X$ and mixed $Y$, based on Gao MI estimator [3]. This estimator hasn't been used successfully.
* `estimators.EDGEMIEstimator`: Based on [4]. I believe this can be used for any variable type. I also haven't obtained any successful results on this.

If you know the type of $X$ and $Y$, you can automatically obtain an estimator by using the function `estimators.get_mi_estimator`. The local parameter specifies if you want to obtain an estimator that provides an estimation per sample (without averaging), this is called pointwise mutual information (I need to change the argument name).

# References

[1] B. C. Ross “Mutual Information between Discrete and Continuous Data Sets”. PLoS ONE 9(2), 2014. <br/>
[2] A. Kraskov, H. Stogbauer and P. Grassberger, “Estimating mutual information”. Phys. Rev. E 69, 2004. <br/>
[3] Gao, Weihao, et al. Estimating Mutual Information for Discrete-Continuous Mixtures. 2018. <br/>
[4] Noshad, Morteza, et al. Scalable Mutual Information Estimation Using Dependence Graphs. 2018. <br/>
