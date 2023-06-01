---
marp: true
slideNumber: true
markdown.marp.enableHtml: true
title: Group 3 - Regression
---
<!--
footer: Members: Achka Pieer, Rickayzen Philipp, Würf Jerome, Tauscher Johannes, Popov Tomislav

15 Minuten Zeitlimit + 5 Minuten fragen
 -->

---
# Group 3: Regression
Content:
1. Introduction
2. Our Dataset
3. Comparison of Regression Models
    1. Least Squares
    2. Least Angle
    3. Partial Least Squares
    4. Ridge Regression
    5. Lasso

---

# 1. Introduction
- Our data set and model selection are based on the book ["The elements of statistical Learning"](https://hastie.su.domains/ElemStatLearn/)
- Our group focuses on linear regression problems
- A linear regression model is denoted function as $E(Y|X)$
    - $Y$ is the continuous output space on $R$
    - $X$ is the input space where an instance is a vector $\textbf{x}$ containing multiple measurements
- Benefits:
    - The trained models are interpretable
    - Reasonable performance on problems with sparse or low signal-to-noise data
---
# 1.i Basic idea behind regression
- $S(a)$: Sum of squared residuals
- $S(a) = \Sigma_{i=1}^N(y_i-a^Tx_i)^2$<img src="residual_square.png" style="height:500px; width:500px;float: right; margin-right: 10px;" />
    $=(y-Xa)^t(y-Xa)$
 - Optimal line be reducing S(a)
-  $a^*=arg\ \min_a{S(a)}\Rightarrow \nabla S(a)=0$
- Problems:
    - Heavily influenced by outlies
    - Tends to overfit
---
# 2. Our Dataset
- UCI Data Repository
- Data proposed in Elements of Statistical Learning
    - [Prostate cancer](https://hastie.su.domains/ElemStatLearn/)
- Synthetic Data

- Training & Test Datasets (80/20)
---
# 3. Comparison of Regression Models
1. Implement models using python libraries
2. Implement selected models from scratch
3. Evaluate and compare the implemented models
---
# 3.i Least Angle
---
# 3.ii Principal Component Regression (PCR)
- Combines Principal Component Analysis (PCA) and Linear Regression
- Reduces complexity and dimensionality
- Process:
    - Standardize predictors
    - Perform PCA on predictors to obtain Principal Components (PCs)
    - Select a subset of PCs based on explained variance
    - Regress response on selected PCs, treating each as an univariate regression
- Key Equations:
    - PCA: $Z_m = Xv_m$
    - PCR: $\hat{y}^{pcr}_{(M)} = \bar{y}1 + \sum_{m=1}^{M} \hat{\theta}_m z_m$
    - Coefficients: $\hat{\beta}^{pcr}(M) = \sum_{m=1}^{M} \hat{\theta}_m v_m$
- Considerations:
    - Selected PCs might lack physical interpretability
    - Standardization of predictors is necessary
    - The choice of M (number of PCs) affects model complexity

---
# 3.iii Partial Least Squares (PLS) - OPEN TO EDIT
- Supervised learning method, related to PCA
- Key Steps:
    - Standardize predictors and responses
    - Compute PLS direction: $Z_1 = \Sigma c_{jk} X_k$
    - Obtain PLS loadings ($\gamma_{1j}$) and weights ($\delta_{1k}$) by regressing responses and predictors on $Z_1$
    - Deflate predictors and responses, repeat for more PLS directions
- Key Equations:
    - PLS Direction: $Z_1 = \Sigma c_{jk} X_k$
    - PLS Loadings: $\gamma_{1j}$
    - PLS Weights: $\delta_{1k}$

---
# 3.iv Elastic Net
- $\min_{w} { \frac{1}{2n_{\text{samples}}} ||X w - y||_2 ^ 2 + \alpha \rho ||w||_1 +
\frac{\alpha(1-\rho)}{2} ||w||_2 ^ 2}$
- based on Least Squares
- combines penalties of the Lasso and Ridge regression
- Lasso adds a penalty based on the $l_1$-norm of the trained coefficients
- Ridge adds a penalty based on the $l_2$-norm of the trained coefficients
- amount of penalty is controlled via the hyper-parameter $\alpha$
---
# 3.v Locally Weighted Regression (LWR)
- Linear regression: $S(a)=(y-Xa)^T(y-Xa)$
- weighted regression:
    - certain data points get more weight than others
    - $S(a)=(y-Xa)^TW(y-Xa)$
- Locally weighted regression:
    - Idea: local points weight points in proximity higher
    - in total E independent weighted regressions
    - $S(a)=(y-Xa)^TW_E(y-Xa)$
    - e.g. $w_i = e^{\frac{-(x_i-x)^2}{2\tau^2}}$
---
# 3.vi Radial Basis Function Regression (RBFR)
- Idea: transform data into a higher dimension and then perform linear regression
- basis function: depends on distance to centre
- radial basis function: $\phi(x) = \phi(||x||)$
- linearly combine set of linear basis functions
- $S(a)=(y-\Phi(X)w)^T(y-\Phi(X)w)$
---
# 4. Evaluation
- For each model and dataset we will compare our implementation with the ones from the libraries
- Model Scores:
    - Mean Squared Error
    $\text{MSE}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} (y_i - \hat{y}_i)^2$
    - Mean Absolute Error
    $\text{MAE}(y, \hat{y}) = \frac{1}{n_{\text{samples}}} \sum_{i=0}^{n_{\text{samples}}-1} \left| y_i - \hat{y}_i \right|$
- Performance benchmarks:
    - Runtime
    - Memory
---
# 5.  Remarks and Outlook
- Currently we use "default" settings for the regression (hyper-)parameters
- in the future more work could be put into tuning the methods for better results.
    - by excluding specific features
    - by engineering new features

---
# 6. Literature 
- [Notes on Regularized Least-Squares](http://cbcl.mit.edu/publications/ps/MIT-CSAIL-TR-2007-025.pdf)

- [Elements of Statistical Learning (Hastie et al.)](https://hastie.su.domains/Papers/ESLII.pdf)
