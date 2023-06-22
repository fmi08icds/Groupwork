import numpy as np
from numpy import ndarray

np.set_printoptions(precision=2)

class SVM:
    """ 
    SVM implementation using Sequential Minimal Optimization 
    
    Parameters
    ----------
    C: float
        regularizarion paramter
    tol: float
        -
    max_iter: int
        maximum number of iterations

    """
    def _smo(self, X: ndarray, y: ndarray, C: float, EPSILON = 0.1):
        """ 
        Sequential Minimal Optimization (SMO):
        Update alpha two elements at a time to maximize 
        the constrained objective function
        
        Reference
        ---------
        https://en.wikipedia.org/wiki/Sequential_minimal_optimization
        https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf
        https://www.youtube.com/watch?v=Mfp7HQKLSAo
        """
        n = y.shape[0]
        self._alpha = np.zeros(n)

        for _ in range(2):
            for i1 in range(n):
                # Select alpha2 based on maximum violation of KKT conditions
                E1 = y[i1] * (self.weights.T @ X[i1]) + self.bias - y[i1]
                max_violation = 0
                i2 = -1
                for j in range(n):
                    if j == i1:
                        continue
                    E2 = y[j] * (self.weights.T @ X[j]) + self.bias - y[j]
                    violation = np.abs(E1 - E2)
                    if violation > max_violation:
                        max_violation = violation
                        i2 = j
                if i2 == -1:
                    continue

                # Filter out the selected alphas
                remaining_mask = np.ones_like(self._alpha, dtype=bool)
                remaining_mask[[i1, i2]] = False
                remaining_alpha = self._alpha[remaining_mask]
                remaining_y = y[remaining_mask]

                # Set alpha2 from alpha1 to satisfy constraints
                zeta = -np.sum(remaining_alpha * remaining_y)
                # alpha2 = (1 / y[i1]) * (zeta - alpha[i1] * y[i1])

                # Set lower and upper bounds for alpha1
                if y[i1] == y[i2]:
                    lower_bound = np.max([0., -y[i1] * (y[i2] * C - zeta)])
                    upper_bound = np.min([C, y[i1] * zeta])
                else:
                    lower_bound = np.max([0., y[i1] * zeta])
                    upper_bound = np.min([C, -y[i1] * (y[i2] * C - zeta)])

                # Compute the extremum of alpha1
                a11 = (y[i1]**2 * self._kernel(X[i1], X[i1])) / -2.
                a1221 = ((y[i1]**2 + y[i2]**2) * self._kernel(X[i1], X[i2])) / 2.
                a22 = (y[i1]**2 * self._kernel(X[i2], X[i2])) / -2.
                b11 = 0
                b1221 = zeta * (y[i1] + y[i2]) * self._kernel(X[i1], X[i2]) / -.2
                b22 = y[i1] * zeta * self._kernel(X[i2], X[i2])
                a = a11 + a1221 + a22
                b = b11 + b1221 + b22
                # Should not matter for extrema or maxima
                c = 0
                quadratic_f = lambda alpha_i: a * alpha_i**2 + b * alpha_i + c
                extremum = -b / (2 * a)

                # Use it as alpha if it is within bounds or 
                # use the bounds themselfes instead
                if extremum > lower_bound and extremum < upper_bound:
                    potential_maxima = np.array([extremum, lower_bound, upper_bound])
                else:
                    potential_maxima = np.array([lower_bound, upper_bound])
                self._alpha[i1] = potential_maxima[np.argmax(quadratic_f(potential_maxima))]

                # Calculate weights and biases from the lagrangian
                u = y * self._alpha / 2.
                self.weights = np.sum(np.diag(u) @ X, axis=0)
                self.bias = np.median(y - self.weights.T @ X.T)
                
                print(f"Iter {i1 + 1}/{n}: weights: {self.weights}  bias: {self.bias:.2f}")

                # Stop when conditions are satisfied
                satisfied = True
                for i in range(n):
                    val = y[i] * self.weights.T @ X[i] + self.bias
                    if self._alpha[i] == 0:
                        if not val >= 1 - EPSILON:
                            satisfied = False 
                            break 
                    elif self._alpha[i] == C:
                        if not val <= 1 + EPSILON:
                            satisfied = False 
                            break 
                    else: 
                        if not (val >= 1 - EPSILON and val <= 1 + EPSILON):
                            satisfied = False 
                            break
                if satisfied:
                    break

    def fit(self, X: ndarray, y: ndarray, C = 1.):
        """ 
        Fit the SVM by optimising for the lagrangians alpha in
        
        max( sum(alpha) - (1/2 alpha * y).T @ _kernel(X) )
        with 0 \le alpha_i \le C forall i; sum(y * alpha) = 0
        """
        d = X.shape[1]
        self.weights = np.zeros(d)
        self.bias = 0.
        self._smo(X, y, C)
        

    def predict(self, X: ndarray):
        """
        Predict the labels y for and input matrix X of shape n times d
        """
        if self.weights is None:
            raise ValueError("Fit the SVM before predicting") 
        
        f = lambda x_i: np.sign(self.weights.T @ x_i + self.bias)
        return np.apply_along_axis(f, 1, X)

    def _kernel(self, x_i: ndarray, x_j: ndarray):
        return np.dot(x_i, x_j)
    
    def hyperplane(self, x_0):
        if not self.weights.shape[0] == 2:
            raise ValueError("Hyperplane only works for 2D")
        
        # If weights are zero, there is no hyperplane seperating the data
        if np.all(self.weights == 0.):
            print("Weights are zero, no hyperplane exists between the data")
            return None
        
        # If w1 is zero and w0 is not: swap the axes
        if self.weights[1] == 0.:
            print("w1 is zero, returning hyperplane with respect to x_1 instead")
            return - (self.weights[1] * x_0 + self.bias) / self.weights[0]
        
        return - (self.weights[0] * x_0 + self.bias) / self.weights[1]

if __name__ == "__main__":
    # Dummy data
    X = np.array([[1, 1], [2, 1], [3, 2], [2, 3], [3, 3], [1, 3]])
    y = np.array([-1, -1, -1, 1, 1, 1])

    from matplotlib import pyplot as plt
    plt.scatter(X[:, 0], X[:, 1], c=y)
    # plt.show()

    svm = SVM()
    svm.fit(X, y)
    
    xs = np.linspace(1., 3.)
    ys = svm.hyperplane(xs)
    plt.plot(xs, ys)
    plt.show()

    new_data = np.array([[4, 3], [1, 2]])
    predictions = svm.predict(new_data)
    print(predictions)