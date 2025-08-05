from leapssn import *
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

"""
Support vector classification problem of Section 7.3.

Given vectors {x_1,...,x_l} ⊂ R^n each of which is classifed
by y_i = 1 or -1, the goal is to find w ∈ R^n and b > 0 such that
the hyperplane w' * x + b splits the data into its two respective 
communities. We model this by minimising

    0.5*‖w‖^2 + C ∑_{i=1}^l max(0, 1-y(w' * x_i))^2.
"""

class SupportVectorClassification(LeapSSN):
    def __init__(self, x, y, C):
        self.x = x # matrix with rows corresponding to [x_i; 1]
        self.y = y
        self.C = C

    def func(self, z):
        # z = [w;b]
        w = z[:-1]
        J = 0.5*np.dot(w,w)
        J += self.C * np.sum(np.maximum(1 - self.y * (self.x @ z), 0) ** 2)
        return J

    def grad(self, z):
        # z = [w;b]
        x, y, C = self.x, self.y, self.C
        a = 1 - y * (x @ z)
        active = a > 0

        gradf = np.zeros_like(z)
        gradf[:-1] = z[:-1]
        gradf += -2 * C * (x[active].T @ (y[active] * a[active]))
        return gradf
        
    def hess(self, z):
        # z = [w;b]
        x, y, C = self.x, self.y, self.C
        a = 1 - y * (x @ z)
        active = a > 0
        x_active = x[active]
        n = z.shape[0] - 1

        H = np.eye(z.shape[0])
        H[-1,-1] = 0.0
        H += 2 * C * (x_active.T @ x_active)
        return H
    
    def prox_solve(self):
        w_0 = 0.5*np.ones(x.shape[1])
        w_k, status, history = self.leapssn(w_0, n_iters=100, adaptive_search=True, 
                                            Lambda_0=3*np.linalg.norm(self.x)*self.C, alpha=1e-1, beta=1e-1,
                                            trace=True, grad_tol= 1e-5, adaptive_search_max_iter=100)
        return w_k, history['matrix_inverses'][-1]

if __name__ == "__main__":

    # Generate a classification set with vectors of size n
    def generate_data(n):
        if n == 2:
            X, y = make_classification(
                n_samples=10_000,
                n_features=2,
                n_informative=2,
                n_redundant=0,
                n_clusters_per_class=1,
                flip_y=0.2,
                class_sep=1.5,
                random_state=43,
            )
        else:
            X, y = make_classification(
                n_samples=10_000,
                n_features=n,
                n_clusters_per_class=1,
                flip_y=0.2,
                class_sep=1.5,
                random_state=43,
            )
        y[y==0] = -1 # make_classification gives y in {0,1} rather than {-1,1}
        x = np.hstack((X.copy(), np.ones((X.shape[0], 1))))
        return x,y

    svc_history = defaultdict(list)
    for n in [2,20,200,2000]:
        for C in [1e-4, 1e-2, 1e0, 1e2, 1e4]:
            x,y = generate_data(n)
            svc = SupportVectorClassification(x,y,C)
            w_k, iters = svc.prox_solve()
            svc_history["linear_system_solves_n_%s"%n].append(iters)

    print(svc_history)
    # Plotting
    if False:
        x,y = generate_data(2)
        C = 1e2
        svc = SupportVectorClassification(x,y,C)
        w_k, iters = svc.prox_solve()

        xx = np.linspace(x_min, x_max)
        yy =  -(w_k[0] * xx + w_k[2]) / w_k[1]

        plt.rcParams.update({"text.usetex": True,"font.family": "serif","font.serif": ["Computer Modern Roman"],})
        plt.close()
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap='bwr', edgecolors='k')
        plt.plot(xx, yy, 'k-')

        plt.xlabel(r"$x_1$", fontsize=20)
        plt.ylabel(r"$x_2$", fontsize=20)
        plt.grid(True)
        plt.savefig("svc_sol.pdf")