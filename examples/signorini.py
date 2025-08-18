
from firedrake import *
from leapssn import *
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse

"""
Signorini contact problem of Section 7.1.

We consider the Signorini problem on a rectangular
domain (0,5)x(0,1). The left- and right-hand sides
are clamped and the beam experiences a body force
due to gravity. It is constrained to live above an
obstacle.

We enforce the box constraints on the contact boundary
with a Moreau-Yosida regularisation. The solution is
discretized with continuous piecewise linear FEM.
"""

class SignoriniProblem(FiredrakeLeapSSN):

    def __init__(self, mesh, gamma):
        self.initialize_glob_prox_newton_class(mesh)
        self.gamma = gamma
    
    def function_space(self, mesh):
        return VectorFunctionSpace(mesh, "Lagrange", 1)
    
    def plus(self, x):
        return conditional(gt(x, 0), x, 0)

    def epsilon(self, u):
        return sym(grad(u))

    def inner_product (self, v,w):
        return inner(grad(v), grad(w))*dx + inner(v,w)*dx

    def obstacle(self, x):
        return as_vector([0,-0.5-0.4*sin(pi*x[0])])

    def objective(self, u):
        E = 200.0
        nu = 0.3
        lmbda_  = (E*nu)/((1+nu)*(1-2*nu))
        mu_ = E/(2*(1+nu))
        gamma = self.gamma

        f = Function(self.V)
        f.interpolate(Constant((0,-10)))

        n = FacetNormal(self.mesh)
        D = Identity(2) + grad(u)  # deformation gradient
        n_def_unnorm = dot(transpose(inv(D)), n)
        n_def = n_def_unnorm / sqrt(dot(n_def_unnorm, n_def_unnorm))

        x = SpatialCoordinate(self.mesh)

        J = (
            mu_*inner(self.epsilon(u), self.epsilon(u))*dx 
            + 0.5*lmbda_*inner(div(u), div(u))*dx - inner(f,u)*dx 
            + gamma * self.plus(dot(u-self.obstacle(x),n_def))**2*ds(3)
        )
        return J
    
    def boundary_conditions(self):
        return DirichletBC(self.V, Constant((0,0)), [1,2])

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true",)

    mesh = RectangleMesh(300, 40, 5, 1)
    problem=SignoriniProblem(mesh, Constant(1e3))
    u = Function(problem.V)

    signorini_newton_history = defaultdict(list)

    u.assign(0)
    try:
        problem.newton_solver(u)
    except:
        pass

    solver = problem.newton_l2_solver(u)
    for g in [1e3,1e4]:
        problem.gamma.assign(g)
        u.assign(0)
        try:
            solver.solve()
            signorini_newton_history["l2_solves"].append(solver.snes.its)
        except:
            pass

    solver = problem.newton_bt_solver(u)
    for g in [1e3,1e4,1e5]:
        problem.gamma.assign(g)
        u.assign(0)
        try:
            solver.solve()
            signorini_newton_history["bt_solves"].append(solver.snes.its)
        except:
            pass

    signorini_prox_history = defaultdict(list)
    proximal_parameters = (0.5,0.25,1.0) # alpha, beta, Lambda_0
    for g in [1e3,1e4,1e5,1e6]:
        problem.gamma.assign(g)
        u.assign(0)
        history = problem.leapssn(u, proximal_parameters, store_iterates=True)
        signorini_prox_history["linear_system_solves"].append(history["linear_system_solves"])
        signorini_prox_history["objective_value"].append(history["objective_value"])
        signorini_prox_history["iterates"].append(history["iterates"])
        signorini_prox_history["accepted_lmbda_value"].append(history["accepted_lmbda_value"])


    print("l2-Newton linear system solves: %s" %(signorini_newton_history["l2_solves"]))
    print("Backtracking Newton linear system solves: %s" %(signorini_newton_history["bt_solves"]))
    print("LeAP SSN linear system solves: %s" %(signorini_prox_history["linear_system_solves"]))

    ## Plotting
    args = parser.parse_args()
    if args.plot:
        plt.rcParams.update({"text.usetex": True,"font.family": "serif","font.serif": ["Computer Modern Roman"],})
        plt.close()
        gs = ["10^3", "10^4", "10^5", "10^6"]
        ms = ["D", "s", "o", "x"]
        for i in range(4):
            x = signorini_prox_history["objective_value"][i] - signorini_prox_history["objective_value"][i][-1]
            plt.plot(range(1,len(x)-1), x[1:-1], "%s-"%(ms[i]), label=r"$\gamma=%s$"%(gs[i]))
        D_0 = norm(signorini_prox_history["iterates"][3][-1], "H1")
        lmbda_prod = np.prod(signorini_prox_history["accepted_lmbda_value"][3])**(1/len(signorini_prox_history["accepted_lmbda_value"][3]))
        y = np.array(4*lmbda_prod*D_0**2) / range(1,len(x)-1) #+ dnorm * D_0 * np.exp(- np.array(range(1,len(x)-1)) / 4)
        plt.plot(range(1,len(x)-1), y, "--", label=r"$4 (\Pi_{j=0}^{93} \lambda_j)^{1/94} \Vert u^* \Vert^2_{H^1}/ k $")
        plt.yscale('log')
        plt.xlabel(r"$k$", fontsize=20)
        plt.ylabel(r"$F(u_k)-F^*$", fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(True)
        plt.legend(loc='lower center', bbox_to_anchor=(0.7, 0.05), fontsize=12)
        plt.tight_layout()
        plt.savefig("signorini_convergence.pdf")

        problem.gamma.assign(1e6)
        u.assign(0)
        problem.leapssn(u, proximal_parameters)
        E = 200.0
        nu = 0.3
        lmbda_  = (E*nu)/((1+nu)*(1-2*nu))
        mu_ = E/(2*(1+nu))
        def sigma(u):
            return lmbda_*tr(problem.epsilon(u))*Identity(2) + 2.0*mu_*problem.epsilon(u)
        W = TensorFunctionSpace(mesh, "CG", 1)
        stress = project(sigma(u), W, name="Stress")
        VTKFile("signorini_sol.pvd").write(u, stress)

        width = 0.25
        mesh = RectangleMesh(300, 40, 5, width)
        V = VectorFunctionSpace(mesh, "Lagrange", 1)
        ob = Function(V)
        def obstacle_v():
            return as_vector([0,problem.obstacle(x)[1]-width])
        ob.interpolate(obstacle_v())
        VTKFile("signorini_obstacle.pvd").write(ob)
