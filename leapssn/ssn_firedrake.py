from firedrake import *
from .utils import suppress_stdout
import numpy as np
from collections import defaultdict

class FiredrakeLeapSSN(object):

    def __init__(self, mesh):
        self.initialize_glob_prox_newton_class(mesh)

    def initialize_glob_prox_newton_class(self, mesh):
        self.lmbda = Constant(1.0)
        self.mesh = mesh
        self.V = self.function_space(mesh)
        self.v = TestFunction(self.V)
        self.w = TrialFunction(self.V)
        self.bc = self.boundary_conditions()
        self.riesz_solver = LinearSolver(assemble(self.inner_product(self.v, self.w), bcs=self.bc), solver_parameters={"ksp_type": "preonly","pc_type": "lu","pc_factor_mat_solver_type": "mumps"})
        self.R = Function(self.V)

    def function_space(self, mesh):
        raise NotImplementedError

    def objective(self, u):
        raise NotImplementedError

    def residual(self, u):
        return derivative(self.objective(u), u, self.v)
    
    def inner_product(self, u):
        raise NotImplementedError

    def boundary_conditions(self):
        return []
    
    def nvp(self, u):
        F = self.residual(u)
        bc = self.bc
        return NonlinearVariationalProblem(F, u, bcs=bc)

    def newton_solver(self, u):
        nvp = self.nvp(u)
        sp = self.newton_solver_parameter()
        return NonlinearVariationalSolver(nvp, solver_parameters=sp, options_prefix="")

    def newton_l2_solver(self, u):
        nvp = self.nvp(u)
        sp = self.newton_l2_solver_parameter()
        return NonlinearVariationalSolver(nvp, solver_parameters=sp, options_prefix="")
    
    def newton_bt_solver(self, u):
        nvp = self.nvp(u)
        sp = self.newton_bt_solver_parameter()
        return NonlinearVariationalSolver(nvp, solver_parameters=sp, options_prefix="")
       
    def newton_solver_parameter(self):
        sp = {
            "snes_monitor": None,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "snes_divergence_tolerance": 1e10,
            "snes_max_it": 200,
            "snes_rtol": 0.0,
        }
        return sp
    
    def newton_l2_solver_parameter(self):
        sp = {
            "snes_monitor": None,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "snes_divergence_tolerance": 1e10,
            "snes_max_it": 200,
            "snes_rtol": 0.0,
            "snes_stol": 0.0,
            "snes_linesearch_type": "l2",
            "snes_atol": 1e-8,
        }
        return sp
    
    def newton_bt_solver_parameter(self):
        sp = {
            "snes_monitor": None,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "snes_divergence_tolerance": 1e10,
            "snes_max_it": 200,
            "snes_rtol": 0.0,
            "snes_stol": 0.0,
            "snes_linesearch_type": "bt",
            "snes_atol": 1e-8,
        }
        return sp
    
    def proximal_solver_parameters(self):
        sp = {
            "snes_monitor": None,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "snes_max_it": 1,
            "snes_atol": 1e10,
            "snes_force_iteration": None}
        return sp
        
    def dual_norm(self, F):
        # This implements the discrete dual norm corresponding to the inner_product for this class
        self.riesz_solver.solve(self.R, assemble(F, bcs=self.bc)) # Invert Riesz map
        return sqrt(assemble(action(F,self.R), bcs=self.bc)) # Apply F to the inversion
        # return np.linalg.norm(assemble(F,bcs=bc).dat.data_ro) # (incorrect) Eucledian dual-norm


    def leapssn(self, u, proximal_parameters=(1/2,1/4,1.0), atol=1e-8, outer_max_its=100, inner_max_its=100, store_iterates=False):

        iters = 0
        history = defaultdict(list)
        lmbda = self.lmbda

        J = self.objective(u)
        F = self.residual(u)

        H = derivative(F, u, self.w) + lmbda*self.inner_product(self.v, self.w)

        bc = self.bc
        sp = self.proximal_solver_parameters()

        u_, d = Function(self.V), Function(self.V)
        alpha, beta, Lambda_k = proximal_parameters
        Lambda_k = Constant(Lambda_k)


        J_ = replace(J, {u: u_})
        F_ = replace(F, {u: u_})
        H_ = replace(H, {u: u_})
        
        history["objective_value"].append(assemble(J))
        history["gradient_dual_norm"].append(self.dual_norm(F))

        if store_iterates:
            with suppress_stdout():
                v = Function(self.V)
                v.assign(u)
                history["iterates"].append(v)


        nvp = NonlinearVariationalProblem(F_, u_, J=H_, bcs=bc)
        nvs = NonlinearVariationalSolver(nvp, solver_parameters=sp, options_prefix="")

        for k in range(outer_max_its):
            # Since PETSc measures the residual with the Euclidean norm, we also do it here.
            if np.linalg.norm(assemble(F,bcs=bc).dat.data_ro) < atol:
                break
            for j in range(inner_max_its):
                with suppress_stdout():
                    lmbda.assign(2**(j) * Lambda_k)
                    u_.assign(u)
                print("lmbda=%s."%lmbda.values())

                nvs.solve()
                iters += 1

                with suppress_stdout():
                    d.assign(u_ - u)
                c1 = assemble(-action(F_, d), bcs=bc)

                dual_norm_F = self.dual_norm(F_)
                c2 = alpha *  dual_norm_F**2 / lmbda.values()
                c3 = assemble(J-J_)
                c4 = beta * lmbda.values() * assemble(self.inner_product(d,d))

                if (c1 >= c2 and c3 >= c4):
                    print("Update accepted.")
                    with suppress_stdout():
                        Lambda_k.assign(lmbda / 2)
                        u.assign(u_)
                    history["accepted_lmbda_value"].append(np.copy(lmbda.values()))
                    history["objective_value"].append(assemble(J))
                    history["gradient_dual_norm"].append(dual_norm_F)

                    if store_iterates:
                        with suppress_stdout():
                            v = Function(self.V)
                            v.assign(u)
                            history["iterates"].append(v)

                    break
        history["linear_system_solves"] = iters
        return history