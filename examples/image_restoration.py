
from firedrake import *
from leapssn import *
import numpy as np
from skimage.transform import rescale
from skimage.data import shepp_logan_phantom
from scipy.interpolate import RegularGridInterpolator

"""
Image restoration problem of Section 7.2.

We consider the Fenchel predual formulation of a BV-regularisation.
The box constraints are handled by a Moreau-Yosida penalty. The 
solution is discretised with Cartesian Raviart-Thomas elements.
The Shepp Logan image is projected to a DG0 space.
"""

class ImageProblem(FiredrakeLeapSSN):

    def __init__(self, mesh, sigma, alpha, epsilon, epsilon_grad, c, nx, ny):
        self.sigma = sigma # noise level
        self.epsilon = epsilon # parameter for L^2-norm regularisation
        self.epsilon_grad = epsilon_grad # parameter for (broken) H^1-semi norm regularisation
        self.alpha = alpha # parameter for BV-norm in primal problem
        self.c = c # Moreau-Yosida penalty
        self.nx = nx # size of image in x-direction
        self.ny = ny # size of image in y-direction
    
        self.initialize_glob_prox_newton_class(mesh)
        self.image()

    def function_space(self, mesh):
        # Cartesian first-order Raviart-Thomas
        return FunctionSpace(mesh, "RTCF", 1)

    def inner_product(self, v, w):
        return inner(grad(v), grad(w))*dx + inner(div(v),div(w))*dx + inner(v,w)*dx

    def plus(self, x):
        return as_vector([conditional(gt(x[0], 0), x[0], 0), conditional(gt(x[1], 0), x[1], 0)])

    def normalize(self, X):
        X = X - np.min(X)
        X = X / np.max(X)
        return X

    def image(self):
        # This method resizes the shepp_logan image, adds noise
        # and interpolates it into a DG0 space.
        nx, ny = self.nx, self.ny
        clean_image = shepp_logan_phantom()
        [nnx,nny] = np.array(clean_image.shape)
        desired_shape = np.array([nx,ny]) #desired_shape
        rescale_factor = desired_shape/[nnx,nny]
        clean_image = rescale(clean_image, rescale_factor)

        Y = FunctionSpace(self.mesh, "DQ", 0)
        f, g = Function(Y), Function(Y)

        W = VectorFunctionSpace(mesh, Y.ufl_element())
        X = assemble(firedrake.__future__.interpolate(mesh.coordinates, W))

        y = np.linspace(0, 1, ny)
        x = np.linspace(0, 1, nx)
        interpolator = RegularGridInterpolator((y, x), np.flip(clean_image.transpose(),1))
        g.dat.data[:] = interpolator(X.dat.data_ro)

        np.random.seed(0)  # make noise fixed.
        noisy_image = clean_image + self.sigma*np.random.randn(nx,ny)
        interpolator = RegularGridInterpolator((y, x), np.flip(noisy_image.transpose(),1))
        f.dat.data[:] = interpolator(X.dat.data_ro)

        self.f = f
        self.g = g

    def objective(self, p):
        f = self.f

        alpha = Constant((self.alpha,self.alpha))
        c = self.c
        epsilon = self.epsilon
        epsilon_grad = self.epsilon_grad

        J = (
            0.5*inner(div(p)+f, div(p)+f)*dx
            + epsilon * inner(p,p)*dx
            + epsilon_grad * inner(grad(p), grad(p))*dx
            + c/(2) * inner(self.plus((p - alpha)),self.plus((p - alpha)))*dx
            + c/(2) * inner(self.plus(-(p + alpha)),self.plus(-(p + alpha)))*dx
        )
        return J
    
    def boundary_conditions(self):
        return DirichletBC(self.V, Constant((0,0)), "on_boundary")

    
class ImageProblemCG(ImageProblem):
    # This uses bilinear continuous FEM instead
    def function_space(self, mesh):
        return VectorFunctionSpace(mesh, "Q", 1)
    def boundary_conditions(self):
        bc1 = DirichletBC(self.V.sub(0), Constant(0), 1)
        bc2 = DirichletBC(self.V.sub(0), Constant(0), 2)
        bc3 = DirichletBC(self.V.sub(1), Constant(0), 3)
        bc4 = DirichletBC(self.V.sub(1), Constant(0), 4)
        bcs = [bc1, bc2, bc3, bc4]
        return bcs

if __name__ == "__main__":

    nx, ny = 400, 400 # pixels in x and y-direction
    alpha = 1e-4 # parameter for BV-norm in primal problem
    epsilon = Constant(0)  # parameter for L^2-norm regularisation
    epsilon_grad = Constant(1e-1) # parameter for (broken) H^1-semi norm regularisation
    c = Constant(1e7) # Moreau-Yosida penalty
    sigma = 0.06 # noise level
    mesh = UnitSquareMesh(nx, ny, quadrilateral=True)

    problem=ImageProblem(mesh, sigma, alpha, epsilon, epsilon_grad, c, nx, ny)
    p = Function(problem.V)

    image_newton_history = defaultdict(list)

    solver = problem.newton_solver(p)
    for c_val in [1e4, 1e5, 1e6]:
        problem.c.assign(c_val)
        p.assign(alpha)
        solver.solve()
        image_newton_history["newton_solves"].append(solver.snes.its)

    bt_solver = problem.newton_bt_solver(p)
    l2_solver = problem.newton_l2_solver(p)
    for c_val in [1e4, 1e5, 1e6, 1e7]:
        problem.c.assign(c_val)
        p.assign(alpha)
        bt_solver.solve()
        image_newton_history["bt_solves"].append(bt_solver.snes.its)
        p.assign(alpha)
        l2_solver.solve()
        image_newton_history["l2_solves"].append(l2_solver.snes.its)

    image_prox_history = defaultdict(list)
    proximal_parameters = (1e-4,1e-4,2**6) # alpha, beta, Lambda_0
    for c_val in [1e4, 1e5, 1e6, 1e7]:
        p.assign(alpha)
        problem.c.assign(c_val)
        history = problem.leapssn(p, proximal_parameters, 1e-8, 200, 200)
        image_prox_history["linear_system_solves"].append(history["linear_system_solves"])

    f, g = problem.f, problem.g
    u = div(p)+f
    u0 = project(u, FunctionSpace(mesh, "DQ", 0))

    mse = assemble(inner(f-g, f-g)*dx) / assemble(Constant(1.0) * dx(mesh))
    psnr_f = 10 * np.log10(1 / mse)
    print("PSNR (f): %s"%psnr_f)

    mse = assemble(inner(u0-g, u0-g)*dx) / assemble(Constant(1.0) * dx(mesh))
    psnr_u = 10 * np.log10(1 / mse)
    print("PSNR (u): %s"%psnr_u)

    print("Newton linear system solves: %s" %(image_newton_history["newton_solves"]))
    print("l2-Newton linear system solves: %s" %(image_newton_history["l2_solves"]))
    print("Backtracking Newton linear system solves: %s" %(image_newton_history["bt_solves"]))
    print("LeAP SSN linear system solves: %s" %(image_prox_history["linear_system_solves"]))

    # Plotting
    if False:
        outfile = VTKFile("image_restoration.pvd")
        U = FunctionSpace(mesh, "DQ", 0)
        h = Function(U)
        h.rename("Image")
        h.project(g); outfile.write(h)
        h.project(f); outfile.write(h)
        h.project(u); outfile.write(h)
