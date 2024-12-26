from dolfinx import mesh, fem
from mpi4py import MPI
import ufl
import numpy as np

# Step 1: Create the mesh for the unit square domain
domain = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)

# Step 2: Define the function space
V = fem.functionspace(domain, ("Lagrange", 1))  # Lagrange elements of degree 1

# Step 3: Define boundary conditions (V = 0 on all boundaries)
def boundary(x):
    return np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)) | \
           np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))

# Create boundary condition object
boundary_dofs = fem.locate_dofs_geometrical(V, boundary)
bc = fem.dirichletbc(value=fem.Constant(domain, 0.0), dofs=boundary_dofs)

# Step 4: Define the variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx  # Bilinear form
L = ufl.Constant(0.0) * v * ufl.dx  # Linear form

# Step 5: Solve the problem
problem = fem.petsc.LinearProblem(a, L, bcs=[bc])
u_sol = problem.solve()

# Step 6: Save the solution for visualization
with fem.io.XDMFFile(domain.comm, "solution.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u_sol)

print("Solution saved to 'solution.xdmf'. Open it with ParaView for visualization.")

