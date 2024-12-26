import numpy as np
import matplotlib.pyplot as plt

def solve_laplace(nx, ny, tol):
    """
    Solve the Laplace equation on a 2D grid.

    Parameters:
        nx (int): Number of grid points in the x-direction.
        ny (int): Number of grid points in the y-direction.
        tol (float): Convergence tolerance.

    Returns:
        np.ndarray: Solution array.
    """
    # Initialize the grid
    u = np.zeros((ny, nx))

    # Boundary conditions
    u[0, :] = 1.0  # Top boundary
    u[-1, :] = 0.0  # Bottom boundary
    u[:, 0] = 0.0  # Left boundary
    u[:, -1] = 0.0  # Right boundary

    # Iterative solver (Gauss-Seidel method)
    error = tol + 1
    while error > tol:
        u_old = u.copy()

        # Update interior points
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                u[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1])

        # Calculate the error (maximum difference between iterations)
        error = np.max(np.abs(u - u_old))

    return u

# Parameters
nx, ny = 50, 50  # Grid size
tol = 1e-4       # Convergence tolerance

# Solve the equation
solution = solve_laplace(nx, ny, tol)

# Plot the result
plt.figure(figsize=(8, 6))
plt.imshow(solution, cmap='hot', origin='lower', extent=[0, 1, 0, 1])
plt.colorbar(label='Potential')
plt.title('Numerical Solution of the Laplace Equation')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

