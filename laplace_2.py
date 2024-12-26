import numpy as np
import matplotlib.pyplot as plt

def solve_laplace_outside_hexagonal(nx, ny, tol):
    """
    Solve the Laplace equation on a 2D grid outside a hexagonal boundary.

    Parameters:
        nx (int): Number of grid points in the x-direction.
        ny (int): Number of grid points in the y-direction.
        tol (float): Convergence tolerance.

    Returns:
        np.ndarray: Solution array.
    """
    # Initialize the grid
    u = np.zeros((ny, nx))

    # Define hexagonal boundary condition (solve outside the hexagon)
    hex_mask = np.zeros((ny, nx), dtype=bool)
    center_x, center_y = nx // 2, ny // 2
    radius = min(nx, ny) // 8

    for i in range(ny):
        for j in range(nx):
            dx = j - center_x
            dy = i - center_y
            # Convert to scaled hexagonal coordinates
            q = (2 / 3) * dx
            r = (-1 / 3) * dx + (np.sqrt(3) / 3) * dy
            if abs(q) <= radius and abs(r) <= radius and abs(q + r) <= radius:
                hex_mask[i, j] = True

    u[hex_mask] = 1.0  # Set boundary condition of the hexagon to 1
    u[~hex_mask] = 0.0  # Initialize outside the hexagon to 0

    # Iterative solver (Gauss-Seidel method)
    error = tol + 1
    while error > tol:
        u_old = u.copy()

        # Update exterior points
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                if not hex_mask[i, j]:
                    u[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1])

        # Calculate the error (maximum difference between iterations)
        error = np.max(np.abs(u - u_old))

    return u, hex_mask

# Parameters
nx, ny = 200, 200  # Grid size
tol = 1e-4       # Convergence tolerance

# Solve the equation
solution, hex_mask = solve_laplace_outside_hexagonal(nx, ny, tol)

# Plot the result
plt.figure(figsize=(8, 6))
plt.imshow(solution, cmap='hot', origin='lower', extent=[0, 1, 0, 1])
plt.colorbar(label='Potential')
plt.title('Numerical Solution of the Laplace Equation Outside Regular Hexagonal Boundary')
plt.xlabel('x')
plt.ylabel('y')

# Overlay the hexagonal mask
y, x = np.meshgrid(range(ny), range(nx), indexing='ij')
plt.contour(x / nx, y / ny, hex_mask, levels=[0.5], colors='cyan', linewidths=2, linestyles='--')

plt.show()


gy, gx = np.gradient(-solution)
plt.figure(figsize=(8, 6))
plt.streamplot(x / nx, y / ny, gx, gy, color=np.sqrt(gx**2 + gy**2), cmap='viridis', linewidth=1)
plt.colorbar(label='Field Intensity')
plt.title('Electric Field Lines Outside Regular Hexagonal Boundary')
plt.xlabel('x')
plt.ylabel('y')
plt.contour(x / nx, y / ny, hex_mask, levels=[0.5], colors='cyan', linewidths=2, linestyles='--')
plt.show()
