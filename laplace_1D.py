import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

def sample_polygon(polygon, density=100):
    """
    Sample the edges of a polygon at high density, excluding duplicate consecutive points.

    Parameters:
    polygon (list of tuples): List of (x, y) vertices of the polygon.
    density (int): Number of points to sample between each pair of vertices.

    Returns:
    list of tuples: High-density sampled points along the polygon.
    """
    sampled_points = []
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]  # Wrap around to the first point
        if p1 == p2:
            continue
        x_vals = np.linspace(p1[0], p2[0], density)
        y_vals = np.linspace(p1[1], p2[1], density)
        sampled_points.extend(zip(x_vals, y_vals))
    return sampled_points

def solve_laplace(boundary_polygon, boundary_value, grid_size=100, tol=1e-6):
    """
    Solve the Laplace equation numerically on a 2D grid with a constant value inside a polygonal boundary.

    Parameters:
    boundary_polygon (list of tuples): List of (x, y) specifying the vertices of the polygon in counterclockwise order.
    boundary_value (float): Value assigned to the boundary and inside the polygon.
    grid_size (int): Size of the grid (NxN).
    tol (float): Convergence tolerance.

    Returns:
    np.array: 2D solution array.
    """
    # Initialize the grid
    N = grid_size
    u = np.zeros((N, N))

    # Map grid points to polygon region
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    xv, yv = np.meshgrid(x, y, indexing='ij')

    polygon_path = Path(boundary_polygon)
    points = np.vstack((xv.ravel(), yv.ravel())).T
    inside_polygon = polygon_path.contains_points(points).reshape(N, N)

    # Apply boundary and interior conditions
    u[inside_polygon] = boundary_value

    # Iterative solver (Gauss-Seidel method)
    converged = False
    while not converged:
        u_old = u.copy()

        # Update interior points
        for i in range(1, N-1):
            for j in range(1, N-1):
                if not inside_polygon[i, j]:
                    u[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1])

        # Check for convergence
        diff = np.max(np.abs(u - u_old))
        if diff < tol:
            converged = True

    return u

def generate_regular_hexagon(center, radius):
    """
    Generate vertices of a regular hexagon.

    Parameters:
    center (tuple): (x, y) coordinates of the hexagon's center.
    radius (float): Distance from the center to any vertex.

    Returns:
    list of tuples: Vertices of the hexagon in counterclockwise order.
    """
    cx, cy = center
    vertices = []
    for i in range(6):
        angle = 2 * np.pi * i / 6
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        vertices.append((x, y))
    return vertices

def calculate_boundary_movement(boundary_polygon, solution, grid_size, growth_rate=1.0):
    """
    Calculate the movement of the boundary proportional to the gradient and in the normal direction.

    Parameters:
    boundary_polygon (list of tuples): List of (x, y) vertices of the polygon.
    solution (np.array): 2D solution array from Laplace solver.
    grid_size (int): Size of the grid (NxN).
    growth_rate (float): Factor controlling the speed of boundary movement.

    Returns:
    list of tuples: New positions of the boundary vertices.
    """
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xv, yv = np.meshgrid(x, y, indexing='ij')

    dx, dy = np.gradient(solution, x, y)

    new_boundary = []
    for i, (x, y) in enumerate(boundary_polygon):
        # Find grid indices closest to the boundary point
        ix = min(max(int(x * (grid_size - 1)), 0), grid_size - 1)
        iy = min(max(int(y * (grid_size - 1)), 0), grid_size - 1)

        # Calculate gradient at the boundary point
        grad_x = dx[ix, iy]
        grad_y = dy[ix, iy]

        # Compute normal direction (left-hand rule: normal = (-dy, dx))
        prev_idx = (i - 1) % len(boundary_polygon)
        next_idx = (i + 1) % len(boundary_polygon)

        prev_point = boundary_polygon[prev_idx]
        next_point = boundary_polygon[next_idx]

        tangent = np.array([next_point[0] - prev_point[0], next_point[1] - prev_point[1]])
        normal = np.array([-tangent[1], tangent[0]])
        normal /= np.linalg.norm(normal)

        # Update position proportional to gradient, normal, and growth rate
        movement = growth_rate * np.sqrt(grad_x**2 + grad_y**2)
        new_x = x - movement * normal[0]
        new_y = y - movement * normal[1]
        new_boundary.append((new_x, new_y))

    return new_boundary

# Example usage
if __name__ == "__main__":
    # Define regular hexagon
    center = (0.5, 0.5)
    radius = 0.3
    boundary_polygon = generate_regular_hexagon(center, radius)

    high_density_polygon = sample_polygon(boundary_polygon, density=1000)  # High density right from the start

    boundary_value = 1.0
    grid_size = 200  # High grid resolution
    growth_rate = 0.01

    # Solve the Laplace equation
    solution = solve_laplace(high_density_polygon, boundary_value, grid_size=grid_size)

    # Calculate new boundary positions
    new_boundary_polygon = calculate_boundary_movement(high_density_polygon, solution, grid_size, growth_rate)

    # Plot the solution
    plt.figure(figsize=(8, 6))
    plt.imshow(solution, extent=[1, 0, 1, 0], origin='lower', cmap='viridis')
    plt.colorbar(label='Potential')
    plt.title('Numerical Solution of Laplace Equation (Regular Hexagon)')
    plt.xlabel('x')
    plt.ylabel('y')

    # Plot the original and new boundary
    original_boundary = np.array(high_density_polygon)  # High density boundary
    new_boundary = np.array(new_boundary_polygon)

    plt.plot(original_boundary[:, 0], original_boundary[:, 1], 'r--', label='Original Boundary')
    plt.plot(new_boundary[:, 0], new_boundary[:, 1], 'b-', label='Updated Boundary')
    plt.legend()
    plt.show()

