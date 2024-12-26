import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.spatial import ConvexHull
from shapely.geometry import Point
import alphashape
from shapely.geometry import Polygon

def solve_laplace_outside_hexagonal(nx, ny, tol, hex_mask=None):
    """
    Solve the Laplace equation on a 2D grid outside a hexagonal boundary.
    """
    u = np.zeros((ny, nx))

    if hex_mask is None:
        hex_mask = np.zeros((ny, nx), dtype=bool)
        center_x, center_y = nx // 2, ny // 2
        radius = min(nx, ny) // 8

        for i in range(ny):
            for j in range(nx):
                dx = j - center_x
                dy = i - center_y
                q = (2 / 3) * dx
                r = (-1 / 3) * dx + (np.sqrt(3) / 3) * dy
                if abs(q) <= radius and abs(r) <= radius and abs(q + r) <= radius:
                    hex_mask[i, j] = True

    u[hex_mask] = 1.0
    u[~hex_mask] = 0.0

    error = tol + 1
    while error > tol:
        u_old = u.copy()
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                if not hex_mask[i, j]:
                    u[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1])

        error = np.max(np.abs(u - u_old))

    return u, hex_mask

def compute_normal_vector(i, j, mask, nx, ny, radius=1):
    """
    Compute the normal vector at a boundary point (i, j), considering a larger neighborhood.
    """
    normal_x, normal_y = 0, 0
    count = 0
    
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            ni, nj = i + di, j + dj
            if 0 <= ni < ny and 0 <= nj < nx and mask[ni, nj]:
                normal_x += (nj - j)
                normal_y += (ni - i)
                count += 1

    if count > 0:
        length = np.sqrt(normal_x**2 + normal_y**2)
        if length > 0:
            normal_x /= length
            normal_y /= length

    return normal_x, normal_y

def grow_boundary_proportional_to_gradient(u, hex_mask, gx, gy, nx, ny, growth_factor=1000):
    """
    Grow the boundary away from the origin proportional to the gradient magnitude.
    """
    new_mask = hex_mask.copy()
    center_x, center_y = nx // 2, ny // 2

    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            if hex_mask[i, j]:
                grad_magnitude = np.sqrt(gx[i, j]**2 + gy[i, j]**2)
                n_x, n_y = compute_normal_vector(i, j, hex_mask, nx, ny, radius=5)

                if grad_magnitude > 0:
                    dx = j - center_x
                    dy = i - center_y
                    distance = np.sqrt(dx**2 + dy**2)

                    if distance > 0:
                        dx /= distance
                        dy /= distance

                    move_distance = growth_factor * grad_magnitude
                    new_x = int(j - n_x * move_distance)
                    new_y = int(i - n_y * move_distance)

                    if 0 <= new_x < nx and 0 <= new_y < ny:
                        new_mask[new_y, new_x] = True

    return new_mask

def fill_concave_hull(mask, nx, ny, alpha=0.1):
    """
    Compute the concave hull (alpha shape) of the boundary mask and fill it.
    """
    y, x = np.where(mask)
    points = np.column_stack((x, y))

    if len(points) > 2:
        alpha_shape = alphashape.alphashape(points, alpha)

        # Create a polygon and fill it
        if alpha_shape.geom_type == 'Polygon':
            filled_mask = np.zeros_like(mask, dtype=bool)
            # Create the filled polygon mask
            polygon = alpha_shape
            minx, miny, maxx, maxy = polygon.bounds
            for i in range(int(miny), int(maxy)):
                for j in range(int(minx), int(maxx)):
                    if polygon.contains(Point(j, i)):
                        filled_mask[i, j] = True
        elif alpha_shape.geom_type == 'MultiPolygon':
            filled_mask = np.zeros_like(mask, dtype=bool)
            # Union of all polygons and fill the region
            for poly in alpha_shape:
                polygon = poly
                minx, miny, maxx, maxy = polygon.bounds
                for i in range(int(miny), int(maxy)):
                    for j in range(int(minx), int(maxx)):
                        if polygon.contains(Point(j, i)):
                            filled_mask[i, j] = True

        return filled_mask

    else:
        return mask  # Return the original mask if no valid concave hull is found

def compute_gradient_magnitude(solution):
    """
    Compute the magnitude of the gradient field of the solution.
    """
    gy, gx = np.gradient(-solution)
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    return gradient_magnitude

# Parameters
nx, ny = 200, 200
tol = 5e-5

# Solve the Laplace equation
updated_solution, updated_mask = solve_laplace_outside_hexagonal(nx, ny, tol)

for i in range(10):
    gy, gx = np.gradient(-updated_solution)
    expanded_mask = grow_boundary_proportional_to_gradient(updated_solution, updated_mask, gx, gy, nx, ny, 200)
    filled_mask = fill_concave_hull(expanded_mask, nx, ny, alpha=0.2)

    plt.figure(figsize=(8, 6))
    plt.imshow(filled_mask, cmap='gray', origin='lower', extent=[0, 1, 0, 1])
    plt.colorbar(label='Boundary (1: Inside, 0: Outside)')
    plt.title('Hexagonal Boundary Mask After Growth and Filling the Gap')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    updated_solution, updated_mask = solve_laplace_outside_hexagonal(nx, ny, tol, filled_mask)

    plt.figure(figsize=(8, 6))
    plt.imshow(updated_solution, cmap='hot', origin='lower', extent=[0, 1, 0, 1])
    plt.colorbar(label='Potential')
    plt.title('Updated Solution of the Laplace Equation with Filled Boundary')
    plt.xlabel('x')
    plt.ylabel('y')

    y, x = np.meshgrid(range(ny), range(nx), indexing='ij')
    plt.contour(x / nx, y / ny, filled_mask, levels=[0.5], colors='cyan', linewidths=2, linestyles='--')
    plt.show()

    gradient_magnitude = compute_gradient_magnitude(updated_solution)

    plt.figure(figsize=(6, 6))
    plt.imshow(gradient_magnitude, origin='lower', extent=(-1, 1, -1, 1), cmap='viridis')
    plt.colorbar(label='Gradient Magnitude')
    plt.title('Gradient Magnitude (Electric Field Strength)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

