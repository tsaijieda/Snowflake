import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

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

    Parameters:
        i (int): y-coordinate of the boundary point.
        j (int): x-coordinate of the boundary point.
        mask (np.ndarray): Boundary mask (True for boundary points, False otherwise).
        nx (int): Grid width.
        ny (int): Grid height.
        radius (int): The size of the neighborhood to consider for calculating the normal vector.

    Returns:
        (dx, dy): Normalized direction vector of the normal at (i, j).
    """
    normal_x, normal_y = 0, 0
    count = 0
    
    # Iterate over a square neighborhood around (i, j)
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            ni, nj = i + di, j + dj
            # Check if the neighbor is within bounds and is part of the boundary
            if 0 <= ni < ny and 0 <= nj < nx and mask[ni, nj]:
                # Vector pointing from (i, j) to (ni, nj)
                normal_x += (nj - j)
                normal_y += (ni - i)
                count += 1
    
    # Normalize the normal vector
    if count > 0:
        length = np.sqrt(normal_x**2 + normal_y**2)
        if length > 0:
            normal_x /= length
            normal_y /= length

    return normal_x, normal_y

def grow_boundary_proportional_to_gradient(u, hex_mask, gx, gy, nx, ny, growth_factor = 500):
    """
    Grow the boundary away from the origin proportional to the gradient magnitude.
    
    Parameters:
        u (np.ndarray): Potential field.
        hex_mask (np.ndarray): Hexagonal boundary mask.
        gx (np.ndarray): Gradient in the x-direction.
        gy (np.ndarray): Gradient in the y-direction.
        nx (int): Number of grid points in the x-direction.
        ny (int): Number of grid points in the y-direction.
        growth_factor (float): Growth factor to control how much the boundary expands.
        
    Returns:
        np.ndarray: Updated hexagonal boundary mask.
    """
    new_mask = hex_mask.copy()
    
    # Get the center of the grid
    center_x, center_y = nx // 2, ny // 2

    # Iterate over the boundary points in the mask
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            if hex_mask[i, j]:  # If the point is on the current boundary
                # Compute the gradient magnitude
                grad_magnitude = np.sqrt(gx[i, j]**2 + gy[i, j]**2)

                n_x, n_y = compute_normal_vector(i, j, hex_mask, nx, ny, radius=10)
                # If the gradient is strong enough, move the boundary point outward
                if grad_magnitude > 0:
                    # Direction vector pointing away from the origin
                    dx = j - center_x
                    dy = i - center_y
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    # Normalize the direction vector
                    if distance > 0:
                        dx /= distance
                        dy /= distance
                    print(n_x, n_y) 
                    # Calculate the new position by moving the point outward
                    move_distance = growth_factor * grad_magnitude
                    new_x = int(j - n_x * move_distance)
                    new_y = int(i - n_y * move_distance)
                    
                    # Ensure the new point is within bounds
                    if 0 <= new_x < nx and 0 <= new_y < ny:
                        new_mask[new_y, new_x] = True  # Mark the new boundary point

    return new_mask

def fill_gap_between_boundaries(old_mask, new_mask):
    """
    Fill the gap between the old boundary and the new expanded boundary.
    
    Parameters:
        old_mask (np.ndarray): The original boundary mask.
        new_mask (np.ndarray): The expanded boundary mask.
        
    Returns:
        np.ndarray: Mask with the gap filled between old and new boundaries.
    """
    # Compute the Euclidean distance transform
    dist_transform = distance_transform_edt(~new_mask)
    
    # Fill the gap by setting all points between the old and new boundary to True
    gap_mask = np.logical_and(~old_mask, dist_transform <= 4)
    
    # Update the new mask with the filled gap
    new_filled_mask = np.logical_or(new_mask, gap_mask)
    
    return new_filled_mask

def is_inner_gap_point(i, j, center_x, center_y, old_mask, new_mask):
    """
    Check if a point is an inner gap point (toward the origin).
    
    Parameters:
        i (int): y-coordinate of the point.
        j (int): x-coordinate of the point.
        center_x (int): x-coordinate of the origin.
        center_y (int): y-coordinate of the origin.
        old_mask (np.ndarray): The original mask.
        new_mask (np.ndarray): The expanded mask.
        
    Returns:
        bool: True if the point lies in the inner gap, False otherwise.
    """
    # Check if the point is outside the old mask and inside the gap
    if old_mask[i, j] or new_mask[i, j]:
        return False

    # Compute the vector to the origin
    dx = center_x - j
    dy = center_y - i
    distance_to_origin = np.sqrt(dx**2 + dy**2)

    # Avoid division by zero
    if distance_to_origin == 0:
        return False

    # Find the nearest boundary in the new mask along the vector to the origin
    for step in np.linspace(0, distance_to_origin, 50):
        x = int(j + step * dx / distance_to_origin)
        y = int(i + step * dy / distance_to_origin)

        if (x == center_x and y == center_y) or new_mask[y, x]:  # If we hit the origin or boundary
            return True

    return False

# Example usage remains the same
def compute_gradient_magnitude(solution):
    """
    Compute the magnitude of the gradient field of the solution.

    Parameters:
        solution (np.ndarray): 2D array representing the scalar field solution.

    Returns:
        np.ndarray: Gradient magnitude at each point.
    """
    # Compute the gradients in both directions
    gy, gx = np.gradient(-solution)  # Negative gradient for electric field
    
    # Compute the magnitude of the gradient
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    
    return gradient_magnitude


def fill_inner_gap(old_mask, new_mask, origin):
    """
    Fill the inner gap between the old mask and the new mask.
    
    Parameters:
        old_mask (np.ndarray): The original boundary mask.
        new_mask (np.ndarray): The expanded boundary mask.
        origin (tuple): Coordinates of the origin (x, y).
        
    Returns:
        np.ndarray: Mask with the inner gap filled.
    """
    center_x, center_y = origin
    nx, ny = new_mask.shape[1], new_mask.shape[0]
    filled_mask = new_mask.copy()

    for i in range(ny):
        for j in range(nx):
            if is_inner_gap_point(i, j, center_x, center_y, old_mask, new_mask):
                filled_mask[i, j] = True

    return filled_mask

# Parameters
nx, ny = 200, 200  # Grid size
tol = 5e-5       # Convergence tolerance

# Solve the equation
updated_solution, updated_mask = solve_laplace_outside_hexagonal(nx, ny, tol)

for i in range(10):
    # Compute the gradient (field)
    gy, gx = np.gradient(-updated_solution)

    # Grow the boundary based on the gradient
    expanded_mask = grow_boundary_proportional_to_gradient(updated_solution, updated_mask, gx, gy, nx, ny, 100)

    # Fill the gap between the old and expanded boundaries
    filled_mask = fill_gap_between_boundaries(updated_mask, expanded_mask)
    #filled_mask = expanded_mask
    # Plot the filled boundary mask
    plt.figure(figsize=(8, 6))
    plt.imshow(filled_mask, cmap='gray', origin='lower', extent=[0, 1, 0, 1])
    plt.colorbar(label='Boundary (1: Inside, 0: Outside)')
    plt.title('Hexagonal Boundary Mask After Growth and Filling the Gap')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # Solve the Laplace equation again with the updated boundary
    updated_solution, updated_mask = solve_laplace_outside_hexagonal(nx, ny, tol, filled_mask)

    # Plot the updated potential field
    plt.figure(figsize=(8, 6))
    plt.imshow(updated_solution, cmap='hot', origin='lower', extent=[0, 1, 0, 1])
    plt.colorbar(label='Potential')
    plt.title('Updated Solution of the Laplace Equation with Filled Boundary')
    plt.xlabel('x')
    plt.ylabel('y')

    # Overlay the updated boundary mask
    y, x = np.meshgrid(range(ny), range(nx), indexing='ij')
    plt.contour(x / nx, y / ny, filled_mask, levels=[0.5], colors='cyan', linewidths=2, linestyles='--')
    plt.show()

    gradient_magnitude = compute_gradient_magnitude(updated_solution)

    # Visualize the gradient magnitude
    plt.figure(figsize=(6, 6))
    plt.imshow(gradient_magnitude, origin='lower', extent=(-1, 1, -1, 1), cmap='viridis')
    plt.colorbar(label='Gradient Magnitude')
    plt.title('Gradient Magnitude (Electric Field Strength)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

