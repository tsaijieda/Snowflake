import numpy as np
import matplotlib.pyplot as plt

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

# Example: Create a simple boundary mask (circle in this case)
nx, ny = 50, 50
mask = np.zeros((ny, nx), dtype=bool)
center_x, center_y = nx // 2, ny // 2
radius = 10

# Create a circular boundary
for i in range(ny):
    for j in range(nx):
        if (i - center_y)**2 + (j - center_x)**2 <= radius**2:
            mask[i, j] = True

# Now calculate the normal vector at each boundary point
normal_vectors = {}

for i in range(1, ny - 1):
    for j in range(1, nx - 1):
        if mask[i, j]:  # If it's a boundary point
            normal_x, normal_y = compute_normal_vector(i, j, mask, nx, ny, radius=2)
            normal_vectors[(i, j)] = (normal_x, normal_y)

# Visualize the normal vectors as little arrows
plt.figure(figsize=(8, 6))
plt.imshow(mask, cmap='gray', origin='lower', extent=[0, 1, 0, 1])
plt.colorbar(label='Boundary (1: Inside, 0: Outside)')
plt.title('Boundary Mask with Normal Vectors')

# Plot the normal vectors as small arrows
for (i, j), (normal_x, normal_y) in normal_vectors.items():
    plt.arrow(j / nx, i / ny, normal_x * 0.02, normal_y * 0.02, color='red', head_width=0.01, head_length=0.02)

plt.xlabel('x')
plt.ylabel('y')
plt.show()

