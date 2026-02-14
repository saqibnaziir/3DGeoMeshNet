import numpy as np
import os
import matplotlib.pyplot as plt
import trimesh

def load_mesh(file_path):
    try:
        mesh = trimesh.load(file_path)
        print(f"Successfully loaded mesh from {file_path}")
        print(f"Mesh vertices shape: {mesh.vertices.shape}")
        return mesh.vertices
    except Exception as e:
        print(f"Error loading mesh from {file_path}: {str(e)}")
        return None

def calculate_errors(pred, gt):
    error = np.linalg.norm(pred - gt, axis=1)
    mean_error = np.mean(error)
    std_error = np.std(error)
    median_error = np.median(error)
    return mean_error, std_error, median_error

# Define file paths
gt_path = "/home/nazir241/PosDoc/COSURAI/Codes/FaceCom/Results/Oct8/Mouthup/85.ply"
pred_path = "/home/nazir241/PosDoc/COSURAI/Codes/FaceCom/Results/Oct8/Mouthup/recon85.ply"
save_path = "/home/nazir241/PosDoc/COSURAI/Codes/FaceCom/Results/Oct8/Mouthup"

# Load meshes
print("Loading ground truth mesh...")
gt_vertices = load_mesh(gt_path)
print("Loading predicted mesh...")
pred_vertices = load_mesh(pred_path)

if gt_vertices is None or pred_vertices is None:
    print("Error: Failed to load one or both meshes. Exiting.")
    exit(1)

# Ensure the meshes have the same shape
if gt_vertices.shape != pred_vertices.shape:
    print(f"Error: Mesh shapes do not match. GT: {gt_vertices.shape}, Pred: {pred_vertices.shape}")
    exit(1)

print("Meshes loaded successfully. Processing...")

# Convert to mm if necessary (assuming the original is in meters)
gt_vertices *= 1000
pred_vertices *= 1000

# Calculate and print error statistics
mean_error, std_error, median_error = calculate_errors(pred_vertices, gt_vertices)
print(f'Error statistics:')
print(f'Mean Error: {mean_error:.3f} mm')
print(f'Std Error: {std_error:.3f} mm')
print(f'Median Error: {median_error:.3f} mm')

# Save error statistics to a file
if not os.path.exists(save_path):
    os.makedirs(save_path)

with open(os.path.join(save_path, 'error_stats.txt'), 'w') as f:
    f.write(f'Mean Error: {mean_error:.3f} mm\n')
    f.write(f'Std Error: {std_error:.3f} mm\n')
    f.write(f'Median Error: {median_error:.3f} mm\n')

print(f"Processing completed. Results saved to {save_path}")