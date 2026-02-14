import os
import numpy as np
import trimesh
from tqdm import tqdm

# Configuration
source_folder = r"/home/nazir241/PosDoc/COSURAI/Codes/FaceCom/Denoising/input"  # Replace with your source folder path
target_folder = r"/home/nazir241/PosDoc/COSURAI/Codes/FaceCom/Denoising/output"  # Replace with your target folder path
gaus_mean = 0.0                         # Mean of the Gaussian noise
gaus_std = 0.001                         # Standard deviation of the Gaussian noise



# Ensure target folder exists
os.makedirs(target_folder, exist_ok=True)

def add_gaussian_noise(mesh, mean=0.0, std=0.001):
    """Adds Gaussian noise to the vertices of a mesh along the vertex normals."""
    # Calculate vertex normals
    normals = mesh.vertex_normals
    
    # Generate Gaussian noise scaled along the normals
    noise = np.random.normal(mean, std, size=(mesh.vertices.shape[0], 1)) * normals
    
    # Add noise to vertices
    noisy_vertices = mesh.vertices + noise
    
    # Return a new mesh with noisy vertices
    noisy_mesh = trimesh.Trimesh(vertices=noisy_vertices, faces=mesh.faces)
    return noisy_mesh

def process_meshes():
    """Loads .ply files from the source folder, adds Gaussian noise, and saves them to the target folder."""
    files = [f for f in os.listdir(source_folder) if f.endswith('.ply')]

    if not files:
        print("No .ply files found in the source folder.")
        return

    for file in tqdm(files, desc="Processing meshes"):
        try:
            # Load mesh
            file_path = os.path.join(source_folder, file)
            mesh = trimesh.load(file_path)

            # Ensure it's a valid mesh
            if not isinstance(mesh, trimesh.Trimesh):
                print(f"Skipping {file}: not a valid mesh.")
                continue

            # Add Gaussian noise
            noisy_mesh = add_gaussian_noise(mesh, gaus_mean, gaus_std)

            # Save noisy mesh
            target_path = os.path.join(target_folder, file)
            noisy_mesh.export(target_path)
        except Exception as e:
            print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    process_meshes()
    print("Processing complete. Check the target folder for noisy meshes.")



