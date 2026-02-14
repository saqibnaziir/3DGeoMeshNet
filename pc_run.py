import os
import os.path as osp
from glob import glob
import warnings
import pickle
import numpy as np
import igl
import trimesh
from tqdm import tqdm  # Added tqdm for progress tracking

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

def compute_principal_curvature_igl(mesh: trimesh.Trimesh):
    """
    Compute the principal curvature magnitudes and directions using LibIGL library.
    Principal curvature describes how a surface curves at each point in two perpendicular directions. 
    At each vertex of a 3D mesh, there are two principal curvatures:
        1. Minimum principal curvature (k1)
        2. Maximum principal curvature (k2)

    Parameters:
        mesh (trimesh.Trimesh): Input triangular mesh.
    
    Returns:
        curvatures (numpy.ndarray): Principal curvatures [k_min, k_max] for each vertex.
        directions (numpy.ndarray): Principal curvature directions (N, 2, 3).
    """
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    
    pd1, pd2, k1, k2 = igl.principal_curvature(vertices, faces)
    curvatures = np.stack([k1, k2], axis=1)  # k1 (min), k2 (max)
    directions = np.stack([pd1, pd2], axis=1)  # Corresponding directions
    
    # Compute derived curvatures
      #mean_curvature = (k1 + k2) / 2
    gaussian_curvature = k1 * k2
    
    # Scale directions by magnitudes
    # scaled_pd1 = pd1 * k1[:, np.newaxis]
    # scaled_pd2 = pd2 * k2[:, np.newaxis]
    # scaled_directions = np.stack([scaled_pd1, scaled_pd2], axis=1)  # Shape: (N, 2, 3)
   
    
    # Dot product curvature
    # product_curvature = pd1*k1
    # product_curvature2 = pd2*k2
    return curvatures, directions
    # return curvatures, gaussian_curvature

def save_principal_curvature(mesh_paths, save_path):
    """
    Compute and save principal curvatures for a list of meshes.
    
    Parameters:
        mesh_paths (List[str]): List of mesh file paths.
        save_path (str): Path to save the computed curvature data.
    """
    curvature_data = {}
    
    # Create a progress bar with total number of meshes
    progress_bar = tqdm(mesh_paths, desc="Processing Meshes", unit="mesh")
    
    for mesh_path in progress_bar:
        try:
            # Update progress bar description with current file
            progress_bar.set_description(f"Processing: {os.path.basename(mesh_path)}")
            
            # Load mesh using trimesh
            mesh = trimesh.load_mesh(mesh_path)
            
            # Compute principal curvatures & mean
            curvatures,directions  = compute_principal_curvature_igl(mesh)
            
            # Store data with filename as key
            curvature_data[os.path.basename(mesh_path)] = {
                "curvatures": curvatures,
                "directions": directions,
                # "scaled_directions": scaled_directions
                # "gaussian_curvature" : gaussian_curvature
               
            }
        
        except Exception as e:
            print(f"\nError processing {mesh_path}: {e}")
    
    # Save the curvature data
    with open(save_path, "wb") as f:
        pickle.dump(curvature_data, f)
    
    print(f"\nPrincipal curvatures saved at: {save_path}")
    print(f"Total meshes processed: {len(mesh_paths)}")

def main():
    # Specify the directory containing the meshes
    train_dir = r"D:\Saqib\COSURAI\FaceCom\data\train"
    
    # Specify the save path for curvature data
    curvature_save_path = r"D:\Saqib\COSURAI\FaceCom\data\curvature_data.pkl"
    
    # Find all .ply files recursively
    mesh_paths = glob(osp.join(train_dir, '**', '*.ply'), recursive=True)
    
    # If no .ply files found, try .obj files
    if not mesh_paths:
        mesh_paths = glob(osp.join(train_dir, '**', '*.obj'), recursive=True)
    
    # Check if mesh paths are found
    if not mesh_paths:
        print("No mesh files (.ply or .obj) found in the specified directory.")
        return
    
    # Compute and save principal curvatures
    save_principal_curvature(mesh_paths, curvature_save_path)

if __name__ == "__main__":
    main()













# import os
# import os.path as osp
# from glob import glob
# import warnings
# import pickle
# import numpy as np
# import igl
# import trimesh

# # Suppress warnings
# warnings.filterwarnings("ignore")
# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# def compute_principal_curvature_igl(mesh: trimesh.Trimesh):
#     """
#     I compute the principal curvature magnitudes and directions using LibIGL library.
#     Principal curvature describes how a surface curves at each point in two perpendicular directions. At each vertex of a 3D mesh, there are two principal curvatures:
#         1_Minimum principal curvature (k1)
#         2_Maximum principal curvature (k2)
#     Parameters:
#         mesh (trimesh.Trimesh): Input triangular mesh.
    
#     Returns:
#         curvatures (numpy.ndarray): Principal curvatures [k_min, k_max] for each vertex.
#         directions (numpy.ndarray): Principal curvature directions (N, 2, 3).
#     """
#     vertices = np.array(mesh.vertices)
#     faces = np.array(mesh.faces)
    
#     pd1, pd2, k1, k2 = igl.principal_curvature(vertices, faces)
#     curvatures = np.stack([k1, k2], axis=1)  # k1 (min), k2 (max)
#     directions = np.stack([pd1, pd2], axis=1)  # Corresponding directions
#     # dotproduct curvature
#     product_curvature = (pd1*k1+ pd2*k2)
    
    
#     return curvatures, directions, product_curvature

# def save_principal_curvature(mesh_paths, save_path):
#     """
#     Compute and save principal curvatures for a list of meshes.
    
#     Parameters:
#         mesh_paths (List[str]): List of mesh file paths.
#         save_path (str): Path to save the computed curvature data.
#     """
#     curvature_data = {}
#     for mesh_path in mesh_paths:
#         try:
#             # Load mesh using trimesh
#             mesh = trimesh.load_mesh(mesh_path)
            
#             # Compute principal curvatures & mean
#             curvatures, directions, product_curvature = compute_principal_curvature_igl(mesh)
            
#             # Store data with filename as key
#             curvature_data[os.path.basename(mesh_path)] = {
#                 "curvatures": curvatures,
#                 "directions": directions,
#                 "product_curvature": product_curvature
#             }
#             print(f"Processed: {mesh_path}")
#         except Exception as e:
#             print(f"Error processing {mesh_path}: {e}")
    
#     # Save the curvature data
#     with open(save_path, "wb") as f:
#         pickle.dump(curvature_data, f)
    
#     print(f"Principal curvatures saved at: {save_path}")

# def main():
#     # Specify the directory containing the meshes
#     train_dir = r"D:\Saqib\COSURAI\FaceCom\data\train"
    
#     # Specify the save path for curvature data
#     curvature_save_path = r"D:\Saqib\COSURAI\FaceCom\data\curvature_data.pkl"
    
#     # Find all .ply files recursively
#     mesh_paths = glob(osp.join(train_dir, '**', '*.ply'), recursive=True)
    
#     # If no .ply files found, try .obj files
#     if not mesh_paths:
#         mesh_paths = glob(osp.join(train_dir, '**', '*.obj'), recursive=True)
    
#     # Check if mesh paths are found
#     if not mesh_paths:
#         print("No mesh files (.ply or .obj) found in the specified directory.")
#         return
    
#     # Compute and save principal curvatures
#     save_principal_curvature(mesh_paths, curvature_save_path)

# if __name__ == "__main__":
#     main()