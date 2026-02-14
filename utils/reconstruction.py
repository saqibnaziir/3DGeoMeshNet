#################### Selective Region Reconstruction #################
# import torch
# from config.config import read_config
# from trimesh import Trimesh, load_mesh
# from .funcs import load_generator, spherical_regularization_loss, save_ply_explicit, get_random_z, load_norm
# import numpy as np
# import os
# import warnings
# import sys
# import trimesh
# from scipy.optimize import linear_sum_assignment
# from scipy.spatial.distance import cdist
# from process import extract_meshes, compute_centroid, reorder_mesh_parts

# warnings.filterwarnings("ignore")
# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# sys.setrecursionlimit(30000)

# def load_normalization_params(config):
#     norm_dict = torch.load(os.path.join(config['dataset_dir'], "norm.pt"))
#     mean = norm_dict['mean']
#     std = norm_dict['std']
#     mean_subtraction_only = std is None
#     return mean, std, mean_subtraction_only

# def normalize_vertices(vertices, mean, std=None):
#     # Check if dimensions match
#     if vertices.shape[0] != mean.shape[0]:
#         # Handle dimension mismatch: Use per-coordinate mean/std instead of per-vertex
#         coord_mean = mean.mean(dim=0, keepdim=True)  # Shape becomes (1, 3)
#         normalized = vertices - coord_mean
        
#         if std is not None:
#             coord_std = std.mean(dim=0, keepdim=True)  # Shape becomes (1, 3)
#             normalized = normalized / coord_std
#     else:
#         # Original behavior when dimensions match
#         normalized = vertices - mean
#         if std is not None:
#             normalized = normalized / std
            
#     return normalized

# def denormalize_vertices(vertices, mean, std=None):
#     # Check if dimensions match
#     if vertices.shape[0] != mean.shape[0]:
#         # Handle dimension mismatch: Use per-coordinate mean/std
#         coord_mean = mean.mean(dim=0, keepdim=True)  # Shape becomes (1, 3)
        
#         if std is not None:
#             coord_std = std.mean(dim=0, keepdim=True)  # Shape becomes (1, 3)
#             vertices = vertices * coord_std
            
#         vertices = vertices + coord_mean
#     else:
#         # Original behavior
#         if std is not None:
#             vertices = vertices * std
#         vertices = vertices + mean
        
#     return vertices

# def identify_defective_regions(input_mesh, partial_mesh, distance_threshold=0.05):
#     """
#     Identify which vertices in the input mesh correspond to defective regions
#     (i.e., regions that are missing or significantly different in the partial mesh)
#     """
#     input_vertices = input_mesh.vertices
#     partial_vertices = partial_mesh.vertices
    
#     # For each input vertex, find the closest partial vertex
#     distances = cdist(input_vertices, partial_vertices)
#     min_distances = np.min(distances, axis=1)
    
#     # Vertices that are far from any partial vertex are considered defective
#     defective_mask = min_distances > distance_threshold
    
#     print(f"Identified {np.sum(defective_mask)} defective vertices out of {len(defective_mask)} total vertices")
#     print(f"Defective percentage: {100 * np.sum(defective_mask) / len(defective_mask):.2f}%")
    
#     return defective_mask

# def compute_partial_matching(partial_file, input_file, distance_threshold=0.05, fallback_threshold=0.1):
#     """
#     Compute matching between partial mesh and input mesh vertices
#     """
#     # Load the meshes
#     partial_mesh = trimesh.load(partial_file)
#     input_mesh = trimesh.load(input_file)
#     print(f"Partial mesh vertices: {partial_mesh.vertices.shape}")
#     print(f"Input mesh vertices: {input_mesh.vertices.shape}")

#     # Initialize matching indices to -1 (no match)
#     matching_indices = -np.ones(input_mesh.vertices.shape[0], dtype=int)
    
#     try:
#         # Try mesh part-based matching first
#         input_mesh_parts = extract_meshes(input_mesh)
#         partial_mesh_parts = extract_meshes(partial_mesh)
        
#         print(f"Extracted {len(input_mesh_parts)} parts from input mesh")
#         print(f"Extracted {len(partial_mesh_parts)} parts from partial mesh")
        
#         # Proceed only if parts were successfully extracted
#         if len(input_mesh_parts) > 0 and len(partial_mesh_parts) > 0:
#             try:
#                 # Try to reorder the partial mesh parts to match input mesh parts
#                 partial_mesh_parts = reorder_mesh_parts(
#                     input_mesh_parts, partial_mesh_parts, 
#                     distance_threshold=distance_threshold
#                 )
#                 print("Successfully reordered mesh parts")
#             except Exception as e:
#                 print(f"Warning: Mesh part reordering failed: {str(e)}. Using original order.")
                
#             # Process each mesh part
#             offset_input = 0
#             offset_partial = 0
            
#             for i in range(len(input_mesh_parts)):
#                 input_part = input_mesh_parts[i]
#                 V_input = input_part.vertices

#                 # Skip if no corresponding partial part
#                 if i >= len(partial_mesh_parts) or partial_mesh_parts[i] is None:
#                     offset_input += V_input.shape[0]
#                     continue

#                 V_partial = partial_mesh_parts[i].vertices
                
#                 # For very small parts, use a simpler matching approach
#                 if V_partial.shape[0] < 10 or V_input.shape[0] < 10:
#                     print(f"Part {i} is very small, using simpler matching")
#                     # Simple nearest neighbor matching
#                     distances = cdist(V_partial, V_input)
#                     for j in range(V_input.shape[0]):
#                         nearest = np.argmin(distances[:, j])
#                         if distances[nearest, j] < fallback_threshold:
#                             matching_indices[offset_input + j] = offset_partial + nearest
#                 else:
#                     # Use Hungarian algorithm for larger parts
#                     cost_matrix = cdist(V_partial, V_input)
#                     row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    
#                     # Filter matches by distance threshold
#                     for r, c in zip(row_ind, col_ind):
#                         if cost_matrix[r, c] < fallback_threshold:
#                             matching_indices[offset_input + c] = offset_partial + r

#                 offset_input += V_input.shape[0]
#                 offset_partial += V_partial.shape[0]
#         else:
#             raise ValueError("No mesh parts extracted")
            
#     except Exception as e:
#         print(f"Part-based matching failed: {str(e)}. Falling back to global matching.")
#         # Reset matching indices
#         matching_indices = -np.ones(input_mesh.vertices.shape[0], dtype=int)
        
#     # Check if we have reasonable matches
#     match_percentage = 100 * (1 - (matching_indices == -1).sum() / len(matching_indices))
#     print(f"Match percentage: {match_percentage:.2f}%")
    
#     # Fallback to global nearest-neighbor if part-based matching fails or yields poor results
#     if match_percentage < 15:
#         print("Low match percentage. Performing global nearest-neighbor matching...")
        
#         # Compute distances between all vertices
#         print("Computing global distance matrix...")
#         distances = cdist(partial_mesh.vertices, input_mesh.vertices)
        
#         # For each input vertex, find the closest partial vertex
#         for i in range(input_mesh.vertices.shape[0]):
#             closest_partial_idx = np.argmin(distances[:, i])
#             # Only match if distance is reasonable
#             if distances[closest_partial_idx, i] < fallback_threshold:
#                 matching_indices[i] = closest_partial_idx
                
#         # Update match percentage
#         match_percentage = 100 * (1 - (matching_indices == -1).sum() / len(matching_indices))
#         print(f"After fallback: {match_percentage:.2f}% vertices matched")
    
#     # Final check
#     if (matching_indices != -1).sum() < 10:
#         print("WARNING: Very few matches found. This will likely result in poor reconstruction.")
    
#     return matching_indices

# def facial_mesh_reconstruction(in_file, partial_file, out_file, config, generator, 
#                               num_iterations=1000, learning_rate=0.01, lambda_reg=0.001, verbose=True):
#     """
#     Selective facial mesh reconstruction that preserves non-defective regions
#     """
#     device = next(generator.parameters()).device
    
#     # Load input mesh
#     mesh_in = load_mesh(in_file)
#     input_vertices = torch.tensor(mesh_in.vertices, dtype=torch.float, device=device)
    
#     # Load normalization parameters
#     mean, std, mean_subtraction_only = load_normalization_params(config)
#     mean = mean.to(device)
#     if std is not None:
#         std = std.to(device)
    
#     if partial_file is None:
#         # No partial mesh provided, just do standard reconstruction
#         normalized_vertices = normalize_vertices(input_vertices, mean, std if not mean_subtraction_only else None)
        
#         with torch.no_grad():
#             z = generator.encoder(normalized_vertices, 1)
#             reconstructed_vertices = generator.decoder(z, 1)
#             reconstructed_vertices = denormalize_vertices(
#                 reconstructed_vertices, mean, std if not mean_subtraction_only else None
#             )
        
#         reconstructed_vertices = reconstructed_vertices.cpu().numpy()
#         reconstructed_mesh = Trimesh(reconstructed_vertices, mesh_in.faces)
#         save_ply_explicit(reconstructed_mesh, out_file)
        
#         if verbose:
#             print(f"Standard reconstruction saved to {out_file}")
#         return reconstructed_mesh
    
#     # Load partial mesh
#     partial_mesh = load_mesh(partial_file)
    
#     # Identify defective regions in the input mesh
#     defective_mask = identify_defective_regions(mesh_in, partial_mesh, distance_threshold=0.05)
#     defective_indices = np.where(defective_mask)[0]
#     non_defective_indices = np.where(~defective_mask)[0]
    
#     if len(defective_indices) == 0:
#         print("No defective regions found. Returning original mesh.")
#         reconstructed_mesh = Trimesh(mesh_in.vertices, mesh_in.faces)
#         save_ply_explicit(reconstructed_mesh, out_file)
#         return reconstructed_mesh
    
#     # Get initial reconstruction for the defective regions
#     normalized_vertices = normalize_vertices(input_vertices, mean, std if not mean_subtraction_only else None)
    
#     with torch.no_grad():
#         z_initial = generator.encoder(normalized_vertices, 1)
#         initial_reconstruction = generator.decoder(z_initial, 1)
    
#     # Create the target vertices: partial mesh vertices for guidance
#     partial_vertices = torch.tensor(partial_mesh.vertices, dtype=torch.float, device=device)
#     normalized_partial_vertices = normalize_vertices(
#         partial_vertices, mean, std if not mean_subtraction_only else None
#     )
    
#     # Compute matching for guidance
#     temp_file = out_file + ".temp.ply"
#     temp_vertices = denormalize_vertices(
#         initial_reconstruction, mean, std if not mean_subtraction_only else None
#     )
#     temp_mesh = Trimesh(temp_vertices.cpu().numpy(), mesh_in.faces)
#     save_ply_explicit(temp_mesh, temp_file)
    
#     matching_indices = compute_partial_matching(partial_file, temp_file)
    
#     # Clean up temporary file
#     if os.path.exists(temp_file):
#         os.remove(temp_file)
    
#     # Convert to tensors
#     matching_indices_tensor = torch.tensor(matching_indices, device=device)
#     defective_mask_tensor = torch.tensor(defective_mask, device=device)
    
#     # Only optimize latent code for reconstruction of defective regions
#     z = z_initial.clone().detach().requires_grad_(True)
#     optimizer = torch.optim.Adam([z], lr=learning_rate)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.8, patience=50, verbose=verbose
#     )
    
#     generator.eval()
    
#     best_loss = float('inf')
#     best_z = z.clone().detach()
    
#     if verbose:
#         print(f"Starting selective reconstruction optimization for {num_iterations} iterations")
#         print(f"Optimizing {len(defective_indices)} defective vertices out of {len(input_vertices)} total")
    
#     for iteration in range(num_iterations):
#         optimizer.zero_grad()
        
#         # Forward pass through decoder
#         reconstructed_vertices_norm = generator.decoder(z, 1)
        
#         # Compute loss only for defective regions that have matches in partial mesh
#         defective_with_matches_mask = defective_mask_tensor & (matching_indices_tensor != -1)
        
#         if defective_with_matches_mask.sum() == 0:
#             # No defective vertices with matches, use regularization only
#             reg_loss = lambda_reg * torch.mean(z ** 2)
#             total_loss = reg_loss
#         else:
#             # Get defective vertices that have matches
#             defective_matched_indices = torch.where(defective_with_matches_mask)[0]
#             corresponding_partial_indices = matching_indices_tensor[defective_with_matches_mask]
            
#             # Reconstruction loss for defective regions
#             reconstructed_defective = reconstructed_vertices_norm[defective_matched_indices]
#             target_partial = normalized_partial_vertices[corresponding_partial_indices].detach()
            
#             reconstruction_loss = torch.mean(torch.sum((reconstructed_defective - target_partial) ** 2, dim=1))
            
#             # Regularization to keep z reasonable
#             reg_loss = lambda_reg * torch.mean(z ** 2)
            
#             total_loss = reconstruction_loss + reg_loss
        
#         # Check for valid loss
#         if not torch.isfinite(total_loss):
#             print(f"Warning: Non-finite loss at iteration {iteration}")
#             continue
        
#         # Backward pass
#         total_loss.backward()
        
#         # Gradient clipping
#         torch.nn.utils.clip_grad_norm_([z], 1.0)
        
#         # Update
#         optimizer.step()
#         scheduler.step(total_loss)
        
#         # Track best
#         if total_loss.item() < best_loss:
#             best_loss = total_loss.item()
#             best_z = z.clone().detach()
        
#         # Progress
#         if verbose and (iteration + 1) % 100 == 0:
#             recon_loss = reconstruction_loss.item() if 'reconstruction_loss' in locals() else 0.0
#             print(f"Iteration {iteration + 1}, Loss: {total_loss.item():.6f} "
#                   f"(Recon: {recon_loss:.6f}, Reg: {reg_loss.item():.6f}), "
#                   f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
#     # Create final mesh by combining original non-defective regions with reconstructed defective regions
#     with torch.no_grad():
#         final_reconstructed_norm = generator.decoder(best_z, 1)
#         final_reconstructed = denormalize_vertices(
#             final_reconstructed_norm, mean, std if not mean_subtraction_only else None
#         )
    
#     # Create final vertices: preserve original non-defective, use reconstructed defective
#     final_vertices = input_vertices.clone()
#     final_vertices[defective_mask] = final_reconstructed[defective_mask]
    
#     # Create and save final mesh
#     final_vertices_np = final_vertices.cpu().numpy()
#     reconstructed_mesh = Trimesh(final_vertices_np, mesh_in.faces)
#     save_ply_explicit(reconstructed_mesh, out_file)
    
#     if verbose:
#         print(f"Selective reconstruction saved to {out_file}")
#         print(f"Final loss: {best_loss:.6f}")
#         print(f"Preserved {len(non_defective_indices)} non-defective vertices")
#         print(f"Reconstructed {len(defective_indices)} defective vertices")
    
#     return reconstructed_mesh

############################ Reconstruction normal with PostProcessing
# import torch
# import torch.nn.functional as F
# from config.config import read_config
# from trimesh import Trimesh, load_mesh
# from .funcs import load_generator, spherical_regularization_loss, save_ply_explicit, get_random_z, load_norm
# import numpy as np
# import os
# import warnings
# import sys
# warnings.filterwarnings("ignore")
# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# sys.setrecursionlimit(30000)

# def load_normalization_params(config):
#     norm_dict = torch.load(os.path.join(config['dataset_dir'], "norm.pt"))
#     mean = norm_dict['mean']
#     std = norm_dict['std']
#     mean_subtraction_only = std is None
#     return mean, std, mean_subtraction_only

# def normalize_vertices(vertices, mean, std=None):
#     normalized = vertices - mean
#     if std is not None:
#         normalized = normalized / std
#     return normalized

# def denormalize_vertices(vertices, mean, std=None):
#     if std is not None:
#         vertices = vertices * std
#     vertices = vertices + mean
#     return vertices

# def smoothness_loss(vertices, faces):
#     """
#     Calculate smoothness loss based on Laplacian operator
#     """
#     # Get unique edges
#     v1 = faces[:, 0]
#     v2 = faces[:, 1]
#     v3 = faces[:, 2]
    
#     edges = torch.cat([
#         torch.stack([v1, v2], dim=1),
#         torch.stack([v2, v3], dim=1),
#         torch.stack([v3, v1], dim=1)
#     ], dim=0)
    
#     # Remove duplicates
#     edges = edges.sort(dim=1)[0]
#     edges = torch.unique(edges, dim=0)
    
#     # Calculate edge lengths
#     v_i = vertices[edges[:, 0]]
#     v_j = vertices[edges[:, 1]]
    
#     # Edge vectors
#     edge_vectors = v_i - v_j
    
#     # Smoothness loss is sum of squared edge lengths
#     return torch.mean(torch.sum(edge_vectors ** 2, dim=1))

# def special_point_refinement(mesh):
#     """
#     Refine mesh by adjusting outlier vertices
#     """
#     vertices = mesh.vertices.copy()
#     for i in range(mesh.vertices.shape[0]):
#         neighbor = mesh.vertex_neighbors[i]
#         if len(neighbor) == 0:
#             continue
#         mean_x = np.mean(vertices[neighbor], axis=0)
#         x = vertices[i]
#         mean_distance = np.mean(np.linalg.norm(mesh.vertices[neighbor] - mean_x, axis=1))
#         if np.linalg.norm(mean_x - x) > 0.5 * mean_distance:
#             vertices[i] = mean_x
#     return Trimesh(vertices, mesh.faces)

# def taubin_smoothing(mesh, iterations=3, lambda_factor=0.5, mu_factor=-0.53):
#     """
#     Apply Taubin smoothing to a mesh to reduce noise while preserving features.
#     This is a shrinkage-free alternative to Laplacian smoothing.
#     """
#     vertices = mesh.vertices.copy()
    
#     # Get vertex adjacency
#     adjacency = mesh.vertex_neighbors
    
#     for _ in range(iterations):
#         # First pass with positive lambda
#         new_vertices = vertices.copy()
#         for i in range(len(vertices)):
#             if len(adjacency[i]) > 0:
#                 neighbor_positions = vertices[adjacency[i]]
#                 centroid = np.mean(neighbor_positions, axis=0)
#                 new_vertices[i] = vertices[i] + lambda_factor * (centroid - vertices[i])
        
#         vertices = new_vertices.copy()
        
#         # Second pass with negative mu
#         new_vertices = vertices.copy()
#         for i in range(len(vertices)):
#             if len(adjacency[i]) > 0:
#                 neighbor_positions = vertices[adjacency[i]]
#                 centroid = np.mean(neighbor_positions, axis=0)
#                 new_vertices[i] = vertices[i] + mu_factor * (centroid - vertices[i])
        
#         vertices = new_vertices
    
#     # Create new mesh with smoothed vertices and original faces
#     smoothed_mesh = Trimesh(vertices=vertices, faces=mesh.faces)
#     return smoothed_mesh

# def facial_mesh_reconstruction(in_file, out_file, config, generator, lambda_reg=None, verbose=True):
#     """
#     Reconstruct a facial mesh with Taubin smoothing
    
#     Args:
#         in_file: Path to input mesh file
#         out_file: Path where output mesh will be saved
#         config: Configuration dictionary
#         generator: Neural network model (encoder-decoder)
#         lambda_reg: Regularization parameter
#         verbose: Whether to print progress messages
#     """
#     device = next(generator.parameters()).device
    
#     # Load input mesh
#     if verbose:
#         print(f"Loading input mesh from {in_file}")
#     mesh_in = load_mesh(in_file)
#     vertices = torch.tensor(mesh_in.vertices, dtype=torch.float).to(device)
    
#     # Load normalization parameters
#     mean, std, mean_subtraction_only = load_normalization_params(config)
#     mean = mean.to(device)
#     if std is not None:
#         std = std.to(device)
    
#     # Normalize input vertices
#     normalized_vertices = normalize_vertices(vertices, mean, std if not mean_subtraction_only else None)
    
#     # Process through the network
#     if verbose:
#         print("Running through neural network...")
#     z = generator.encoder(normalized_vertices, 1)
#     reconstructed_vertices = generator.decoder(z, 1)
    
#     # Denormalize the output vertices
#     reconstructed_vertices = denormalize_vertices(
#         reconstructed_vertices, 
#         mean,
#         std if not mean_subtraction_only else None
#     )
    
#     # Create reconstructed mesh
#     reconstructed_vertices = reconstructed_vertices.detach().cpu().numpy()
#     reconstructed_mesh = Trimesh(reconstructed_vertices, mesh_in.faces)
    
#     # Apply Taubin smoothing for post-processing
#     if verbose:
#         print("Applying Taubin smoothing...")
#     processed_mesh = taubin_smoothing(reconstructed_mesh)
    
#     # Further refinement to fix outliers (optional)
#     processed_mesh = special_point_refinement(processed_mesh)
    
#     # Save the final mesh
#     save_ply_explicit(processed_mesh, out_file)
    
#     if verbose:
#         print(f"Processed mesh saved to {out_file}")
    
#     return processed_mesh

############################ Reconstruction normal without PostProcessing
import torch
from config.config import read_config
from trimesh import Trimesh, load_mesh
from .funcs import load_generator, spherical_regularization_loss, save_ply_explicit, get_random_z, load_norm
import numpy as np
import os
import warnings
import sys
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
sys.setrecursionlimit(30000)
def load_normalization_params(config):
    norm_dict = torch.load(os.path.join(config['dataset_dir'], "norm.pt"))
    mean = norm_dict['mean']
    std = norm_dict['std']
    mean_subtraction_only = std is None
    return mean, std, mean_subtraction_only
# ################################# for same no. of vertices count
def normalize_vertices(vertices, mean, std=None):
    normalized = vertices - mean
    if std is not None:
        normalized = normalized / std
    return normalized
def denormalize_vertices(vertices, mean, std=None):
    if std is not None:
        vertices = vertices * std
    vertices = vertices + mean
    return vertices
def facial_mesh_reconstruction(in_file, out_file, config, generator, lambda_reg=None, verbose=True):
    device = next(generator.parameters()).device
    # Load input mesh
    mesh_in = load_mesh(in_file)
    vertices = torch.tensor(mesh_in.vertices, dtype=torch.float).to(device)
    # Load normalization parameters
    mean, std, mean_subtraction_only = load_normalization_params(config)
    mean = mean.to(device)
    if std is not None:
        std = std.to(device)
    # Normalize input vertices
    normalized_vertices = normalize_vertices(vertices, mean, std if not mean_subtraction_only else None)
    # Process through the network
    z = generator.encoder(normalized_vertices, 1)
    reconstructed_vertices = generator.decoder(z, 1)
    # Denormalize the output vertices
    reconstructed_vertices = denormalize_vertices(
        reconstructed_vertices, 
        mean,
        std if not mean_subtraction_only else None
    )
    # Create and save reconstructed mesh
    reconstructed_vertices = reconstructed_vertices.detach().cpu().numpy()
    reconstructed_mesh = Trimesh(reconstructed_vertices, mesh_in.faces)
    save_ply_explicit(reconstructed_mesh, out_file)
    if verbose:
        print(f"Reconstructed mesh saved to {out_file}")
    return reconstructed_mesh


def generate_face_sample(out_file, config, generator):
    """Generate a random face mesh by sampling z and decoding. Used by scripts/generate_samples.py."""
    generator.eval()
    device = next(generator.parameters()).device
    z = get_random_z(config['z_length'], requires_grad=False)
    mean, std = load_norm(config)
    mean = mean.to(device)
    std = std.to(device) if std is not None else None
    out = generator.decoder(z.to(device), 1).detach().cpu()
    out = out * std + mean if std is not None else out + mean
    template_mesh = load_mesh(config["template_file"])
    mesh = Trimesh(out.numpy(), template_mesh.faces)
    save_ply_explicit(mesh, out_file)


############################ Denoising 
# import torch
# from config.config import read_config
# from trimesh import Trimesh, load_mesh
# from .funcs import load_generator, spherical_regularization_loss, save_ply_explicit, get_random_z, load_norm
# import numpy as np
# import os
# import warnings
# import sys
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from scipy.sparse import lil_matrix
# from scipy.sparse.linalg import spsolve

# warnings.filterwarnings("ignore")
# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# sys.setrecursionlimit(30000)

# def load_normalization_params(config):
#     norm_dict = torch.load(os.path.join(config['dataset_dir'], "norm.pt"))
#     mean = norm_dict['mean']
#     std = norm_dict['std']
#     mean_subtraction_only = std is None
#     return mean, std, mean_subtraction_only

# def normalize_vertices(vertices, mean, std=None):
#     normalized = vertices - mean
#     if std is not None:
#         normalized = normalized / std
#     return normalized

# def denormalize_vertices(vertices, mean, std=None):
#     if std is not None:
#         vertices = vertices * std
#     vertices = vertices + mean
#     return vertices

# def check_topology(mesh1, mesh2):
#     """Check if two meshes have the same topology"""
#     same_vertices = mesh1.vertices.shape == mesh2.vertices.shape
#     same_faces = np.array_equal(mesh1.faces, mesh2.faces)
#     return same_vertices, same_faces

# def analyze_mesh_properties(mesh, name):
#     """Analyze key properties of a mesh"""
#     properties = {
#         "Name": name,
#         "Vertices": mesh.vertices.shape[0],
#         "Faces": mesh.faces.shape[0],
#         "Vertex Range": (np.min(mesh.vertices, axis=0), np.max(mesh.vertices, axis=0)),
#         "Vertex Mean": np.mean(mesh.vertices, axis=0),
#         "Vertex Std": np.std(mesh.vertices, axis=0),
#         "Mesh Volume": mesh.volume if hasattr(mesh, 'volume') else None,
#         "Mesh Area": mesh.area if hasattr(mesh, 'area') else None,
#     }
#     return properties

# def visualize_vertex_distributions(original_vertices, reconstructed_vertices, save_path=None):
#     """Visualize the distribution of vertex coordinates"""
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
#     # Plot x, y, z distributions
#     for i, dim in enumerate(['X', 'Y', 'Z']):
#         axes[i].hist(original_vertices[:, i], bins=50, alpha=0.5, label='Original')
#         axes[i].hist(reconstructed_vertices[:, i], bins=50, alpha=0.5, label='Reconstructed')
#         axes[i].set_title(f'{dim} Coordinate Distribution')
#         axes[i].legend()
    
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path)
#         plt.close()
#     else:
#         plt.show()

# def detect_anomalies(vertices, threshold=3.0):
#     """Detect anomalous vertices based on z-score"""
#     mean = np.mean(vertices, axis=0)
#     std = np.std(vertices, axis=0)
#     z_scores = np.abs((vertices - mean) / std)
#     anomalies = np.where(np.any(z_scores > threshold, axis=1))[0]
#     return anomalies

# def build_adjacency_matrix(mesh):
#     """Build vertex adjacency matrix from mesh faces"""
#     num_vertices = mesh.vertices.shape[0]
#     adjacency = lil_matrix((num_vertices, num_vertices), dtype=np.float32)
    
#     # Populate adjacency matrix from faces
#     for face in mesh.faces:
#         for i in range(3):  # Triangle has 3 vertices
#             v1 = face[i]
#             v2 = face[(i + 1) % 3]  # Next vertex in face (circular)
#             adjacency[v1, v2] = 1
#             adjacency[v2, v1] = 1  # Undirected graph
    
#     return adjacency.tocsr()  # Convert to CSR for efficient operations

# def compute_vertex_weights(mesh):
#     """Compute cotangent weights for Laplacian smoothing"""
#     num_vertices = mesh.vertices.shape[0]
#     weights = lil_matrix((num_vertices, num_vertices), dtype=np.float32)
    
#     # For simplicity, use uniform weights instead of cotangent weights
#     adj_matrix = build_adjacency_matrix(mesh)
#     for i in range(num_vertices):
#         neighbors = adj_matrix[i].nonzero()[1]
#         degree = len(neighbors)
#         if degree > 0:
#             for j in neighbors:
#                 weights[i, j] = 1.0 / degree
    
#     return weights.tocsr()

# def laplacian_mesh_denoising(mesh, iterations=5, lambda_factor=0.5, preserve_volume=True, debug=False):
#     if debug:
#         print(f"Starting mesh denoising with {iterations} iterations, lambda={lambda_factor}")
    
#     # Work with a copy of the mesh
#     denoised_mesh = Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy())
#     vertices = denoised_mesh.vertices
    
#     # Build adjacency matrix
#     adjacency = build_adjacency_matrix(denoised_mesh)
    
#     # Compute initial centroid if preserving volume
#     if preserve_volume:
#         initial_centroid = np.mean(vertices, axis=0)
    
#     # Perform iterations of Laplacian smoothing
#     for i in range(iterations):
#         new_vertices = vertices.copy()
        
#         # For each vertex, compute new position based on neighbors
#         for v_idx in range(vertices.shape[0]):
#             # Get neighbors
#             neighbors = adjacency[v_idx].nonzero()[1]
            
#             if len(neighbors) > 0:
#                 # Compute centroid of neighbors
#                 neighbor_centroid = np.mean(vertices[neighbors], axis=0)
                
#                 # Move vertex towards centroid by lambda_factor
#                 new_vertices[v_idx] = vertices[v_idx] + lambda_factor * (neighbor_centroid - vertices[v_idx])
        
#         # Update vertices
#         vertices = new_vertices
        
#         # Preserve volume by restoring original centroid
#         if preserve_volume:
#             current_centroid = np.mean(vertices, axis=0)
#             vertices += (initial_centroid - current_centroid)
        
#         if debug and (i == 0 or i == iterations - 1 or i % 5 == 0):
#             print(f"  Iteration {i+1}: Vertex range [{np.min(vertices):.4f}, {np.max(vertices):.4f}]")
    
#     # Update mesh with denoised vertices
#     denoised_mesh.vertices = vertices
    
#     # Check for invalid faces
#     if not denoised_mesh.is_watertight:
#         if debug:
#             print("Warning: Denoised mesh is not watertight")
    
#     return denoised_mesh

# def advanced_mesh_denoising(mesh, method='laplacian', strength=0.5, iterations=5, debug=False):

#     if method == 'laplacian':
#         return laplacian_mesh_denoising(mesh, iterations, strength, True, debug)
#     elif method == 'bilateral':
#         # Implement bilateral mesh denoising (more advanced)
#         if debug:
#             print("Using bilateral mesh filter (placeholder)")
#         # For now, fall back to Laplacian with modified parameters
#         return laplacian_mesh_denoising(mesh, iterations, strength * 0.7, True, debug)
#     elif method == 'taubin':
#         # Implement Taubin smoothing (lambda/mu scheme)
#         if debug:
#             print("Using Taubin lambda/mu smoothing (placeholder)")
#         # Alternating positive and negative smoothing factors
#         mesh1 = laplacian_mesh_denoising(mesh, iterations // 2, strength, False, debug)
#         return laplacian_mesh_denoising(mesh1, iterations // 2, -strength * 0.5, True, debug)
#     else:
#         if debug:
#             print(f"Unknown denoising method '{method}', falling back to laplacian")
#         return laplacian_mesh_denoising(mesh, iterations, strength, True, debug)

# def facial_mesh_reconstruction(in_file, out_file, config, generator, 
#                               lambda_reg=None, verbose=True, debug=True,
#                               denoise=True, denoise_method='laplacian', 
#                               denoise_strength=0.3, denoise_iterations=3):
#     device = next(generator.parameters()).device
    
#     # Load input mesh
#     mesh_in = load_mesh(in_file)
    
#     if debug:
#         print(f"Original mesh - Vertices: {mesh_in.vertices.shape}, Faces: {mesh_in.faces.shape}")
    
#     # Apply denoising if requested
#     if denoise:
#         if verbose:
#             print(f"Applying {denoise_method} denoising (strength={denoise_strength}, iterations={denoise_iterations})...")
        
#         # Create directory for intermediate results if debugging
#         if debug:
#             intermediate_dir = os.path.join(os.path.dirname(out_file), "intermediate")
#             os.makedirs(intermediate_dir, exist_ok=True)
            
#             # Save original mesh for comparison
#             original_copy_path = os.path.join(intermediate_dir, "original.ply")
#             mesh_in.export(original_copy_path)
#             if verbose:
#                 print(f"Original mesh saved to {original_copy_path}")
        
#         # Detect anomalies in the original mesh
#         anomalies = detect_anomalies(mesh_in.vertices, threshold=5.0)
#         if len(anomalies) > 0:
#             print(f"Detected {len(anomalies)} anomalous vertices in original mesh")
#             if debug and len(anomalies) < 20:
#                 print(f"Anomalous vertex indices: {anomalies}")
        
#         # Apply denoising
#         denoised_mesh = advanced_mesh_denoising(
#             mesh_in, 
#             method=denoise_method,
#             strength=denoise_strength,
#             iterations=denoise_iterations,
#             debug=debug
#         )
        
#         # Save denoised mesh for inspection if debugging
#         if debug:
#             denoised_path = os.path.join(intermediate_dir, "denoised.ply")
#             denoised_mesh.export(denoised_path)
#             if verbose:
#                 print(f"Denoised mesh saved to {denoised_path}")
        
#         # Use denoised mesh for further processing
#         mesh_in = denoised_mesh
    
#     vertices = torch.tensor(mesh_in.vertices, dtype=torch.float).to(device)
    
#     if debug:
#         print(f"Input mesh after preprocessing - Vertices: {vertices.shape}, Faces: {mesh_in.faces.shape}")
#         print(f"Device: {device}, Vertex dtype: {vertices.dtype}")
    
#     # Load normalization parameters
#     mean, std, mean_subtraction_only = load_normalization_params(config)
#     mean = mean.to(device)
#     if std is not None:
#         std = std.to(device)
    
#     if debug:
#         print(f"Mean shape: {mean.shape}, Mean range: [{mean.min().item()}, {mean.max().item()}]")
#         if std is not None:
#             print(f"Std shape: {std.shape}, Std range: [{std.min().item()}, {std.max().item()}]")
#         else:
#             print("Using mean subtraction only (no std normalization)")
    
#     # Check for NaN or Inf values
#     if torch.isnan(vertices).any() or torch.isinf(vertices).any():
#         print("WARNING: Input vertices contain NaN or Inf values!")
    
#     # Normalize input vertices
#     normalized_vertices = normalize_vertices(vertices, mean, std if not mean_subtraction_only else None)
    
#     if debug:
#         print(f"Normalized vertices range: [{normalized_vertices.min().item()}, {normalized_vertices.max().item()}]")
#         if torch.isnan(normalized_vertices).any() or torch.isinf(normalized_vertices).any():
#             print("WARNING: Normalized vertices contain NaN or Inf values!")
    
#     # Prepare batch for encoder
#     batch_size = 1
    
#     # Check model's expected input format
#     try:
#         # Process through the network - inspect shapes at each step
#         if debug:
#             print(f"Input to encoder - Shape: {normalized_vertices.shape}")
        
#         # Handle reshaping based on expected input format
#         if len(normalized_vertices.shape) == 2:  # [num_vertices, feature_dim]
#             # Flatten if needed by model
#             flattened = normalized_vertices.view(batch_size, -1)
#             if debug:
#                 print(f"Reshaped for batch - Shape: {flattened.shape}")
        
#         # Process through encoder
#         z = generator.encoder(normalized_vertices, batch_size)
        
#         if debug:
#             print(f"Latent code z - Shape: {z.shape}")
#             print(f"z range: [{z.min().item()}, {z.max().item()}]")
#             if torch.isnan(z).any() or torch.isinf(z).any():
#                 print("WARNING: Latent code contains NaN or Inf values!")
        
#         # Process through decoder
#         reconstructed_vertices = generator.decoder(z, batch_size)
        
#         if debug:
#             print(f"Reconstructed vertices - Shape: {reconstructed_vertices.shape}")
#             print(f"Reconstructed range: [{reconstructed_vertices.min().item()}, {reconstructed_vertices.max().item()}]")
#             if torch.isnan(reconstructed_vertices).any() or torch.isinf(reconstructed_vertices).any():
#                 print("WARNING: Reconstructed vertices contain NaN or Inf values!")
        
#         # Denormalize the output vertices
#         reconstructed_vertices = denormalize_vertices(
#             reconstructed_vertices, mean, std if not mean_subtraction_only else None
#         )
        
#         if debug:
#             print(f"Denormalized vertices - Shape: {reconstructed_vertices.shape}")
#             print(f"Denormalized range: [{reconstructed_vertices.min().item()}, {reconstructed_vertices.max().item()}]")
#             if torch.isnan(reconstructed_vertices).any() or torch.isinf(reconstructed_vertices).any():
#                 print("WARNING: Denormalized vertices contain NaN or Inf values!")
        
#         # Create reconstruction error metrics
#         mse = torch.mean((vertices - reconstructed_vertices) ** 2).item()
#         mae = torch.mean(torch.abs(vertices - reconstructed_vertices)).item()
        
#         if debug:
#             print(f"Reconstruction error - MSE: {mse:.6f}, MAE: {mae:.6f}")
        
#         # Create and save reconstructed mesh
#         reconstructed_vertices_np = reconstructed_vertices.detach().cpu().numpy()
        
#         # Detect anomalies in reconstructed vertices
#         anomalies = detect_anomalies(reconstructed_vertices_np, threshold=5.0)
#         if len(anomalies) > 0:
#             print(f"WARNING: Detected {len(anomalies)} anomalous vertices in reconstruction!")
#             if debug and len(anomalies) < 20:
#                 print(f"Anomalous vertex indices: {anomalies}")
#                 print(f"Anomalous vertex values: {reconstructed_vertices_np[anomalies]}")
        
#         # Optional: Apply post-processing denoising to the reconstructed mesh
#         # This can help smooth out any remaining artifacts
#         final_post_denoise = False  # Set to True to enable post-reconstruction denoising
#         if final_post_denoise:
#             if verbose:
#                 print("Applying post-reconstruction denoising...")
            
#             # Create temporary mesh
#             temp_mesh = Trimesh(reconstructed_vertices_np, mesh_in.faces)
            
#             # Apply gentle denoising
#             smoothed_mesh = advanced_mesh_denoising(
#                 temp_mesh,
#                 method='laplacian',
#                 strength=0.1,  # Use lower strength for final pass
#                 iterations=2,
#                 debug=debug
#             )
            
#             reconstructed_vertices_np = smoothed_mesh.vertices
        
#         # Create reconstructed mesh
#         reconstructed_mesh = Trimesh(reconstructed_vertices_np, mesh_in.faces)
        
#         # Check topology
#         same_vert_count, same_faces = check_topology(mesh_in, reconstructed_mesh)
#         if debug:
#             print(f"Topology check - Same vertex count: {same_vert_count}, Same faces: {same_faces}")
        
#         if not same_vert_count:
#             print(f"WARNING: Vertex count mismatch! Original: {mesh_in.vertices.shape[0]}, Reconstructed: {reconstructed_mesh.vertices.shape[0]}")
        
#         if not same_faces:
#             print(f"WARNING: Face definitions don't match between original and reconstructed meshes!")
        
#         # Additional analysis
#         orig_props = analyze_mesh_properties(mesh_in, "Original")
#         recon_props = analyze_mesh_properties(reconstructed_mesh, "Reconstructed")
        
#         if debug:
#             print("\nMesh Properties Comparison:")
#             for key in orig_props:
#                 print(f"{key}: {orig_props[key]} (Original) vs {recon_props[key]} (Reconstructed)")
        
#         # Save the reconstructed mesh
#         save_ply_explicit(reconstructed_mesh, out_file)
        
#         # Visualize vertex distributions
#         if debug:
#             vis_path = os.path.join(os.path.dirname(out_file), "vertex_distribution.png")
#             visualize_vertex_distributions(mesh_in.vertices, reconstructed_vertices_np, vis_path)
#             print(f"Vertex distribution visualization saved to {vis_path}")
        
#         if verbose:
#             print(f"Reconstructed mesh saved to {out_file}")
        
#         return reconstructed_mesh, mse, mae
    
#     except Exception as e:
#         print(f"ERROR during reconstruction: {e}")
#         import traceback
#         traceback.print_exc()
#         return None, None, None

# def batch_reconstruction(input_dir, output_dir, config_path, model_path, num_samples=None,
#                         denoise=True, denoise_method='laplacian', 
#                         denoise_strength=0.3, denoise_iterations=3):
#     """Process multiple meshes for batch testing"""
#     config = read_config(config_path)
#     generator = load_generator(config, model_path)
#     generator.eval()
    
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Get all mesh files
#     mesh_files = [f for f in os.listdir(input_dir) if f.endswith(('.ply', '.obj'))]
#     if num_samples and num_samples < len(mesh_files):
#         mesh_files = mesh_files[:num_samples]
    
#     results = []
    
#     for mesh_file in mesh_files:
#         in_path = os.path.join(input_dir, mesh_file)
#         out_path = os.path.join(output_dir, f"reconstructed_{mesh_file}")
        
#         print(f"\nProcessing {mesh_file}...")
#         reconstructed_mesh, mse, mae = facial_mesh_reconstruction(
#             in_path, out_path, config, generator, 
#             verbose=True, debug=True,
#             denoise=denoise, 
#             denoise_method=denoise_method,
#             denoise_strength=denoise_strength,
#             denoise_iterations=denoise_iterations
#         )
        
#         if reconstructed_mesh is not None:
#             results.append({
#                 'file': mesh_file,
#                 'mse': mse,
#                 'mae': mae
#             })
    
#     # Summarize results
#     if results:
#         avg_mse = np.mean([r['mse'] for r in results])
#         avg_mae = np.mean([r['mae'] for r in results])
#         print(f"\nSummary - Average MSE: {avg_mse:.6f}, Average MAE: {avg_mae:.6f}")
    
#     return results



# # # # # # # # # # # # # # # Local path only
# import torch
# from config.config import read_config
# from trimesh import Trimesh, load_mesh
# from .funcs import load_generator, spherical_regularization_loss, save_ply_explicit, get_random_z, load_norm
# import numpy as np
# import os
# import warnings
# import sys
# from torch_geometric.data import Data
# warnings.filterwarnings("ignore")
# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# sys.setrecursionlimit(30000)

# def load_normalization_params(config):
#     norm_dict = torch.load(os.path.join(config['dataset_dir'], "norm.pt"))
#     mean = norm_dict['mean']
#     std = norm_dict['std']
#     mean_subtraction_only = std is None
#     return mean, std, mean_subtraction_only

# def normalize_vertices(vertices, mean, std=None):
#     normalized = vertices - mean
#     if std is not None:
#         normalized = normalized / std
#     return normalized

# def denormalize_vertices(vertices, mean, std=None):
#     if std is not None:
#         vertices = vertices * std
#     vertices = vertices + mean
#     return vertices

# def facial_mesh_reconstruction(in_file, out_file, config, generator, lambda_reg=None, verbose=True):
#     device = next(generator.parameters()).device
    
#     # Load input mesh
#     mesh_in = load_mesh(in_file)
#     vertices = torch.tensor(mesh_in.vertices, dtype=torch.float).to(device)
    
#     # Load normalization parameters
#     mean, std, mean_subtraction_only = load_normalization_params(config)
#     mean = mean.to(device)
#     if std is not None:
#         std = std.to(device)
    
#     # Normalize input vertices
#     normalized_vertices = normalize_vertices(vertices, mean, std if not mean_subtraction_only else None)
    
#     # Convert vertices to the format expected by the encoder
#     # The encoder expects features as input (x) in the form of a flattened tensor
#     # For a local-only path, we just need to format the vertices as features
#     x = normalized_vertices.reshape(-1, normalized_vertices.shape[-1])  # Reshape to [num_nodes, num_features]
    
#     # Process through the network - using only local path
#     # Create a batch of size 1
#     batch_size = 1
    
#     # Use the encoder directly with the prepared features
#     z = generator.encoder(x, batch_size)
    
#     # Use the decoder to reconstruct vertices
#     reconstructed_vertices = generator.decoder(z, batch_size)
    
#     # Reshape the reconstructed vertices if necessary
#     # If the decoder outputs a flattened tensor, we need to reshape it back to [num_vertices, 3]
#     if reconstructed_vertices.dim() == 2 and reconstructed_vertices.shape[0] != vertices.shape[0]:
#         reconstructed_vertices = reconstructed_vertices.reshape(vertices.shape)
    
#     # Denormalize the output vertices
#     reconstructed_vertices = denormalize_vertices(
#         reconstructed_vertices, 
#         mean,
#         std if not mean_subtraction_only else None
#     )
    
#     # Create and save reconstructed mesh
#     reconstructed_vertices = reconstructed_vertices.detach().cpu().numpy()
#     reconstructed_mesh = Trimesh(reconstructed_vertices, mesh_in.faces)
#     save_ply_explicit(reconstructed_mesh, out_file)
    
#     if verbose:
#         print(f"Reconstructed mesh saved to {out_file}")
    
#     return reconstructed_mesh

# # If you need to create a PyG Data object for your model, you can use this function
# def create_graph_data(vertices, edge_index=None):
#     """
#     Creates a PyG Data object from vertices and optional edge_index
    
#     Args:
#         vertices: Tensor of shape [num_vertices, feature_dim]
#         edge_index: Optional edge connectivity tensor of shape [2, num_edges]
        
#     Returns:
#         PyG Data object
#     """
#     if edge_index is None:
#         # If edge_index is not provided, you might need to create it based on your mesh structure
#         # This is just a placeholder - you should replace with your actual edge creation logic
#         num_vertices = vertices.shape[0]
#         edge_index = torch.empty((2, 0), dtype=torch.long)
        
#     data = Data(x=vertices, edge_index=edge_index)
#     return data

# # If you need to handle PyG batches specifically
# def facial_mesh_reconstruction_with_pyg(in_file, out_file, config, generator, edge_index_creator=None, lambda_reg=None, verbose=True):
#     """
#     Alternative reconstruction function that explicitly uses PyG Data objects
#     """
#     device = next(generator.parameters()).device
    
#     # Load input mesh
#     mesh_in = load_mesh(in_file)
#     vertices = torch.tensor(mesh_in.vertices, dtype=torch.float).to(device)
    
#     # Load normalization parameters
#     mean, std, mean_subtraction_only = load_normalization_params(config)
#     mean = mean.to(device)
#     if std is not None:
#         std = std.to(device)
    
#     # Normalize input vertices
#     normalized_vertices = normalize_vertices(vertices, mean, std if not mean_subtraction_only else None)
    
#     # Create edge index if edge_index_creator function is provided
#     edge_index = None
#     if edge_index_creator is not None:
#         edge_index = edge_index_creator(mesh_in)
#         edge_index = edge_index.to(device)
    
#     # Create a PyG Data object
#     data = create_graph_data(normalized_vertices, edge_index)
    
#     # Create a batch with a single graph
#     # Note: In PyG, you would typically use DataLoader to create batches
#     # For a single graph, we can simulate a batch by adding batch attribute
#     data.batch = torch.zeros(normalized_vertices.shape[0], dtype=torch.long, device=device)
    
#     # Forward pass through the model
#     reconstructed_vertices, z = generator(data)
    
#     # Denormalize the output vertices
#     reconstructed_vertices = denormalize_vertices(
#         reconstructed_vertices, 
#         mean,
#         std if not mean_subtraction_only else None
#     )
    
#     # Create and save reconstructed mesh
#     reconstructed_vertices = reconstructed_vertices.detach().cpu().numpy()
#     reconstructed_mesh = Trimesh(reconstructed_vertices, mesh_in.faces)
#     save_ply_explicit(reconstructed_mesh, out_file)
    
#     if verbose:
#         print(f"Reconstructed mesh saved to {out_file}")
    
#     return reconstructed_mesh, z
############################ Reconstruction normal
# import torch
# from config.config import read_config
# from trimesh import Trimesh, load_mesh
# from .funcs import load_generator, spherical_regularization_loss, save_ply_explicit, get_random_z, load_norm
# import numpy as np
# import os
# import warnings
# import sys
# warnings.filterwarnings("ignore")
# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# sys.setrecursionlimit(30000)
# def load_normalization_params(config):
#     norm_dict = torch.load(os.path.join(config['dataset_dir'], "norm.pt"))
#     mean = norm_dict['mean']
#     std = norm_dict['std']
#     mean_subtraction_only = std is None
#     return mean, std, mean_subtraction_only
# # ################################# for same no. of vertices count
# def normalize_vertices(vertices, mean, std=None):
#     normalized = vertices - mean
#     if std is not None:
#         normalized = normalized / std
#     return normalized
# def denormalize_vertices(vertices, mean, std=None):
#     if std is not None:
#         vertices = vertices * std
#     vertices = vertices + mean
#     return vertices
# def facial_mesh_reconstruction(in_file, out_file, config, generator, lambda_reg=None, verbose=True):
#     device = next(generator.parameters()).device
#     # Load input mesh
#     mesh_in = load_mesh(in_file)
#     vertices = torch.tensor(mesh_in.vertices, dtype=torch.float).to(device)
#     # Load normalization parameters
#     mean, std, mean_subtraction_only = load_normalization_params(config)
#     mean = mean.to(device)
#     if std is not None:
#         std = std.to(device)
#     # Normalize input vertices
#     normalized_vertices = normalize_vertices(vertices, mean, std if not mean_subtraction_only else None)
#     # Process through the network
#     z = generator.encoder(normalized_vertices, 1)
#     reconstructed_vertices = generator.decoder(z, 1)
#     # Denormalize the output vertices
#     reconstructed_vertices = denormalize_vertices(
#         reconstructed_vertices, 
#         mean,
#         std if not mean_subtraction_only else None
#     )
#     # Create and save reconstructed mesh
#     reconstructed_vertices = reconstructed_vertices.detach().cpu().numpy()
#     reconstructed_mesh = Trimesh(reconstructed_vertices, mesh_in.faces)
#     save_ply_explicit(reconstructed_mesh, out_file)
#     if verbose:
#         print(f"Reconstructed mesh saved to {out_file}")
#     return reconstructed_mesh



######debuging
# Add to existing imports
# from trimesh.registration import icp
# from scipy.spatial import cKDTree  # Note correct capitalization

# def align_to_template(source_mesh, template_mesh_path, config):
#     """Ensures 1:1 vertex correspondence between source and template"""
#     template_mesh = load_mesh("D:\Saqib\COSURAI\FaceCom\data\template.ply")
    
#     # Validate topology match
#     if len(source_mesh.vertices) != len(template_mesh.vertices):
#         raise ValueError(f"Vertex count mismatch: Source {len(source_mesh.vertices)} vs Template {len(template_mesh.vertices)}")

#     # 1. Rigid alignment using ICP
#     matrix, transformed, _ = icp(
#         source_mesh.vertices, 
#         template_mesh.vertices,
#         max_iterations=100,
#         reflection=False
#     )
    
#     # 2. Establish vertex correspondence through nearest neighbors
#     template_kdtree = cKDTree(template_mesh.vertices)
#     _, nn_indices = template_kdtree.query(transformed)
    
#     # 3. Reorder source vertices to match template order
#     reordered_vertices = np.zeros_like(template_mesh.vertices)
#     reordered_vertices[nn_indices] = source_mesh.vertices
    
#     return Trimesh(vertices=reordered_vertices, faces=template_mesh.faces)

# def facial_mesh_reconstruction(in_file, out_file, config, generator, lambda_reg=None, verbose=True, template_path=None):
#     """
#     Reconstruct facial mesh with diagnostic information and alignment
    
#     Args:
#         in_file (str): Input mesh file path
#         out_file (str): Output mesh file path
#         config (dict): Configuration dictionary
#         generator: Generator model
#         lambda_reg (float, optional): Regularization parameter
#         verbose (bool, optional): Whether to print progress
#         template_path (str, optional): Path to template mesh for alignment
    
#     Returns:
#         trimesh.Trimesh: Reconstructed mesh
#     """
#     device = next(generator.parameters()).device
    
#     # Load input mesh
#     mesh_in = load_mesh(in_file)
    
#     # Align mesh to template if provided
#     if template_path is not None:
#         mesh_in = align_to_template(mesh_in, template_path, config)
    
#     vertices = torch.tensor(mesh_in.vertices, dtype=torch.float).to(device)
    
#     # Print diagnostic information about input mesh
#     if verbose:
#         print(f"\n----- MESH DIAGNOSTICS: {os.path.basename(in_file)} -----")
#         print(f"Vertices count: {len(mesh_in.vertices)}")
#         print(f"Faces count: {len(mesh_in.faces)}")
#         print(f"Mesh volume: {mesh_in.volume:.4f}")
#         print(f"Is watertight: {mesh_in.is_watertight}")
#         print(f"Is winding consistent: {mesh_in.is_winding_consistent}")
        
#         # Check for degenerate faces
#         degen_faces = sum(1 for f in mesh_in.faces if len(set(f)) < 3)
#         print(f"Degenerate faces: {degen_faces}")
        
#         # Check vertex range and distribution
#         v_min = mesh_in.vertices.min(axis=0)
#         v_max = mesh_in.vertices.max(axis=0)
#         v_range = v_max - v_min
#         print(f"Vertex range: x=[{v_min[0]:.4f}, {v_max[0]:.4f}], y=[{v_min[1]:.4f}, {v_max[1]:.4f}], z=[{v_min[2]:.4f}, {v_max[2]:.4f}]")
#         print(f"Mesh bounding box: {v_range}")
    
#     # Load normalization parameters
#     mean, std, mean_subtraction_only = load_normalization_params(config)
#     mean = mean.to(device)
#     if std is not None:
#         std = std.to(device)
    
#     # Print normalization stats
#     if verbose:
#         print("\n----- NORMALIZATION PARAMETERS -----")
#         print(f"Mean: {mean.cpu().numpy()}")
#         print(f"Using mean subtraction only: {mean_subtraction_only}")
#         if not mean_subtraction_only:
#             print(f"Std: {std.cpu().numpy()}")
            
#         # Check if vertices are within expected normalization range
#         normalized_vertices = normalize_vertices(vertices, mean, std if not mean_subtraction_only else None)
#         norm_min = normalized_vertices.min(dim=0)[0].cpu().numpy()
#         norm_max = normalized_vertices.max(dim=0)[0].cpu().numpy()
#         print(f"Normalized vertex range: x=[{norm_min[0]:.4f}, {norm_max[0]:.4f}], y=[{norm_min[1]:.4f}, {norm_max[1]:.4f}], z=[{norm_min[2]:.4f}, {norm_max[2]:.4f}]")
        
#         # Standard range should typically be around [-1, 1] or [-3, 3] depending on your model
#         expected_range = 3.0
#         if np.any(np.abs(norm_min) > expected_range) or np.any(np.abs(norm_max) > expected_range):
#             print(f"WARNING: Normalized vertices outside expected range [-{expected_range}, {expected_range}]")
            
#             # If extreme values detected and no alignment done, recommend alignment
#             if template_path is None and (np.any(np.abs(norm_min) > 10) or np.any(np.abs(norm_max) > 10)):
#                 print("CRITICAL: Extreme normalization values detected! Consider using the template_path parameter for vertex alignment.")
    
#     # Normalize input vertices
#     normalized_vertices = normalize_vertices(vertices, mean, std if not mean_subtraction_only else None)
    
#     # Process through the network
#     z = generator.encoder(normalized_vertices, 1)
    
#     # Check latent code for anomalies
#     if verbose:
#         print("\n----- LATENT CODE DIAGNOSTICS -----")
#         z_np = z.detach().cpu().numpy()
#         print(f"Latent code shape: {z_np.shape}")
#         print(f"Latent code range: [{z_np.min():.4f}, {z_np.max():.4f}]")
#         print(f"Latent code mean: {z_np.mean():.4f}")
#         print(f"Latent code std: {z_np.std():.4f}")
        
#         # Check if latent code has extreme values
#         if np.abs(z_np).max() > 5.0:
#             print("WARNING: Extreme values in latent code - may indicate normalization issues")
    
#     reconstructed_vertices = generator.decoder(z, 1)
    
#     # Check reconstructed vertices before denormalization
#     if verbose:
#         print("\n----- RECONSTRUCTED VERTICES (BEFORE DENORMALIZATION) -----")
#         recon_norm = reconstructed_vertices.detach().cpu().numpy()
#         print(f"Min values: {recon_norm.min(axis=0)}")
#         print(f"Max values: {recon_norm.max(axis=0)}")
        
#         # Check for NaN or Inf values
#         if np.isnan(recon_norm).any() or np.isinf(recon_norm).any():
#             print("WARNING: NaN or Inf values detected in reconstructed vertices!")
    
#     # Denormalize the output vertices
#     reconstructed_vertices = denormalize_vertices(
#         reconstructed_vertices, 
#         mean,
#         std if not mean_subtraction_only else None
#     )
    
#     # Final reconstruction diagnostics
#     if verbose:
#         print("\n----- FINAL RECONSTRUCTION DIAGNOSTICS -----")
#         recon_final = reconstructed_vertices.detach().cpu().numpy()
#         print(f"Min values: {recon_final.min(axis=0)}")
#         print(f"Max values: {recon_final.max(axis=0)}")
        
#         # Calculate reconstruction error
#         input_vertices_np = vertices.detach().cpu().numpy()
#         mean_error = np.mean(np.sqrt(np.sum((recon_final - input_vertices_np)**2, axis=1)))
#         max_error = np.max(np.sqrt(np.sum((recon_final - input_vertices_np)**2, axis=1)))
#         print(f"Mean vertex distance error: {mean_error:.6f}")
#         print(f"Max vertex distance error: {max_error:.6f}")
        
#         # Check for spikes in the mesh
#         vertex_diffs = np.abs(np.diff(recon_final, axis=0))
#         max_diff = np.max(vertex_diffs)
#         mean_diff = np.mean(vertex_diffs)
#         print(f"Max adjacent vertex difference: {max_diff:.6f}")
#         print(f"Mean adjacent vertex difference: {mean_diff:.6f}")
        
#         if max_diff > 10 * mean_diff:
#             print("WARNING: Large differences between adjacent vertices detected - likely spikes in the mesh")
    
#     # Create and save reconstructed mesh
#     reconstructed_vertices = reconstructed_vertices.detach().cpu().numpy()
#     reconstructed_mesh = Trimesh(reconstructed_vertices, mesh_in.faces)
#     save_ply_explicit(reconstructed_mesh, out_file)
    
#     if verbose:
#         print(f"\nReconstructed mesh saved to {out_file}")
#         print("----- RECONSTRUCTION COMPLETE -----\n")
    
#     return reconstructed_mesh

########################################## for ignoring teh diff no. of vertices count
# def normalize_vertices(vertices, mean, std=None):
#     """
#     Normalize vertices using mean and optionally std with error handling
    
#     Args:
#         vertices (torch.Tensor): Input vertices
#         mean (torch.Tensor): Mean for normalization
#         std (torch.Tensor, optional): Standard deviation for normalization
    
#     Returns:
#         torch.Tensor: Normalized vertices
#     """
#     try:
#         normalized = vertices - mean
#         if std is not None:
#             normalized = normalized / std
#         return normalized
#     except RuntimeError as e:
#         # Handle size mismatch
#         print(f"Warning: Size mismatch in normalize_vertices: {e}")
#         print(f"Vertices shape: {vertices.shape}, Mean shape: {mean.shape}")
        
#         # Option 1: Return original vertices (skip normalization)
#         return vertices
        
#         # Option 2 (alternative): Resize mean/std to match vertices
#         # if vertices.shape[0] < mean.shape[0]:
#         #     resized_mean = mean[:vertices.shape[0]]
#         #     resized_std = None if std is None else std[:vertices.shape[0]]
#         #     normalized = vertices - resized_mean
#         #     if resized_std is not None:
#         #         normalized = normalized / resized_std
#         #     return normalized
#         # else:
#         #     # Handle case where vertices is larger than mean
#         #     print("Cannot normalize: vertices has more points than normalization data")
#         #     return vertices

# def denormalize_vertices(vertices, mean, std=None):
#     """
#     Denormalize vertices using mean and optionally std with error handling
    
#     Args:
#         vertices (torch.Tensor): Normalized vertices
#         mean (torch.Tensor): Mean for denormalization
#         std (torch.Tensor, optional): Standard deviation for denormalization
    
#     Returns:
#         torch.Tensor: Denormalized vertices
#     """
#     try:
#         if std is not None:
#             vertices = vertices * std
#         vertices = vertices + mean
#         return vertices
#     except RuntimeError as e:
#         # Handle size mismatch
#         print(f"Warning: Size mismatch in denormalize_vertices: {e}")
#         print(f"Vertices shape: {vertices.shape}, Mean shape: {mean.shape}")
        
#         # Return original vertices if mismatch occurs
#         return vertices

# def facial_mesh_reconstruction(in_file, out_file, config, generator, lambda_reg=None, verbose=True):
#     """
#     Reconstruct facial mesh with error handling for size mismatches
    
#     Args:
#         in_file (str): Input mesh file path
#         out_file (str): Output mesh file path
#         config (dict): Configuration dictionary
#         generator: Generator model
#         lambda_reg (float, optional): Regularization parameter
#         verbose (bool, optional): Whether to print progress
    
#     Returns:
#         trimesh.Trimesh: Reconstructed mesh or None if error occurs
#     """
#     try:
#         device = next(generator.parameters()).device
        
#         # Load input mesh
#         mesh_in = load_mesh(in_file)
#         vertices = torch.tensor(mesh_in.vertices, dtype=torch.float).to(device)
        
#         # Load normalization parameters
#         mean, std, mean_subtraction_only = load_normalization_params(config)
#         mean = mean.to(device)
#         if std is not None:
#             std = std.to(device)
        
#         # Print shapes for debugging
#         if verbose:
#             print(f"Input vertices shape: {vertices.shape}")
#             print(f"Mean shape: {mean.shape}")
#             if std is not None:
#                 print(f"Std shape: {std.shape}")
        
#         # Check for size mismatch before normalizing
#         if vertices.shape[0] != mean.shape[0]:
#             print(f"Warning: Size mismatch between vertices ({vertices.shape[0]}) and mean ({mean.shape[0]})")
#             print(f"File: {in_file}")
#             print("Using original vertices without normalization")
#             reconstructed_vertices = vertices.detach().cpu().numpy()
#         else:
#             # Normalize input vertices
#             normalized_vertices = normalize_vertices(vertices, mean, std if not mean_subtraction_only else None)
            
#             # Process through the network
#             z = generator.encoder(normalized_vertices, 1)
#             reconstructed_vertices = generator.decoder(z, 1)
            
#             # Denormalize the output vertices
#             reconstructed_vertices = denormalize_vertices(
#                 reconstructed_vertices, 
#                 mean,
#                 std if not mean_subtraction_only else None
#             )
#             reconstructed_vertices = reconstructed_vertices.detach().cpu().numpy()
        
#         # Create and save reconstructed mesh
#         reconstructed_mesh = Trimesh(reconstructed_vertices, mesh_in.faces)
#         save_ply_explicit(reconstructed_mesh, out_file)
        
#         if verbose:
#             print(f"Reconstructed mesh saved to {out_file}")
        
#         return reconstructed_mesh
    
#     except Exception as e:
#         print(f"Error processing {in_file}: {e}")
#         print(f"Skipping this file and continuing...")
#         return None

# ########################################



# # ################################## Mesh Reconstructon original without normlaization with mean sub
# # def facial_mesh_reconstruction(in_file, out_file, config, generator, lambda_reg=None, verbose=True):
# #     device = next(generator.parameters()).device
# #     mesh_in = load_mesh(in_file)
    
# #     # Convert input mesh vertices to torch tensor and move to the correct device
# #     vertices = torch.tensor(mesh_in.vertices, dtype=torch.float).to(device)
    
# #     # Normalize vertices using mean and std from config
# #     mean, std = load_norm(config)
# #     mean = mean.to(device)
# #     std = std.to(device)

# #     # Normalize input vertices
# #     normalized_vertices = (vertices - mean) / std

# #     # Pass vertices through encoder and decoder
# #     z = generator.encoder(normalized_vertices, 1)  # Get latent representation from encoder
# #     reconstructed_vertices = generator.decoder(z, 1)  # Decode latent space to 3D mesh

# #     # Denormalize the output mesh
# #     reconstructed_vertices = reconstructed_vertices * std + mean

# #     # Move the result back to CPU for further processing
# #     reconstructed_vertices = reconstructed_vertices.detach().cpu().numpy()

# #     # Create reconstructed mesh and save it
# #     reconstructed_mesh = Trimesh(reconstructed_vertices, mesh_in.faces)
# #     save_ply_explicit(reconstructed_mesh, out_file)

# #     if verbose:
# #         print(f"Reconstructed mesh saved to {out_file}")
    
# #     return reconstructed_mesh



# def rigid_registration(in_mesh, config, verbose=True):
#     if verbose:
#         print("rigid registration...")

#     # mesh = in_mesh.copy()
#     mesh = in_mesh
#     template_mesh = load_mesh(config["template_file"])

#     centroid = mesh.centroid
#     mesh.vertices -= mesh.centroid
#     T, _, _ = icp(mesh.vertices, template_mesh.vertices, max_iterations=50)
#     mesh.apply_transform(T)

#     return mesh, T, centroid


# def fit(in_mesh, generator, config, device, max_iters=1000, loss_convergence=1e-6, lambda_reg=None,
#         verbose=True, dis_percent=None):
#     if verbose:
#         sys.stdout.write("\rFitting...")
#         sys.stdout.flush()

#     mesh = in_mesh.copy()
#     template_mesh = load_mesh(config["template_file"])

#     generator.eval()

#     target_pc = torch.tensor(mesh.vertices, dtype=torch.float).to(device)

#     z = get_random_z(generator.z_length, requires_grad=True, jitter=True)

#     mean, std = load_norm(config)
#     mean = mean.to(device)
#     std = std.to(device)
#     faces = torch.tensor(template_mesh.faces).to(device)

#     optimizer = torch.optim.Adam([z], lr=0.1)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

#     if not lambda_reg:
#         lambda_reg = config['lambda_reg']
#     last_loss = math.inf
#     iters = 0
#     for i in range(max_iters):
#         optimizer.zero_grad()

#         out = generator(z.to(device), 1)
#         out = out * std + mean

#         loss_reg = spherical_regularization_loss(z)
#         loss = loss_reg

#         distance = torch.sqrt(distance_from_reference_mesh(target_pc, out, faces))
#         if dis_percent:
#             # 只取距离最小的一部分顶点
#             distance, idx = torch.sort(distance)
#             distance = distance[:int(dis_percent * len(distance))]
#         loss_dfrm = torch.mean(distance)
#         loss = loss_dfrm + lambda_reg * loss_reg

#         if verbose:
#             sys.stdout.write(
#                 "\rFitting...\tIter {}, loss_recon: {:.6f}, loss_reg: {:.6f}".format(i + 1,
#                                                                                       loss_dfrm.item(),
#                                                                                       loss_reg.item()))
#             sys.stdout.flush()
#         if math.fabs(last_loss - loss.item()) < loss_convergence:
#             iters = i
#             break

#         last_loss = loss.item()
#         loss.backward()
#         optimizer.step()
#         scheduler.step()

#     out = generator(z.to(device), 1)
#     out = out * std + mean
#     fit_mesh = Trimesh(out.detach().cpu(), template_mesh.faces)
#     if verbose:
#         print("")

#     return fit_mesh


# def post_processing(in_mesh_fit, in_mesh_faulty, device, laplacian=True, verbose=True):
#     if verbose:
#         print("post processing...")

#     def get_color_mesh(mesh, idx, init_color=True, color=None):
#         if color is None:
#             color = [255, 0, 0, 255]
#         color_mesh = mesh.copy()

#         if init_color:
#             color_array = np.zeros((mesh.vertices.shape[0], 4), dtype=np.uint8)  # RGBA颜色
#             color_array[idx] = color
#             color_mesh.visual.vertex_colors = color_array
#         else:
#             color_mesh.visual.vertex_colors[idx] = color
#         return color_mesh

#     def extract_connected_components(mesh: Trimesh, idx):
#         visited = set()
#         components = []

#         def dfs(vertex, component):
#             if vertex in visited:
#                 return
#             visited.add(vertex)
#             component.add(vertex)
#             for neighbor in mesh.vertex_neighbors[vertex]:
#                 if neighbor in idx:
#                     dfs(neighbor, component)

#         for vertex in idx:
#             if vertex not in visited:
#                 component = set()
#                 dfs(vertex, component)
#                 components.append(component)

#         return components

#     def expand_connected_component(mesh, component_, distance):
#         expanded_component = set()
#         component = component_.copy()

#         for _ in range(distance):
#             new_neighbors = set()
#             for vertex in component:
#                 neighbors = mesh.vertex_neighbors[vertex]
#                 for neighbor in neighbors:
#                     if neighbor not in component and neighbor not in expanded_component:
#                         new_neighbors.add(neighbor)
#             expanded_component.update(new_neighbors)
#             component.update(new_neighbors)

#         return expanded_component

#     def special_point_refinement(mesh: Trimesh):
#         vertices = mesh.vertices
#         for i in tqdm(range(mesh.vertices.shape[0])):
#             neighbor = mesh.vertex_neighbors[i]
#             mean_x = np.mean(vertices[neighbor], axis=0)
#             x = vertices[i]
#             mean_distance = np.mean(np.linalg.norm(mesh.vertices[neighbor] - mean_x, axis=1))
#             if np.linalg.norm(mean_x - x) > 0.5 * mean_distance:
#                 vertices[i] = mean_x
#         return Trimesh(vertices, mesh.faces)

#     def projection(source_mesh: Trimesh, largest_component_mask, target_mesh: Trimesh, max_iters=1000):
#         x = torch.tensor(source_mesh.vertices, dtype=torch.float).to(device)
#         normal_vectors = torch.tensor(source_mesh.vertex_normals, dtype=torch.float).to(device)
#         ndf = torch.randn(source_mesh.vertices.shape[0]).detach().to(device)
#         ndf.requires_grad = True

#         optimizer = torch.optim.Adam([ndf], lr=0.1)
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

#         last_loss = math.inf
#         for i in range(max_iters):
#             optimizer.zero_grad()

#             out = x + normal_vectors * torch.unsqueeze(ndf, 1)
#             distance = distance_from_reference_mesh(out[~largest_component_mask],
#                                                     torch.tensor(target_mesh.vertices, dtype=torch.float).to(device),
#                                                     torch.tensor(target_mesh.faces).to(device))
#             distance = torch.sqrt(distance)
#             loss_dfrm = torch.mean(distance)

#             loss_smoothness = smoothness_loss(out, torch.tensor(source_mesh.faces).to(device))
#             # 投影时保持平滑，防止某些顶点投影到别的曲面上去

#             # loss = loss_dfrm + 1 * loss_smoothness
#             loss = loss_dfrm

#             if verbose:
#                 sys.stdout.write("\rProjection... Iter {}, Loss: {}".format(i + 1, loss.item()))
#                 sys.stdout.flush()
#             if i > 100 and math.fabs(last_loss - loss.item()) < 1e-6:
#                 break

#             last_loss = loss.item()
#             loss.backward()
#             optimizer.step()
#             scheduler.step()

#         out = x + normal_vectors * torch.unsqueeze(ndf, 1)
#         out = out.detach().cpu()

#         # 还要把largest_component_mask对应顶点变为原来的顶点
#         out[largest_component_mask] = torch.tensor(source_mesh.vertices, dtype=torch.float)[largest_component_mask]

#         new_mesh = Trimesh(out, source_mesh.faces)
#         if verbose:
#             print("")

#         return new_mesh

#     def find_nearest_vertices(target_vertices, source_vertices, k=1):
#         tree = cKDTree(source_vertices)
#         distances, indices = tree.query(target_vertices, k=k)
#         return indices, distances

#     # 读取mesh
#     fit_mesh = in_mesh_fit.copy()
#     faulty_mesh = in_mesh_faulty.copy()

#     # 1.阈值法识别fit_mesh中缺损部分（“修补”部分）
#     distance = distance_from_reference_mesh(torch.tensor(fit_mesh.vertices, dtype=torch.float).to(device),
#                                             torch.tensor(faulty_mesh.vertices, dtype=torch.float).to(device),
#                                             torch.tensor(faulty_mesh.faces).to(device)).cpu().numpy()
#     idx = np.where(distance > 4)[0]  # 阈值
#     color_mesh = get_color_mesh(fit_mesh, idx)
#     # color_mesh.export(join(out_path, "color_1.ply"))

#     # 2.计算最大联通分量（缺损部分）以及扩展部分
#     connected_components = extract_connected_components(fit_mesh, idx)
#     if len(connected_components) == 0:
#         return fit_mesh
#     largest_component = max(connected_components, key=len)
#     expanded_component = expand_connected_component(fit_mesh, largest_component, 2)
#     color_mesh = get_color_mesh(fit_mesh, list(largest_component))
#     color_mesh = get_color_mesh(color_mesh, list(expanded_component), False, [0, 255, 0, 255])
#     # color_mesh.export(join(out_path, 'color_2.ply'))

#     # 3.最优化法向位移场（投影）
#     vertex_mask = np.zeros(len(fit_mesh.vertices), dtype=bool)
#     vertex_mask[list(largest_component)] = True
#     projection_mesh = projection(fit_mesh, vertex_mask, faulty_mesh)
#     # projection_mesh.export(join(out_path, "projection.ply"))

#     # 4.将最大联通分量的顶点逐一进行位移。 位移量：扩展部分中K个最近顶点[投影]时位移的均值
#     vertices_expanded = fit_mesh.vertices[list(expanded_component)]
#     normal_displacement = (projection_mesh.vertices - fit_mesh.vertices)[list(expanded_component)]
#     indices, distances = find_nearest_vertices(fit_mesh.vertices, vertices_expanded, k=15)
#     completion_mesh = projection_mesh.copy()
#     for id in largest_component:
#         mean_displacement = np.mean(normal_displacement[indices[id]], axis=0)
#         completion_mesh.vertices[id] += mean_displacement

#     def laplacian_smoothing(iterations=2, smoothing_factor=0.5):
#         vertices_to_smooth = list(expanded_component)
#         vertices_to_smooth.extend(list(largest_component))

#         # 循环进行拉普拉斯平滑处理
#         for _ in range(iterations):
#             smoothed_vertices = []
#             for vertex_index in vertices_to_smooth:
#                 vertex = completion_mesh.vertices[vertex_index]
#                 neighbors = completion_mesh.vertex_neighbors[vertex_index]
#                 neighbor_vertices = completion_mesh.vertices[neighbors]
#                 smoothed_vertex = vertex + smoothing_factor * np.mean(neighbor_vertices - vertex, axis=0)
#                 smoothed_vertices.append(smoothed_vertex)
#             # 更新要平滑的顶点坐标
#             for i, vertex_index in enumerate(vertices_to_smooth):
#                 completion_mesh.vertices[vertex_index] = smoothed_vertices[i]

#     # 5.拉普拉斯平滑处理修复区域
#     if laplacian:
#         laplacian_smoothing()


#     if verbose:
#         print("refinement")
#     for i in range(5):
#         completion_mesh = special_point_refinement(completion_mesh)
#     # completion_mesh.export(join(out_path, 'refinement.ply'))

#     # done
#     print("done!")
#     return completion_mesh


# def facial_mesh_completion(in_file, out_file, config, generator, lambda_reg=None, verbose=True, rr=False,
#                             dis_percent=None):
#     dir = os.path.dirname(in_file)
#     device = generator.parameters().__next__().device

#     mesh_in = load_mesh(in_file)

#     if rr:
#         mesh_in, T, centroid = rigid_registration(mesh_in, config, verbose=verbose)

#     # save_ply_explicit(mesh_in, "rr.ply")

#     mesh_fit = fit(mesh_in, generator, config, device, lambda_reg=lambda_reg, verbose=verbose, loss_convergence=1e-7,
#                     dis_percent=dis_percent)
#     mesh_com = post_processing(mesh_fit, mesh_in, device, verbose=verbose)

#     if rr:
#         mesh_com.apply_transform(np.linalg.inv(T))
#         mesh_com.vertices += centroid

#     save_ply_explicit(mesh_com, out_file)
    
#     return mesh_com




###############################Backup Reconstruion Feb 4
# import torch
# import torchvision.utils
# from config.config import read_config
# from trimesh import Trimesh, load_mesh
# from .funcs import load_generator, spherical_regularization_loss, save_ply_explicit, get_random_z, load_norm
# import torch.nn.functional as F
# import os
# from os.path import join
# import warnings
# from tqdm import tqdm
# import numpy as np
# from queue import Queue
# import math
# from pytorch3d.structures import Meshes
# from pytorch3d.loss import chamfer_distance
# from .pytorch3d_extend import distance_from_reference_mesh, smoothness_loss
# from trimesh.registration import icp
# from scipy.spatial import cKDTree
# import sys
# from PIL import Image
# import torchvision.transforms as transforms
# from .render import render_d
# from .funcs import load_generator, spherical_regularization_loss, save_ply_explicit, get_random_z, load_norm
# # from .funcs import load_model, spherical_regularization_loss, save_ply_explicit, get_random_z, load_norm




# warnings.filterwarnings("ignore")
# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# sys.setrecursionlimit(30000)

# def load_normalization_params(config):
#     """
#     Load normalization parameters and check if using mean subtraction only
    
#     Args:
#         config (dict): Configuration dictionary containing normalization parameters
    
#     Returns:
#         tuple: (mean, std, mean_subtraction_only)
#     """
#     norm_dict = torch.load(os.path.join(config['dataset_dir'], "norm.pt"))
#     mean = norm_dict['mean']
#     std = norm_dict['std']
#     # Check if we're using mean subtraction only (std will be None in this case)
#     mean_subtraction_only = std is None
#     return mean, std, mean_subtraction_only

# def normalize_vertices(vertices, mean, std=None):
#     """
#     Normalize vertices using mean and optionally std
    
#     Args:
#         vertices (torch.Tensor): Input vertices
#         mean (torch.Tensor): Mean for normalization
#         std (torch.Tensor, optional): Standard deviation for normalization
    
#     Returns:
#         torch.Tensor: Normalized vertices
#     """
#     normalized = vertices - mean
#     if std is not None:
#         normalized = normalized / std
#     return normalized

# def denormalize_vertices(vertices, mean, std=None):
#     """
#     Denormalize vertices using mean and optionally std
    
#     Args:
#         vertices (torch.Tensor): Normalized vertices
#         mean (torch.Tensor): Mean for denormalization
#         std (torch.Tensor, optional): Standard deviation for denormalization
    
#     Returns:
#         torch.Tensor: Denormalized vertices
#     """
#     if std is not None:
#         vertices = vertices * std
#     vertices = vertices + mean
#     return vertices

# def facial_mesh_reconstruction(in_file, out_file, config, generator, lambda_reg=None, verbose=True):
#     """
#     Reconstruct facial mesh with updated normalization handling
    
#     Args:
#         in_file (str): Input mesh file path
#         out_file (str): Output mesh file path
#         config (dict): Configuration dictionary
#         generator: Generator model
#         lambda_reg (float, optional): Regularization parameter
#         verbose (bool, optional): Whether to print progress
    
#     Returns:
#         trimesh.Trimesh: Reconstructed mesh
#     """
#     device = next(generator.parameters()).device
    
#     # Load input mesh
#     mesh_in = load_mesh(in_file)
#     vertices = torch.tensor(mesh_in.vertices, dtype=torch.float).to(device)
    
#     # Load normalization parameters
#     mean, std, mean_subtraction_only = load_normalization_params(config)
#     mean = mean.to(device)
#     if std is not None:
#         std = std.to(device)
    
#     # Normalize input vertices
#     normalized_vertices = normalize_vertices(vertices, mean, std if not mean_subtraction_only else None)
    
#     # Process through the network
#     z = generator.encoder(normalized_vertices, 1)
#     reconstructed_vertices = generator.decoder(z, 1)
    
#     # Denormalize the output vertices
#     reconstructed_vertices = denormalize_vertices(
#         reconstructed_vertices, 
#         mean,
#         std if not mean_subtraction_only else None
#     )
    
#     # Create and save reconstructed mesh
#     reconstructed_vertices = reconstructed_vertices.detach().cpu().numpy()
#     reconstructed_mesh = Trimesh(reconstructed_vertices, mesh_in.faces)
#     save_ply_explicit(reconstructed_mesh, out_file)
    
#     if verbose:
#         print(f"Reconstructed mesh saved to {out_file}")
    
#     return reconstructed_mesh

# ########################################

# def generate_face_sample(out_file, config, generator):
#     generator.eval()
#     device = generator.parameters().__next__().device
#     z = get_random_z(generator.z_length, requires_grad=False)
#     mean, std = load_norm(config)
#     out = generator(z.to(device), 1).detach().cpu()
#     out = out * std + mean
#     template_mesh = load_mesh(config["template_file"])
#     mesh = Trimesh(out, template_mesh.faces)
#     save_ply_explicit(mesh, out_file)

# # ################################## Mesh Reconstructon original without normlaization with mean sub
# # def facial_mesh_reconstruction(in_file, out_file, config, generator, lambda_reg=None, verbose=True):
# #     device = next(generator.parameters()).device
# #     mesh_in = load_mesh(in_file)
    
# #     # Convert input mesh vertices to torch tensor and move to the correct device
# #     vertices = torch.tensor(mesh_in.vertices, dtype=torch.float).to(device)
    
# #     # Normalize vertices using mean and std from config
# #     mean, std = load_norm(config)
# #     mean = mean.to(device)
# #     std = std.to(device)

# #     # Normalize input vertices
# #     normalized_vertices = (vertices - mean) / std

# #     # Pass vertices through encoder and decoder
# #     z = generator.encoder(normalized_vertices, 1)  # Get latent representation from encoder
# #     reconstructed_vertices = generator.decoder(z, 1)  # Decode latent space to 3D mesh

# #     # Denormalize the output mesh
# #     reconstructed_vertices = reconstructed_vertices * std + mean

# #     # Move the result back to CPU for further processing
# #     reconstructed_vertices = reconstructed_vertices.detach().cpu().numpy()

# #     # Create reconstructed mesh and save it
# #     reconstructed_mesh = Trimesh(reconstructed_vertices, mesh_in.faces)
# #     save_ply_explicit(reconstructed_mesh, out_file)

# #     if verbose:
# #         print(f"Reconstructed mesh saved to {out_file}")
    
# #     return reconstructed_mesh



# def rigid_registration(in_mesh, config, verbose=True):
#     if verbose:
#         print("rigid registration...")

#     # mesh = in_mesh.copy()
#     mesh = in_mesh
#     template_mesh = load_mesh(config["template_file"])

#     centroid = mesh.centroid
#     mesh.vertices -= mesh.centroid
#     T, _, _ = icp(mesh.vertices, template_mesh.vertices, max_iterations=50)
#     mesh.apply_transform(T)

#     return mesh, T, centroid


# def fit(in_mesh, generator, config, device, max_iters=1000, loss_convergence=1e-6, lambda_reg=None,
#         verbose=True, dis_percent=None):
#     if verbose:
#         sys.stdout.write("\rFitting...")
#         sys.stdout.flush()

#     mesh = in_mesh.copy()
#     template_mesh = load_mesh(config["template_file"])

#     generator.eval()

#     target_pc = torch.tensor(mesh.vertices, dtype=torch.float).to(device)

#     z = get_random_z(generator.z_length, requires_grad=True, jitter=True)

#     mean, std = load_norm(config)
#     mean = mean.to(device)
#     std = std.to(device)
#     faces = torch.tensor(template_mesh.faces).to(device)

#     optimizer = torch.optim.Adam([z], lr=0.1)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

#     if not lambda_reg:
#         lambda_reg = config['lambda_reg']
#     last_loss = math.inf
#     iters = 0
#     for i in range(max_iters):
#         optimizer.zero_grad()

#         out = generator(z.to(device), 1)
#         out = out * std + mean

#         loss_reg = spherical_regularization_loss(z)
#         loss = loss_reg

#         distance = torch.sqrt(distance_from_reference_mesh(target_pc, out, faces))
#         if dis_percent:
#             # 只取距离最小的一部分顶点
#             distance, idx = torch.sort(distance)
#             distance = distance[:int(dis_percent * len(distance))]
#         loss_dfrm = torch.mean(distance)
#         loss = loss_dfrm + lambda_reg * loss_reg

#         if verbose:
#             sys.stdout.write(
#                 "\rFitting...\tIter {}, loss_recon: {:.6f}, loss_reg: {:.6f}".format(i + 1,
#                                                                                       loss_dfrm.item(),
#                                                                                       loss_reg.item()))
#             sys.stdout.flush()
#         if math.fabs(last_loss - loss.item()) < loss_convergence:
#             iters = i
#             break

#         last_loss = loss.item()
#         loss.backward()
#         optimizer.step()
#         scheduler.step()

#     out = generator(z.to(device), 1)
#     out = out * std + mean
#     fit_mesh = Trimesh(out.detach().cpu(), template_mesh.faces)
#     if verbose:
#         print("")

#     return fit_mesh


# def post_processing(in_mesh_fit, in_mesh_faulty, device, laplacian=True, verbose=True):
#     if verbose:
#         print("post processing...")

#     def get_color_mesh(mesh, idx, init_color=True, color=None):
#         if color is None:
#             color = [255, 0, 0, 255]
#         color_mesh = mesh.copy()

#         if init_color:
#             color_array = np.zeros((mesh.vertices.shape[0], 4), dtype=np.uint8)  # RGBA颜色
#             color_array[idx] = color
#             color_mesh.visual.vertex_colors = color_array
#         else:
#             color_mesh.visual.vertex_colors[idx] = color
#         return color_mesh

#     def extract_connected_components(mesh: Trimesh, idx):
#         visited = set()
#         components = []

#         def dfs(vertex, component):
#             if vertex in visited:
#                 return
#             visited.add(vertex)
#             component.add(vertex)
#             for neighbor in mesh.vertex_neighbors[vertex]:
#                 if neighbor in idx:
#                     dfs(neighbor, component)

#         for vertex in idx:
#             if vertex not in visited:
#                 component = set()
#                 dfs(vertex, component)
#                 components.append(component)

#         return components

#     def expand_connected_component(mesh, component_, distance):
#         expanded_component = set()
#         component = component_.copy()

#         for _ in range(distance):
#             new_neighbors = set()
#             for vertex in component:
#                 neighbors = mesh.vertex_neighbors[vertex]
#                 for neighbor in neighbors:
#                     if neighbor not in component and neighbor not in expanded_component:
#                         new_neighbors.add(neighbor)
#             expanded_component.update(new_neighbors)
#             component.update(new_neighbors)

#         return expanded_component

#     def special_point_refinement(mesh: Trimesh):
#         vertices = mesh.vertices
#         for i in tqdm(range(mesh.vertices.shape[0])):
#             neighbor = mesh.vertex_neighbors[i]
#             mean_x = np.mean(vertices[neighbor], axis=0)
#             x = vertices[i]
#             mean_distance = np.mean(np.linalg.norm(mesh.vertices[neighbor] - mean_x, axis=1))
#             if np.linalg.norm(mean_x - x) > 0.5 * mean_distance:
#                 vertices[i] = mean_x
#         return Trimesh(vertices, mesh.faces)

#     def projection(source_mesh: Trimesh, largest_component_mask, target_mesh: Trimesh, max_iters=1000):
#         x = torch.tensor(source_mesh.vertices, dtype=torch.float).to(device)
#         normal_vectors = torch.tensor(source_mesh.vertex_normals, dtype=torch.float).to(device)
#         ndf = torch.randn(source_mesh.vertices.shape[0]).detach().to(device)
#         ndf.requires_grad = True

#         optimizer = torch.optim.Adam([ndf], lr=0.1)
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

#         last_loss = math.inf
#         for i in range(max_iters):
#             optimizer.zero_grad()

#             out = x + normal_vectors * torch.unsqueeze(ndf, 1)
#             distance = distance_from_reference_mesh(out[~largest_component_mask],
#                                                     torch.tensor(target_mesh.vertices, dtype=torch.float).to(device),
#                                                     torch.tensor(target_mesh.faces).to(device))
#             distance = torch.sqrt(distance)
#             loss_dfrm = torch.mean(distance)

#             loss_smoothness = smoothness_loss(out, torch.tensor(source_mesh.faces).to(device))
#             # 投影时保持平滑，防止某些顶点投影到别的曲面上去

#             # loss = loss_dfrm + 1 * loss_smoothness
#             loss = loss_dfrm

#             if verbose:
#                 sys.stdout.write("\rProjection... Iter {}, Loss: {}".format(i + 1, loss.item()))
#                 sys.stdout.flush()
#             if i > 100 and math.fabs(last_loss - loss.item()) < 1e-6:
#                 break

#             last_loss = loss.item()
#             loss.backward()
#             optimizer.step()
#             scheduler.step()

#         out = x + normal_vectors * torch.unsqueeze(ndf, 1)
#         out = out.detach().cpu()

#         # 还要把largest_component_mask对应顶点变为原来的顶点
#         out[largest_component_mask] = torch.tensor(source_mesh.vertices, dtype=torch.float)[largest_component_mask]

#         new_mesh = Trimesh(out, source_mesh.faces)
#         if verbose:
#             print("")

#         return new_mesh

#     def find_nearest_vertices(target_vertices, source_vertices, k=1):
#         tree = cKDTree(source_vertices)
#         distances, indices = tree.query(target_vertices, k=k)
#         return indices, distances

#     # 读取mesh
#     fit_mesh = in_mesh_fit.copy()
#     faulty_mesh = in_mesh_faulty.copy()

#     # 1.阈值法识别fit_mesh中缺损部分（“修补”部分）
#     distance = distance_from_reference_mesh(torch.tensor(fit_mesh.vertices, dtype=torch.float).to(device),
#                                             torch.tensor(faulty_mesh.vertices, dtype=torch.float).to(device),
#                                             torch.tensor(faulty_mesh.faces).to(device)).cpu().numpy()
#     idx = np.where(distance > 4)[0]  # 阈值
#     color_mesh = get_color_mesh(fit_mesh, idx)
#     # color_mesh.export(join(out_path, "color_1.ply"))

#     # 2.计算最大联通分量（缺损部分）以及扩展部分
#     connected_components = extract_connected_components(fit_mesh, idx)
#     if len(connected_components) == 0:
#         return fit_mesh
#     largest_component = max(connected_components, key=len)
#     expanded_component = expand_connected_component(fit_mesh, largest_component, 2)
#     color_mesh = get_color_mesh(fit_mesh, list(largest_component))
#     color_mesh = get_color_mesh(color_mesh, list(expanded_component), False, [0, 255, 0, 255])
#     # color_mesh.export(join(out_path, 'color_2.ply'))

#     # 3.最优化法向位移场（投影）
#     vertex_mask = np.zeros(len(fit_mesh.vertices), dtype=bool)
#     vertex_mask[list(largest_component)] = True
#     projection_mesh = projection(fit_mesh, vertex_mask, faulty_mesh)
#     # projection_mesh.export(join(out_path, "projection.ply"))

#     # 4.将最大联通分量的顶点逐一进行位移。 位移量：扩展部分中K个最近顶点[投影]时位移的均值
#     vertices_expanded = fit_mesh.vertices[list(expanded_component)]
#     normal_displacement = (projection_mesh.vertices - fit_mesh.vertices)[list(expanded_component)]
#     indices, distances = find_nearest_vertices(fit_mesh.vertices, vertices_expanded, k=15)
#     completion_mesh = projection_mesh.copy()
#     for id in largest_component:
#         mean_displacement = np.mean(normal_displacement[indices[id]], axis=0)
#         completion_mesh.vertices[id] += mean_displacement

#     def laplacian_smoothing(iterations=2, smoothing_factor=0.5):
#         vertices_to_smooth = list(expanded_component)
#         vertices_to_smooth.extend(list(largest_component))

#         # 循环进行拉普拉斯平滑处理
#         for _ in range(iterations):
#             smoothed_vertices = []
#             for vertex_index in vertices_to_smooth:
#                 vertex = completion_mesh.vertices[vertex_index]
#                 neighbors = completion_mesh.vertex_neighbors[vertex_index]
#                 neighbor_vertices = completion_mesh.vertices[neighbors]
#                 smoothed_vertex = vertex + smoothing_factor * np.mean(neighbor_vertices - vertex, axis=0)
#                 smoothed_vertices.append(smoothed_vertex)
#             # 更新要平滑的顶点坐标
#             for i, vertex_index in enumerate(vertices_to_smooth):
#                 completion_mesh.vertices[vertex_index] = smoothed_vertices[i]

#     # 5.拉普拉斯平滑处理修复区域
#     if laplacian:
#         laplacian_smoothing()


#     if verbose:
#         print("refinement")
#     for i in range(5):
#         completion_mesh = special_point_refinement(completion_mesh)
#     # completion_mesh.export(join(out_path, 'refinement.ply'))

#     # done
#     print("done!")
#     return completion_mesh


# def facial_mesh_completion(in_file, out_file, config, generator, lambda_reg=None, verbose=True, rr=False,
#                             dis_percent=None):
#     dir = os.path.dirname(in_file)
#     device = generator.parameters().__next__().device

#     mesh_in = load_mesh(in_file)

#     if rr:
#         mesh_in, T, centroid = rigid_registration(mesh_in, config, verbose=verbose)

#     # save_ply_explicit(mesh_in, "rr.ply")

#     mesh_fit = fit(mesh_in, generator, config, device, lambda_reg=lambda_reg, verbose=verbose, loss_convergence=1e-7,
#                     dis_percent=dis_percent)
#     mesh_com = post_processing(mesh_fit, mesh_in, device, verbose=verbose)

#     if rr:
#         mesh_com.apply_transform(np.linalg.inv(T))
#         mesh_com.vertices += centroid

#     save_ply_explicit(mesh_com, out_file)
    
#     return mesh_com



