import torch
import trimesh
from trimesh import Trimesh, load_mesh
import os
from os.path import join
import sys
import numpy as np
import warnings

# Import your model
# from models import FMGenModel  # Adjust import path as needed
# from config.config import read_config
# from utils.funcs import load_generator, load_norm, save_ply_explicit


from tqdm import tqdm

from queue import Queue
import math
from pytorch3d.structures import Meshes
# from pytorch3d.loss import chamfer_distance
# from .pytorch3d_extend import distance_from_reference_mesh, smoothness_loss
from trimesh.registration import icp
from scipy.spatial import cKDTree
import sys
from PIL import Image
import torchvision.transforms as transforms
# from .render import render_d
from .funcs import load_generator, spherical_regularization_loss, save_ply_explicit, get_random_z, load_norm
# from .funcs import load_model, spherical_regularization_loss, save_ply_explicit, get_random_z, load_norm


warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
sys.setrecursionlimit(30000)

def facial_mesh_completion(in_file, out_file, config, generator, lambda_reg=None, 
                           verbose=True, rr=False, dis_percent=None):
    """
    Complete a partial facial mesh using the trained generator
    
    Args:
    - in_file: Path to input partial mesh
    - out_file: Path to save completed mesh
    - config: Configuration dictionary
    - generator: Trained FMGenModel
    - lambda_reg: Regularization parameter
    - verbose: Print progress details
    - rr: Perform rigid registration
    - dis_percent: Percentage of vertices to consider for distance computation
    """
    device = next(generator.parameters()).device
    
    # Load input mesh
    mesh_in = load_mesh(in_file)
    
    # Optional rigid registration
    if rr:
        mesh_in, T, centroid = rigid_registration(mesh_in, config, verbose=verbose)
    
    # Fit the generator to the partial mesh
    mesh_fit = fit(mesh_in, generator, config, device, 
                   lambda_reg=lambda_reg, 
                   verbose=verbose, 
                   loss_convergence=1e-7,
                   dis_percent=dis_percent)
    
    # Post-process the fitted mesh
    mesh_com = post_processing(mesh_fit, mesh_in, device, verbose=verbose)
    
    # Restore original registration if needed
    if rr:
        mesh_com.apply_transform(np.linalg.inv(T))
        mesh_com.vertices += centroid
    
    # Save completed mesh
    save_ply_explicit(mesh_com, out_file)

def rigid_registration(in_mesh, config, verbose=True):
    """Perform rigid registration of mesh"""
    if verbose:
        print("Rigid registration...")
    
    mesh = in_mesh
    template_mesh = load_mesh(config["template_file"])
    
    centroid = mesh.centroid
    mesh.vertices -= mesh.centroid
    
    from trimesh.registration import icp
    T, _, _ = icp(mesh.vertices, template_mesh.vertices, max_iterations=50)
    mesh.apply_transform(T)
    
    return mesh, T, centroid

def fit(in_mesh, generator, config, device, max_iters=1000, 
        loss_convergence=1e-6, lambda_reg=None, verbose=True, dis_percent=None):
    """
    Fit generator to partial mesh
    
    Args similar to facial_mesh_completion
    Returns fitted mesh
    """
    if verbose:
        print("\rFitting...")
    
    mesh = in_mesh.copy()
    template_mesh = load_mesh(config["template_file"])
    
    generator.eval()
    target_pc = torch.tensor(mesh.vertices, dtype=torch.float).to(device)
    
    # Initialize latent vector z
    z = torch.randn(1, generator.z_length, requires_grad=True, device=device)
    
    # Load normalization parameters
    mean, std = load_norm(config)
    mean = mean.to(device)
    std = std.to(device)
    faces = torch.tensor(template_mesh.faces).to(device)
    
    # Adam optimizer with learning rate scheduler
    optimizer = torch.optim.Adam([z], lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    
    lambda_reg = lambda_reg or config['lambda_reg']
    last_loss = float('inf')
    
    for i in range(max_iters):
        optimizer.zero_grad()
        
        # Forward pass through generator
        out = generator(z, 1)  # Adjusted for your model's signature
        out = out * std + mean
        
        # Compute losses
        loss_reg = torch.mean(z**2)  # Simple regularization
        
        # Compute distance from target mesh
        distance = torch.sqrt(distance_from_reference_mesh(target_pc, out, faces))
        
        if dis_percent:
            distance, _ = torch.sort(distance)
            distance = distance[:int(dis_percent * len(distance))]
        
        loss_dfrm = torch.mean(distance)
        loss = loss_dfrm + lambda_reg * loss_reg
        
        if verbose:
            print(f"\rFitting... Iter {i+1}, loss_recon: {loss_dfrm.item():.6f}, loss_reg: {loss_reg.item():.6f}", end='')
        
        if abs(last_loss - loss.item()) < loss_convergence:
            break
        
        last_loss = loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    out = generator(z, 1)
    out = out * std + mean
    fit_mesh = Trimesh(out.detach().cpu().numpy(), template_mesh.faces)
    
    return fit_mesh

def post_processing(in_mesh_fit, in_mesh_faulty, device, laplacian=True, verbose=True):
    """
    Post-process the fitted mesh to improve completion
    
    Note: This is a simplified version. You might need to adapt 
    specific implementation details based on your exact requirements.
    """
    fit_mesh = in_mesh_fit.copy()
    faulty_mesh = in_mesh_faulty.copy()
    
    # Placeholder for more advanced post-processing
    # Consider adding more sophisticated mesh refinement techniques
    
    # Simple smoothing
    if laplacian:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import laplacian
        
        vertices = fit_mesh.vertices
        edges = fit_mesh.edges
        
        # Create sparse adjacency matrix
        row_indices = edges[:, 0]
        col_indices = edges[:, 1]
        data = np.ones(len(edges))
        adjacency = csr_matrix((data, (row_indices, col_indices)), 
                                shape=(len(vertices), len(vertices)))
        
        # Perform smoothing
        smoothed_vertices = vertices.copy()
        for _ in range(5):  # Smoothing iterations
            lap = laplacian(adjacency, normed=True)
            smoothed_vertices += 0.1 * lap.dot(smoothed_vertices)
        
        fit_mesh.vertices = smoothed_vertices
    
    return fit_mesh

def distance_from_reference_mesh(source_points, target_points, target_faces):
    """
    Compute distance between point clouds
    
    Placeholder implementation - replace with your preferred distance metric
    """
    from pytorch3d.loss import chamfer_distance
    
    distance, _ = chamfer_distance(source_points.unsqueeze(0), 
                                   target_points.unsqueeze(0), 
                                   target_faces)
    return distance

def main():
    # Example usage
    config = read_config('D:\Saqib\COSURAI\FaceCom\config\config.cfg')
    generator = load_generator(config)  # Load your trained generator
    
    input_mesh = 'D:\Saqib\COSURAI\FaceCom\partial.ply'
    output_mesh = 'D:\Saqib\COSURAI\FaceCom\compartial.ply'
    
    facial_mesh_completion(input_mesh, output_mesh, config, generator)

if __name__ == "__main__":
    main()