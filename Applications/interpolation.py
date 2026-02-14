import torch
import os
import numpy as np
from trimesh import Trimesh
import torchvision.utils

from config.config import read_config
from trimesh import load_mesh
import torch.nn.functional as F
import torch
from torch_geometric.data import Data, Batch

from utils.funcs import load_generator, load_norm, save_ply_explicit

def interpolate_latent_space(generator, mesh1, mesh2, config, device, steps=10, verbose=True):
    """
    Interpolate between two 3D meshes in latent space and decode to generate intermediate meshes.
    
    Args:
        generator: The neural network model with encoder and decoder methods.
        mesh1: First input mesh (Trimesh object)
        mesh2: Second input mesh (Trimesh object)
        config: Configuration dictionary
        device: Torch device to run computations on
        steps: Number of interpolation steps between the two meshes
        verbose: Whether to print progress
    
    Returns:
        List of interpolated Trimesh objects
    """
    # Load normalization parameters
    mean, std = load_norm(config)
    mean = mean.to(device)
    std = std.to(device)

    # Convert mesh vertices to normalized torch tensors
    vertices1 = torch.tensor(mesh1.vertices, dtype=torch.float).to(device)
    vertices2 = torch.tensor(mesh2.vertices, dtype=torch.float).to(device)

    # Normalize input vertices
    normalized_vertices1 = (vertices1 - mean) / std
    normalized_vertices2 = (vertices2 - mean) / std

    # Prepare input for the encoder (this is the critical part)
    from torch_geometric.data import Data, Batch

    # Create PyTorch Geometric Data objects
    data1 = Data(x=normalized_vertices1)
    data2 = Data(x=normalized_vertices2)
    
    # Create a batch
    batch = Batch.from_data_list([data1, data2])
    batch = batch.to(device)

    # Encode meshes to latent space (note the batch argument)
    z = generator.encoder(batch.x, batch.num_graphs)

    # Split the latent vectors
    z1, z2 = z.chunk(2)

    # Create interpolation steps
    alphas = torch.linspace(0, 1, steps).to(device)
    interpolated_meshes = []

    # Interpolate and decode
    for i, alpha in enumerate(alphas):
        if verbose:
            print(f"Interpolating step {i+1}/{steps}")
        
        # Linear interpolation in latent space
        z_interp = alpha * z2 + (1 - alpha) * z1

        # Decode interpolated latent vector
        decoded_vertices = generator.decoder(z_interp, 1)

        # Denormalize vertices
        decoded_vertices = decoded_vertices * std + mean

        # Convert to numpy and create Trimesh
        interp_vertices = decoded_vertices.detach().cpu().numpy()
        interp_mesh = Trimesh(interp_vertices, mesh1.faces)
        interpolated_meshes.append(interp_mesh)

    return interpolated_meshes

def mesh_interpolation(interpolation_folder, config_path, output_folder, steps=10, verbose=True):
    """
    Perform mesh interpolation for all pairs of meshes in the interpolation folder.
    
    Args:
        interpolation_folder: Path to folder containing input meshes
        config_path: Path to configuration file
        output_folder: Path to save interpolated meshes
        steps: Number of interpolation steps
        verbose: Whether to print progress
    """
    # Read configuration
    config = read_config(config_path)

    # Prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load generator model
    generator = load_generator(config)
    generator = generator.to(device)
    generator.eval()

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all mesh files in the interpolation folder
    mesh_files = [f for f in os.listdir(interpolation_folder) if f.endswith(('.ply', '.obj'))]

    # Sort mesh files to ensure deterministic pairing
    mesh_files.sort()

    # Interpolate between consecutive pairs of meshes
    for i in range(0, len(mesh_files) - 1, 2):
        mesh1_path = os.path.join(interpolation_folder, mesh_files[i])
        mesh2_path = os.path.join(interpolation_folder, mesh_files[i+1])

        # Load meshes
        mesh1 = load_mesh(mesh1_path)
        mesh2 = load_mesh(mesh2_path)

        # Generate interpolated meshes
        interpolated_meshes = interpolate_latent_space(
            generator, mesh1, mesh2, config, device, steps=steps, verbose=verbose
        )

        # Save interpolated meshes
        base_name = f"interpolation_{mesh_files[i][:-4]}_{mesh_files[i+1][:-4]}"
        for j, interp_mesh in enumerate(interpolated_meshes):
            output_path = os.path.join(
                output_folder, 
                f"{base_name}_step_{j:03d}.ply"
            )
            save_ply_explicit(interp_mesh, output_path)
            if verbose:
                print(f"Saved interpolated mesh: {output_path}")

    print("Mesh interpolation completed.")

# Example usage
if __name__ == "__main__":
    # Paths to be configured
    INTERPOLATION_FOLDER = r"D:\Saqib\COSURAI\FaceCom\INTERPOLATION\input"  # Folder containing input meshes
    CONFIG_PATH = "D:\Saqib\COSURAI\FaceCom\config\config.cfg"  # Path to your configuration file
    OUTPUT_FOLDER = r"D:\Saqib\COSURAI\FaceCom\INTERPOLATION\output"  # Folder to save interpolated meshes

    # Perform mesh interpolation
    mesh_interpolation(
        interpolation_folder=INTERPOLATION_FOLDER,
        config_path=CONFIG_PATH,
        output_folder=OUTPUT_FOLDER,
        steps=10,
        verbose=True
    )