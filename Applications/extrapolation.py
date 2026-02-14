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

def interpolate_or_extrapolate_latent_space(generator, mesh1, mesh2, config, device, steps=10, verbose=True, mode='interpolation'):
    """
    Interpolate or extrapolate between two 3D meshes in latent space and decode to generate intermediate/extended meshes.
    
    Args:
        generator: The neural network model with encoder and decoder methods.
        mesh1: First input mesh (Trimesh object)
        mesh2: Second input mesh (Trimesh object)
        config: Configuration dictionary
        device: Torch device to run computations on
        steps: Number of interpolation/extrapolation steps 
        verbose: Whether to print progress
        mode: 'interpolation' or 'extrapolation'
    
    Returns:
        Tuple containing:
        - List of interpolated or extrapolated Trimesh objects
        - Neutral/First mesh Trimesh object
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

    # Prepare input for the encoder
    data1 = Data(x=normalized_vertices1)
    data2 = Data(x=normalized_vertices2)
    
    # Create a batch
    batch = Batch.from_data_list([data1, data2])
    batch = batch.to(device)

    # Encode meshes to latent space
    z = generator.encoder(batch.x, batch.num_graphs)

    # Split the latent vectors
    z1, z2 = z.chunk(2)

    # Compute the direction vector in latent space
    z_direction = z2 - z1

    # Create interpolation/extrapolation steps
    if mode == 'extrapolation':
        # For extrapolation, we extend beyond the original range
        # Negative values will be before mesh1, positive beyond mesh2
        alphas = torch.linspace(-1, 2, steps).to(device)
    else:  # interpolation
        # Standard interpolation between 0 and 1
        alphas = torch.linspace(0, 1, steps).to(device)

    processed_meshes = []

    # Create neutral mesh (first input mesh)
    decoded_neutral = generator.decoder(z1, 1)
    decoded_neutral = decoded_neutral * std + mean
    neutral_vertices = decoded_neutral.detach().cpu().numpy()
    neutral_mesh = Trimesh(neutral_vertices, mesh1.faces)

    # Interpolate/Extrapolate and decode
    for i, alpha in enumerate(alphas):
        if verbose:
            print(f"{'Extrapolating' if mode == 'extrapolation' else 'Interpolating'} step {i+1}/{steps}")
        
        # Linear interpolation/extrapolation in latent space
        z_processed = z1 + alpha * z_direction

        # Decode processed latent vector
        decoded_vertices = generator.decoder(z_processed, 1)

        # Denormalize vertices
        decoded_vertices = decoded_vertices * std + mean

        # Convert to numpy and create Trimesh
        processed_vertices = decoded_vertices.detach().cpu().numpy()
        processed_mesh = Trimesh(processed_vertices, mesh1.faces)
        processed_meshes.append(processed_mesh)

    return processed_meshes, neutral_mesh

def mesh_interpolation_or_extrapolation(interpolation_folder, config_path, output_folder, mode='interpolation', steps=10, verbose=True):
    """
    Perform mesh interpolation or extrapolation for all pairs of meshes in the input folder.
    
    Args:
        interpolation_folder: Path to folder containing input meshes
        config_path: Path to configuration file
        output_folder: Path to save processed meshes
        mode: 'interpolation' or 'extrapolation'
        steps: Number of interpolation/extrapolation steps
        verbose: Whether to print progress
    """
    # Validate mode
    if mode not in ['interpolation', 'extrapolation']:
        raise ValueError("Mode must be either 'interpolation' or 'extrapolation'")

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

    # Process meshes
    for i in range(0, len(mesh_files) - 1, 2):
        mesh1_path = os.path.join(interpolation_folder, mesh_files[i])
        mesh2_path = os.path.join(interpolation_folder, mesh_files[i+1])

        # Load meshes
        mesh1 = load_mesh(mesh1_path)
        mesh2 = load_mesh(mesh2_path)

        # Generate processed meshes and neutral mesh
        processed_meshes, neutral_mesh = interpolate_or_extrapolate_latent_space(
            generator, mesh1, mesh2, config, device, 
            steps=steps, verbose=verbose, mode=mode
        )

        # Save neutral mesh
        neutral_output_path = os.path.join(
            output_folder, 
            f"neutral_{mesh_files[i][:-4]}.ply"
        )
        save_ply_explicit(neutral_mesh, neutral_output_path)
        if verbose:
            print(f"Saved neutral mesh: {neutral_output_path}")

        # Save processed meshes
        base_name = f"{mode}_{mesh_files[i][:-4]}_{mesh_files[i+1][:-4]}"
        for j, processed_mesh in enumerate(processed_meshes):
            output_path = os.path.join(
                output_folder, 
                f"{base_name}_step_{j:03d}.ply"
            )
            save_ply_explicit(processed_mesh, output_path)
            if verbose:
                print(f"Saved {mode} mesh: {output_path}")

    print(f"Mesh {mode} completed.")

# Example usage
if __name__ == "__main__":
    # Paths to be configured
    INPUT_FOLDER = r"D:\Saqib\COSURAI\FaceCom\INTERPOLATION\input"  # Folder containing input meshes
    CONFIG_PATH = r"D:\Saqib\COSURAI\FaceCom\config\config.cfg"  # Path to your configuration file
    
    # Interpolation output folder
    INTERPOLATION_OUTPUT_FOLDER = r"D:\Saqib\COSURAI\FaceCom\INTERPOLATION\output"
    
    # Extrapolation output folder
    EXTRAPOLATION_OUTPUT_FOLDER = r"D:\Saqib\COSURAI\FaceCom\EXTRAPOLATION\output"

    # Perform interpolation
    mesh_interpolation_or_extrapolation(
        interpolation_folder=INPUT_FOLDER,
        config_path=CONFIG_PATH,
        output_folder=INTERPOLATION_OUTPUT_FOLDER,
        mode='interpolation',
        steps=10,
        verbose=True
    )

    # Perform extrapolation
    mesh_interpolation_or_extrapolation(
        interpolation_folder=INPUT_FOLDER,
        config_path=CONFIG_PATH,
        output_folder=EXTRAPOLATION_OUTPUT_FOLDER,
        mode='extrapolation',
        steps=10,
        verbose=True
    )



# import torch
# import os
# import numpy as np
# from trimesh import Trimesh
# import torchvision.utils

# from config.config import read_config
# from trimesh import load_mesh
# import torch.nn.functional as F
# import torch
# from torch_geometric.data import Data, Batch

# from utils.funcs import load_generator, load_norm, save_ply_explicit

# def interpolate_latent_space(generator, mesh1, mesh2, config, device, steps=10, verbose=True, extrapolation=False):
#     """
#     Interpolate or extrapolate between two 3D meshes in latent space and decode to generate intermediate/extended meshes.
    
#     Args:
#         generator: The neural network model with encoder and decoder methods.
#         mesh1: First input mesh (Trimesh object)
#         mesh2: Second input mesh (Trimesh object)
#         config: Configuration dictionary
#         device: Torch device to run computations on
#         steps: Number of interpolation/extrapolation steps 
#         verbose: Whether to print progress
#         extrapolation: Whether to perform extrapolation instead of interpolation
    
#     Returns:
#         Tuple containing:
#         - List of interpolated or extrapolated Trimesh objects
#         - Neutral/First mesh Trimesh object
#     """
#     # Load normalization parameters
#     mean, std = load_norm(config)
#     mean = mean.to(device)
#     std = std.to(device)

#     # Convert mesh vertices to normalized torch tensors
#     vertices1 = torch.tensor(mesh1.vertices, dtype=torch.float).to(device)
#     vertices2 = torch.tensor(mesh2.vertices, dtype=torch.float).to(device)

#     # Normalize input vertices
#     normalized_vertices1 = (vertices1 - mean) / std
#     normalized_vertices2 = (vertices2 - mean) / std

#     # Prepare input for the encoder
#     data1 = Data(x=normalized_vertices1)
#     data2 = Data(x=normalized_vertices2)
    
#     # Create a batch
#     batch = Batch.from_data_list([data1, data2])
#     batch = batch.to(device)

#     # Encode meshes to latent space
#     z = generator.encoder(batch.x, batch.num_graphs)

#     # Split the latent vectors
#     z1, z2 = z.chunk(2)

#     # Compute the direction vector in latent space
#     z_direction = z2 - z1

#     # Create extrapolation/interpolation steps
#     if extrapolation:
#         # For extrapolation, we extend beyond the original range
#         # Negative values will be before mesh1, positive beyond mesh2
#         alphas = torch.linspace(-1, 2, steps).to(device)
#     else:
#         # Standard interpolation between 0 and 1
#         alphas = torch.linspace(0, 1, steps).to(device)

#     extrapolated_meshes = []

#     # Create neutral mesh (first input mesh)
#     decoded_neutral = generator.decoder(z1, 1)
#     decoded_neutral = decoded_neutral * std + mean
#     neutral_vertices = decoded_neutral.detach().cpu().numpy()
#     neutral_mesh = Trimesh(neutral_vertices, mesh1.faces)

#     # Extrapolate and decode
#     for i, alpha in enumerate(alphas):
#         if verbose:
#             print(f"{'Extrapolating' if extrapolation else 'Interpolating'} step {i+1}/{steps}")
        
#         # Linear extrapolation/interpolation in latent space
#         z_ext = z1 + alpha * z_direction

#         # Decode extrapolated latent vector
#         decoded_vertices = generator.decoder(z_ext, 1)

#         # Denormalize vertices
#         decoded_vertices = decoded_vertices * std + mean

#         # Convert to numpy and create Trimesh
#         ext_vertices = decoded_vertices.detach().cpu().numpy()
#         ext_mesh = Trimesh(ext_vertices, mesh1.faces)
#         extrapolated_meshes.append(ext_mesh)

#     return extrapolated_meshes, neutral_mesh

# def mesh_extrapolation(interpolation_folder, config_path, output_folder, steps=10, verbose=True):
#     """
#     Perform mesh extrapolation for all pairs of meshes in the interpolation folder.
    
#     Args:
#         interpolation_folder: Path to folder containing input meshes
#         config_path: Path to configuration file
#         output_folder: Path to save extrapolated meshes
#         steps: Number of extrapolation steps
#         verbose: Whether to print progress
#     """
#     # Read configuration
#     config = read_config(config_path)

#     # Prepare device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load generator model
#     generator = load_generator(config)
#     generator = generator.to(device)
#     generator.eval()

#     # Ensure output folder exists
#     os.makedirs(output_folder, exist_ok=True)

#     # Get all mesh files in the interpolation folder
#     mesh_files = [f for f in os.listdir(interpolation_folder) if f.endswith(('.ply', '.obj'))]

#     # Sort mesh files to ensure deterministic pairing
#     mesh_files.sort()

#     # Extrapolate between consecutive pairs of meshes
#     for i in range(0, len(mesh_files) - 1, 2):
#         mesh1_path = os.path.join(interpolation_folder, mesh_files[i])
#         mesh2_path = os.path.join(interpolation_folder, mesh_files[i+1])

#         # Load meshes
#         mesh1 = load_mesh(mesh1_path)
#         mesh2 = load_mesh(mesh2_path)

#         # Generate extrapolated meshes and neutral mesh
#         extrapolated_meshes, neutral_mesh = interpolate_latent_space(
#             generator, mesh1, mesh2, config, device, 
#             steps=steps, verbose=verbose, extrapolation=True
#         )

#         # Save neutral mesh
#         neutral_output_path = os.path.join(
#             output_folder, 
#             f"neutral_{mesh_files[i][:-4]}.ply"
#         )
#         save_ply_explicit(neutral_mesh, neutral_output_path)
#         if verbose:
#             print(f"Saved neutral mesh: {neutral_output_path}")

#         # Save extrapolated meshes
#         base_name = f"extrapolation_{mesh_files[i][:-4]}_{mesh_files[i+1][:-4]}"
#         for j, ext_mesh in enumerate(extrapolated_meshes):
#             output_path = os.path.join(
#                 output_folder, 
#                 f"{base_name}_step_{j:03d}.ply"
#             )
#             save_ply_explicit(ext_mesh, output_path)
#             if verbose:
#                 print(f"Saved extrapolated mesh: {output_path}")

#     print("Mesh extrapolation completed.")

# # Example usage
# if __name__ == "__main__":
#     # Paths to be configured
#     INTERPOLATION_FOLDER = r"D:\Saqib\COSURAI\FaceCom\INTERPOLATION\input"  # Folder containing input meshes
#     CONFIG_PATH = r"D:\Saqib\COSURAI\FaceCom\config\config.cfg"  # Path to your configuration file
#     EXTRAPOLATION_OUTPUT_FOLDER = r"D:\Saqib\COSURAI\FaceCom\EXTRAPOLATION\output"  # Folder to save extrapolated meshes

#     # Perform mesh extrapolation
#     mesh_extrapolation(
#         interpolation_folder=INTERPOLATION_FOLDER,
#         config_path=CONFIG_PATH,
#         output_folder=EXTRAPOLATION_OUTPUT_FOLDER,
#         steps=10,
#         verbose=True
#     )