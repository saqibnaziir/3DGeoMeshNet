import os
import numpy as np
from plyfile import PlyData
import torch
import math

def read_ply(file_path):
    """Reads a .ply file and returns the vertex data as a numpy array."""
    ply_data = PlyData.read(file_path)
    vertices = np.array([list(vertex) for vertex in ply_data['vertex']])
    return vertices

def normalize_mesh(vertices, mean, std):
    """Normalizes the mesh vertices using the provided mean and std."""
    # Convert PyTorch tensors to NumPy arrays
    mean = mean.numpy()
    std = std.numpy()
    return (vertices - mean) / std

def calculate_mse(gt_vertices, noisy_vertices):
    """Calculates the Mean Squared Error (MSE) between two vertex arrays."""
    return np.mean((gt_vertices - noisy_vertices) ** 2)

def calculate_psnr(gt_vertices, noisy_vertices):
    """Calculates the PSNR between GT and noisy vertices."""
    mse = calculate_mse(gt_vertices, noisy_vertices)
    if mse == 0:
        return float('inf')  # Perfect match
    max_i = np.max(gt_vertices)
    psnr = 20 * math.log10(max_i / math.sqrt(mse))
    return psnr

def process_meshes_with_normalization(gt_folder, noisy_folder, norm_file):
    """Processes meshes with normalization and calculates PSNR."""
    # Load normalization parameters
    norm_data = torch.load(norm_file)
    mean = norm_data['mean']  # Mean used for normalization
    std = norm_data['std']    # Std dev used for normalization

    gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith('.ply')])
    noisy_files = sorted([f for f in os.listdir(noisy_folder) if f.endswith('.ply')])

    if len(gt_files) != len(noisy_files):
        print("Mismatch in number of meshes between GT and Noisy folders.")
        return

    for gt_file, noisy_file in zip(gt_files, noisy_files):
        gt_path = os.path.join(gt_folder, gt_file)
        noisy_path = os.path.join(noisy_folder, noisy_file)
        
        gt_vertices = read_ply(gt_path)
        noisy_vertices = read_ply(noisy_path)
        
        if gt_vertices.shape != noisy_vertices.shape:
            print(f"Shape mismatch for {gt_file} and {noisy_file}. Skipping...")
            continue

        # Normalize meshes
        gt_vertices_normalized = normalize_mesh(gt_vertices, mean, std)
        noisy_vertices_normalized = normalize_mesh(noisy_vertices, mean, std)

        # Calculate PSNR
        psnr = calculate_psnr(gt_vertices_normalized, noisy_vertices_normalized)
        print(f"PSNR for {gt_file}: {psnr:.2f} dB")

# Paths to the folders
gt_folder = '/home/nazir241/PosDoc/COSURAI/Codes/FaceCom/Denoising/gt'
noisy_folder = '/home/nazir241/PosDoc/COSURAI/Codes/FaceCom/Denoising/pred'
norm_file = '/home/nazir241/PosDoc/COSURAI/Codes/FaceCom/Denoising/norm.pt'

process_meshes_with_normalization(gt_folder, noisy_folder, norm_file)

