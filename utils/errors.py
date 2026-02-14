import torch
import numpy as np
import trimesh
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss.point_mesh_distance import point_face_distance
from torch import Tensor

def load_mesh(file_path):
    try:
        mesh = trimesh.load(file_path)
        verts = torch.tensor(mesh.vertices, dtype=torch.float32)
        faces = torch.tensor(mesh.faces, dtype=torch.int64)
        return verts, faces
    except Exception as e:
        print(f"Error loading mesh from {file_path}: {str(e)}")
        return None, None

def point_mesh_face_distance_single_direction(
        meshes: Meshes,
        pcls: Pointclouds,
        min_triangle_area: float = 1e-6,
):
    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area
    )

    return point_to_face

def distance_from_reference_mesh(points: Tensor, mesh_vertices: Tensor, mesh_faces: Tensor):
    """
        return distance^2 from mesh for every point in points
    """
    meshes = Meshes([mesh_vertices], [mesh_faces])
    pcs = Pointclouds([points])
    distances = point_mesh_face_distance_single_direction(meshes, pcs)
    return distances

def smoothness_loss(vertices: Tensor, faces: Tensor):
    meshes = Meshes([vertices], [faces])
    return mesh_laplacian_smoothing(meshes)

def compute_chamfer_distance(pred_points, gt_points):
    loss, _ = chamfer_distance(pred_points, gt_points)
    return loss

def compute_mean_point_to_surface_distance(pred_points, gt_vertices, gt_faces):
    distances = distance_from_reference_mesh(pred_points, gt_vertices, gt_faces)
    return distances.mean()

def evaluate_mesh(pred_mesh_path, gt_mesh_path, num_samples=10000, device='cuda'):
    # Load meshes
    pred_verts, pred_faces = load_mesh(pred_mesh_path)
    gt_verts, gt_faces = load_mesh(gt_mesh_path)

    if pred_verts is None or gt_verts is None:
        return None, None, None

    # Move to specified device
    pred_verts = pred_verts.to(device)
    pred_faces = pred_faces.to(device)
    gt_verts = gt_verts.to(device)
    gt_faces = gt_faces.to(device)

    # Create Meshes objects
    pred_mesh = Meshes(verts=[pred_verts], faces=[pred_faces])
    gt_mesh = Meshes(verts=[gt_verts], faces=[gt_faces])

    # Sample points from the meshes
    pred_points = sample_points_from_meshes(pred_mesh, num_samples)
    gt_points = sample_points_from_meshes(gt_mesh, num_samples)

    # Compute Chamfer distance
    chamfer_dist = compute_chamfer_distance(pred_points, gt_points)

    # Compute mean point-to-surface distance
    mean_p2s_dist = compute_mean_point_to_surface_distance(pred_points, gt_verts, gt_faces)

    # Compute smoothness loss
    smoothness = mesh_laplacian_smoothing(pred_mesh)

    return chamfer_dist.item(), mean_p2s_dist.item(), smoothness.item()

def main():
    # List of your reconstructed mesh files and corresponding ground truth files
    mesh_pairs = [
        ("D:\Saqib\COSURAI\FaceCom\1.ply", "D:\Saqib\COSURAI\FaceCom\recon1.ply"),
        ("D:\Saqib\COSURAI\FaceCom\2.ply", "D:\Saqib\COSURAI\FaceCom\recon2.ply"),
        # Add more pairs as needed
    ]

    results = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for pred_mesh_path, gt_mesh_path in mesh_pairs:
        chamfer_dist, mean_p2s_dist, smoothness = evaluate_mesh(pred_mesh_path, gt_mesh_path, device=device)
        if chamfer_dist is not None:
            results.append({
                "pred_mesh": pred_mesh_path,
                "gt_mesh": gt_mesh_path,
                "chamfer_distance": chamfer_dist,
                "mean_point_to_surface_distance": mean_p2s_dist,
                "smoothness": smoothness
            })
        else:
            print(f"Skipping evaluation for {pred_mesh_path} and {gt_mesh_path} due to loading error.")

    # Print or save results
    for result in results:
        print(f"Results for {result['pred_mesh']}:")
        print(f"  Chamfer Distance: {result['chamfer_distance']:.6f}")
        print(f"  Mean Point-to-Surface Distance: {result['mean_point_to_surface_distance']:.6f}")
        print(f"  Smoothness: {result['smoothness']:.6f}")
        print()

if __name__ == "__main__":
    main()