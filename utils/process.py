import argparse
import numpy as np
import pymeshlab as pm
import trimesh as trimesh
from trimesh import Trimesh, load_mesh
from trimesh.registration import icp
from trimesh.repair import fill_holes, fix_normals
from scipy.spatial import cKDTree, Delaunay
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def trimesh_to_pymeshlab(mesh):
    # Extract vertices and faces
    vertices = mesh.vertices
    faces = mesh.faces

    # Create a pymeshlab Mesh
    ms = pm.MeshSet()
    pm_mesh = pm.Mesh(vertices, faces)

    # Add the mesh to the MeshSet
    ms.add_mesh(pm_mesh, "converted_mesh")
    
    return ms

def pymeshlab_to_trimesh(ms):
    mesh = ms.current_mesh()
    
    # Extract vertex and face data
    vertices = np.array(mesh.vertex_matrix())
    faces = np.array(mesh.face_matrix())

    # Create a Trimesh object
    tm_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    return tm_mesh

def extract_meshes(mesh):
    mesh_parts = mesh.split(only_watertight=False)
    
    # Create new meshes with each CC part
    new_meshes = []
    for part in mesh_parts:
        new_mesh = trimesh.Trimesh(vertices=part.vertices, faces=part.faces)
        new_meshes.append(new_mesh)
    
    return new_meshes

def compute_centroid(mesh):
    # Supposons que mesh.vertices est un np.array de shape (N, 3)
    return np.mean(mesh.vertices, axis=0)

def reorder_mesh_parts(template_mesh_parts, mesh_parts, distance_threshold=np.inf):
    # Calcul des barycentres
    template_centroids = np.array([compute_centroid(mesh) for mesh in template_mesh_parts])
    mesh_centroids = np.array([compute_centroid(mesh) for mesh in mesh_parts])
    
    # Calcul des distances entre chaque barycentre de template et ceux de mesh_parts
    distances = cdist(template_centroids, mesh_centroids)
    
    # Réorganisation
    reordered = []
    used_indices = set()
    for i in range(len(template_mesh_parts)):
        # Trouve l'indice du mesh le plus proche non encore utilisé
        min_dist = np.inf
        min_index = None
        for j in range(len(mesh_parts)):
            if j not in used_indices and distances[i, j] < min_dist:
                min_dist = distances[i, j]
                min_index = j
        # Appliquer le seuil si nécessaire
        if min_index is not None and min_dist <= distance_threshold:
            reordered.append(mesh_parts[min_index])
            used_indices.add(min_index)
        else:
            reordered.append(None)  # Pas de correspondance proche

    return reordered

def fill_mesh(template_file, input_file, output_file):


    in_mesh = load_mesh(input_file)
    template_mesh = load_mesh(template_file)
    mesh = in_mesh

    T, _, _ = icp(mesh.vertices, template_mesh.vertices, max_iterations=50)
    mesh.apply_transform(T)#move the mesh towards the template
    #split the template into its CC parts

    template_mesh_parts= extract_meshes(template_mesh)
    mesh_parts = extract_meshes(mesh)
    mesh_parts=reorder_mesh_parts(template_mesh_parts,mesh_parts,distance_threshold=0.05)

    for i in range(len(template_mesh_parts)):
        print(template_mesh_parts[i].vertices.shape)
        if(mesh_parts[i] is not None):
            print(mesh_parts[i].vertices.shape)
        else:
            print("none")

    #process each mesh part : 
    for i in range(len(template_mesh_parts)):
        # Load vertex and face data
        V_template = template_mesh_parts[i].vertices          # (N_template, 3)
        F_template = template_mesh_parts[i].faces             # (M, 3)        
        if(i>len(mesh_parts)-1):
            #if a CC is missing duplicate it
            mesh_parts.append(trimesh.Trimesh(vertices=V_template, faces=F_template))
        else :
            if(mesh_parts[i] is None):
                mesh_parts[i]=template_mesh_parts[i]
            else:        
                #fill the missing part
                V_mesh = mesh_parts[i].vertices                  # (N_mesh, 3)
                F_mesh = mesh_parts[i].faces                       # (K, 3)

                # Compute distances and do one-to-one matching
                cost_matrix = cdist(V_mesh, V_template)
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                # Add unmatched vertices
                all_template_indices = np.arange(len(V_template))
                unmatched_template_indices = np.setdiff1d(all_template_indices, col_ind)

                V_unmatched = V_template[unmatched_template_indices]

                # Step 5: Concatenate vertices
                V_combined = np.vstack([V_mesh, V_unmatched])
                   
                new_mesh= trimesh.Trimesh(vertices=V_combined)

                # match the faces of the new mesh to the template
                #compute again a one-to-one matching between the faces of the new mesh and the template
                V_mesh = new_mesh.vertices                  # (N_mesh, 3)
                F_mesh = new_mesh.faces                       # (K, 3)
                # Compute distances and do one-to-one matching
                cost_matrix = cdist(V_template, V_mesh)
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                # Mapping: template_mesh[i] corresponds to new_mesh[col_ind[i]]
                new_faces = []
                for face in F_template:
                    new_faces.append([col_ind[v] for v in face])
                new_faces = np.array(new_faces)

                #renumber the vertices of the new mesh so as they are in the same order than in the template according to the one-to-one matching
                V_mesh_reordered = V_mesh[col_ind]

                #Change accordingly the faces' vertices' labels
                mapping = np.zeros_like(col_ind)
                mapping[col_ind] = np.arange(len(col_ind))
                faces_updated = mapping[new_faces]

                mesh_parts[i]= trimesh.Trimesh(vertices=V_mesh_reordered, faces=faces_updated)

    new_mesh = trimesh.util.concatenate(mesh_parts) 
    new_mesh.apply_transform(np.linalg.inv(T))
    new_mesh.export(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-template", type=str, required=True)
    parser.add_argument("-input", type=str, required=True)
    parser.add_argument("-output", type=str, required=True)
    args = parser.parse_args()
    template_file=args.template
    input_file=args.input
    output_file=args.output
    fill_mesh(template_file, input_file, output_file)


