import argparse
import numpy as np
import pymeshlab as pm
import trimesh as trimesh
from trimesh import Trimesh, load_mesh
from trimesh.registration import icp
from trimesh.repair import fill_holes, fix_normals
from scipy.spatial import KDTree

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

def intersection(mesh1,mesh2,save=False,name=""):
    # get the set of unique faces for each mesh
    faces1_set = set(map(tuple, mesh1.faces))
    faces2_set = set(map(tuple, mesh2.faces))

    unique_faces2 = np.array(list(faces2_set - faces1_set))  # Faces in mesh2 but not in mesh1

    mesh2_unique = Trimesh(vertices=mesh2.vertices, faces=unique_faces2)

    return mesh2_unique

# Function to find boundary vertices
def find_boundary_vertices(mesh):
    edges_sorted = np.sort(mesh.edges, axis=1)
    unique_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]  # Edges appearing once
    boundary_vertices = np.unique(boundary_edges)
    return boundary_vertices

def fill_mesh(template_file, input_file, output_file):


    in_mesh = load_mesh(input_file)
    template_mesh = load_mesh(template_file)
    mesh = in_mesh
    mesh.vertices -= mesh.centroid
    template_mesh.vertices -= template_mesh.centroid
    template_mesh.vertices -= template_mesh.centroid


    T, _, _ = icp(mesh.vertices, template_mesh.vertices, max_iterations=50)
    mesh.apply_transform(T)#move the mesh towards the template
    template_mesh.apply_transform(np.linalg.inv(T))#move the template mesh towards the input


    # STEP 1 : Fill the holes by adding faces (no new vertices)
    ms=trimesh_to_pymeshlab(mesh)
    ms.meshing_close_holes() #adds faces to close holes

    #STEP 2 : detect the created part that filled the holes
    mesh1 = mesh 
    mesh2 = pymeshlab_to_trimesh(ms)
    holes=intersection(mesh1,mesh2)


    # STEP 3 : Remove in the template the faces that correspond to the holes
    #split the template into its CC parts
    template_mesh_parts = template_mesh.split(only_watertight=False)
    #process each CC part
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    for m in template_mesh_parts:
        vertices_comp = m.vertices
        faces_comp = m.faces

        # Build KDTree for nearest neighbor search
        tree = KDTree(vertices_comp)
        distances, indices = tree.query(holes.vertices)

        if np.max(distances) > 0.1:
            # No removal needed
            new_vertices = vertices_comp
            new_faces = faces_comp
        else:
            # Find indices of vertices to remove
            indices_to_remove = set(indices)

            # Create mask to retain only the vertices that are NOT removed
            mask = np.ones(len(vertices_comp), dtype=bool)
            mask[list(indices_to_remove)] = False
            new_vertices = vertices_comp[mask]

            # Update the indices of remaining vertices
            old_to_new = -np.ones(len(vertices_comp), dtype=int)
            old_to_new[mask] = np.arange(len(new_vertices))


            # Separate retained and removed faces
            new_faces = []
            for face in faces_comp:
                if all(old_to_new[v] != -1 for v in face):  # Face remains
                    new_faces.append([old_to_new[v] for v in face])

            new_faces = np.array(new_faces)

            # If the new mesh is empty, skip adding it
            if len(new_faces) > 0:
                new_mesh = Trimesh(vertices=new_vertices, faces=new_faces)
                new_mesh_parts = new_mesh.split(only_watertight=False)
                if len(new_mesh_parts) > 0:
                    new_mesh = max(new_mesh_parts, key=lambda m: len(m.faces))
                    new_vertices = new_mesh.vertices
                    new_faces = new_mesh.faces
            else:
                new_vertices = np.array([])
                new_faces = np.array([])

        # Append retained vertices and faces
        if len(new_vertices) > 0:
            all_vertices.extend(new_vertices.tolist())
            all_faces.extend((np.array(new_faces) + vertex_offset).tolist())
            vertex_offset += len(new_vertices)
        

    # Convert lists to numpy arrays
    all_vertices = np.array(all_vertices)
    all_faces = np.array(all_faces)

    # Create the template mesh with holes as in the defect input
    template_mesh_with_holes = Trimesh(vertices=all_vertices, faces=all_faces)


    #STEP 4 : Extract the part of the template that corresponds to the created holes
    vertices_template = template_mesh.vertices
    vertices_holes = template_mesh_with_holes.vertices
    faces_template = template_mesh.faces
    tree = KDTree(vertices_holes)
    distances, indices = tree.query(vertices_template)
    mask = distances > 0.005
    retained_vertices = vertices_template[mask]

    old_to_new = -np.ones(len(vertices_template), dtype=int)
    old_to_new[mask] = np.arange(len(retained_vertices))
    # Filter faces: Keep faces only if all vertices are retained
    retained_faces = []
    for face in faces_template:
        if all(old_to_new[v] != -1 for v in face):  # If all vertices in the face are kept
            retained_faces.append([old_to_new[v] for v in face])

    retained_faces = np.array(retained_faces)
    holes_template = Trimesh(vertices=retained_vertices, faces=retained_faces)

    #STEP 5 : merge the template holes with the mesh with holes

    # Find boundary vertices of both meshes
    boundary_holes = find_boundary_vertices(holes_template)
    boundary_template = find_boundary_vertices(template_mesh_with_holes)

    # Extract boundary coordinates
    boundary_coords_holes = holes_template.vertices[boundary_holes]
    boundary_coords_template = template_mesh_with_holes.vertices[boundary_template]

    # Build KDTree from template mesh boundary
    tree = KDTree(boundary_coords_template)

    # Find nearest neighbors
    distances, nn_indices = tree.query(boundary_coords_holes)

    # Replace boundary vertices in holes_template with corresponding ones in template_mesh_with_holes
    holes_template.vertices[boundary_holes] = boundary_coords_template[nn_indices]

    # Merge both meshes
    merged_mesh = trimesh.util.concatenate([template_mesh_with_holes, holes_template])

    # **Fill remaining holes**
    ms=trimesh_to_pymeshlab(merged_mesh)
    ms.meshing_close_holes()

    # Save  the result
    ms.save_current_mesh(output_file)


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


