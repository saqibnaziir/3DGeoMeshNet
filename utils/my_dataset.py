##################################### WIth PC
# import os.path as osp
# import random
# import numpy as np
# import torch
# from torch_geometric.data import Data, Dataset, InMemoryDataset
# from tqdm import tqdm
# from trimesh import Trimesh, load_mesh
# from utils.funcs import save_ply_explicit, get_edge_index
# from concurrent.futures import ProcessPoolExecutor
# from glob import glob
# import pickle  # For loading curvature data


# class Normalize(object):
#     def __init__(self, mean=None, std=None, mean_subtraction_only=True):
#         """
#         Initialize normalization with optional mean subtraction only mode
        
#         Args:
#             mean (torch.Tensor, optional): Mean to subtract
#             std (torch.Tensor, optional): Standard deviation to divide by
#             mean_subtraction_only (bool, optional): If True, only perform mean subtraction
#         """
#         self.mean = mean
#         self.std = None if mean_subtraction_only else std
#         self.mean_subtraction_only = mean_subtraction_only

#     def __call__(self, data):
#         """
#         Normalize the data by subtracting mean and optionally dividing by std
        
#         Args:
#             data (torch_geometric.data.Data): Input data to normalize
        
#         Returns:
#             torch_geometric.data.Data: Normalized data
#         """
#         assert self.mean is not None, 'Initialize mean to normalize with'
#         if not self.mean_subtraction_only:
#             assert self.std is not None, 'Initialize std to normalize with'
        
#         mean = torch.as_tensor(self.mean, dtype=data.x.dtype, device=data.x.device)
#         data.x = data.x - mean
        
#         if not self.mean_subtraction_only:
#             std = torch.as_tensor(self.std, dtype=data.x.dtype, device=data.x.device)
#             data.x = data.x / std
        
#         data.x.nan_to_num_(0.0)
#         return data


# class MyDataset(InMemoryDataset):
#     def __init__(self, config, dtype='train', mean_subtraction_only=False):
#         """
#         Initialize the dataset with optional mean subtraction mode
        
#         Args:
#             config (dict): Configuration dictionary
#             dtype (str, optional): Dataset type ('train' or 'eval')
#             mean_subtraction_only (bool, optional): If True, use 1.0 for std
#         """
#         assert dtype in ['train', 'eval'], "Invalid dtype!"

#         self.config = config
#         self.root = config['dataset_dir']
#         self.mean_subtraction_only = mean_subtraction_only

#         super(MyDataset, self).__init__(self.root)

#         data_path = self.processed_paths[0] if dtype == 'train' else self.processed_paths[1]
#         norm_path = self.processed_paths[2]
#         edge_index_path = self.processed_paths[3]

#         norm_dict = torch.load(norm_path)
#         self.mean, self.std = norm_dict['mean'], norm_dict['std']
#         self.data, self.slices = torch.load(data_path)
#         self.edge_index = torch.load(edge_index_path)

#     @property
#     def processed_file_names(self):
#         return ['training.pt', 'eval.pt', 'norm.pt', "edge_index.pt"]

#     def process(self):
#         meshes = []
#         train_data, eval_data = [], []
#         train_features = []
#         train_dir = osp.join(self.root, "train")

#         files = glob(osp.join(train_dir, '**', '*.ply'), recursive=True)
#         curvature_path = osp.join(self.root, "curvature_data.pkl")
#         with open(curvature_path, "rb") as f:
#             curvature_data = pickle.load(f)

#         with ProcessPoolExecutor(max_workers=8) as executor:
#             futures = [executor.submit(load_mesh, file) for file in files]
#             for future in tqdm(futures):
#                 meshes.append(future.result())

#         edge_index = get_edge_index(meshes[0].vertices, meshes[0].faces)
#         random.shuffle(meshes)
#         count = int(0.8 * len(meshes))

#         for i in range(len(meshes)):
#             mesh_verts = torch.Tensor(meshes[i].vertices)
#             filename = osp.basename(files[i])
#             curvatures = torch.Tensor(curvature_data[filename]["curvatures"])

#             # Normalize curvatures
#             curvature_mean = curvatures.mean(dim=0, keepdim=True)
#             curvature_std = curvatures.std(dim=0, keepdim=True)
#             curvature_std[curvature_std == 0] = 1.0
#             normalized_curvatures = (curvatures - curvature_mean) / curvature_std

#             # Combine features
#             node_features = torch.cat([mesh_verts, normalized_curvatures], dim=1)
#             data = Data(x=mesh_verts, y=mesh_verts, edge_index=edge_index)

#             if i < count:
#                 train_data.append(data)
#                 train_features.append(node_features)
#             else:
#                 eval_data.append(data)

#         # Calculate mean and std for all features
#         mean_train = torch.mean(torch.stack(train_features), dim=0)
#         std_train = torch.std(torch.stack(train_features), dim=0)
#         std_train[std_train == 0] = 1.0  # Avoid division by zero
        
#         norm_dict = {'mean': mean_train, 'std': None if self.mean_subtraction_only else std_train}
        
#         # Extract the vertex-related portion of the mean
#         vertex_mean = mean_train[:, :3]  # Select the first 3 dimensions (x, y, z)
#         mesh = Trimesh(vertices=vertex_mean, faces=meshes[0].faces)  # Use vertex-related mean
#         save_ply_explicit(mesh, self.config['template_file'])

#         print("Transforming...")
#         transform = Normalize(mean_train, std_train, mean_subtraction_only=self.mean_subtraction_only)
#         train_data = [transform(x) for x in train_data]
#         eval_data = [transform(x) for x in eval_data]

#         print("Saving...")
#         torch.save(self.collate(train_data), self.processed_paths[0])
#         torch.save(self.collate(eval_data), self.processed_paths[1])
#         torch.save(norm_dict, self.processed_paths[2])
#         torch.save(edge_index, self.processed_paths[3])
#         torch.save(norm_dict, osp.join(self.config['dataset_dir'], "norm.pt"))
        
      



############################## Backup without PC #####################
import os.path as osp
import random
import numpy as np
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from tqdm import tqdm
from trimesh import Trimesh, load_mesh
import os
from utils.funcs import save_ply_explicit, get_edge_index
from concurrent.futures import ProcessPoolExecutor
from glob import glob
import pickle  # For loading curvature data

class Normalize(object):
    def __init__(self, mean=None, std=None, mean_subtraction_only=True):
        """
        Initialize normalization with optional mean subtraction only mode
        
        Args:
            mean (torch.Tensor, optional): Mean to subtract
            std (torch.Tensor, optional): Standard deviation to divide by
            mean_subtraction_only (bool, optional): If True, only perform mean subtraction
        """
        self.mean = mean
        # Store std only if we're not doing mean subtraction only
        self.std = None if mean_subtraction_only else std
        self.mean_subtraction_only = mean_subtraction_only

    def __call__(self, data):
        """
        Normalize the data by subtracting mean and optionally dividing by std
        
        Args:
            data (torch_geometric.data.Data): Input data to normalize
        
        Returns:
            torch_geometric.data.Data: Normalized data
        """
        assert self.mean is not None, 'Initialize mean to normalize with'
        if not self.mean_subtraction_only:
            assert self.std is not None, 'Initialize std to normalize with'
        
        # Convert mean to tensor with same dtype and device as data
        mean = torch.as_tensor(self.mean, dtype=data.x.dtype, device=data.x.device)
        
        # Subtract mean from x and y
        data.x = data.x - mean
        data.y = data.y - mean
        
        # Only divide by std if not mean_subtraction_only
        if not self.mean_subtraction_only:
            std = torch.as_tensor(self.std, dtype=data.x.dtype, device=data.x.device)
            data.x = data.x / std
            data.y = data.y / std
        
        # Handle NaN values in one go
        data.x.nan_to_num_(0.0)
        data.y.nan_to_num_(0.0)
        
        return data


class MyDataset(InMemoryDataset):

    def __init__(self, config, dtype='train', mean_subtraction_only=False):
        """
        Initialize the dataset with optional mean subtraction mode
        
        Args:
            config (dict): Configuration dictionary
            dtype (str, optional): Dataset type ('train' or 'eval')
            mean_subtraction_only (bool, optional): If True, use 1.0 for std
        """
        assert dtype in ['train', 'eval'], "Invalid dtype!"

        self.config = config
        self.root = config['dataset_dir']
        self.mean_subtraction_only = mean_subtraction_only

        super(MyDataset, self).__init__(self.root)

        data_path = self.processed_paths[0]
        if dtype == 'eval':
            data_path = self.processed_paths[1]
        norm_path = self.processed_paths[2]
        edge_index_path = self.processed_paths[3]

        norm_dict = torch.load(norm_path, weights_only=False)
        self.mean, self.std = norm_dict['mean'], norm_dict['std']
        self.data, self.slices = torch.load(data_path, weights_only=False)
        self.edge_index = torch.load(edge_index_path, weights_only=False)

    @property
    def processed_file_names(self):
        processed_files = ['training.pt', 'eval.pt', 'norm.pt', "edge_index.pt"]
        return processed_files

    def process(self):
        meshes = []
        train_data, eval_data = [], []
        train_vertices = []
        train_dir = osp.join(self.root, "train")

        # Recursively find all .ply files
        files = glob(osp.join(train_dir, '**', '*.ply'), recursive=True)
        
        # Load precomputed curvature data
        curvature_path = osp.join(self.root, "curvature_data.pkl")
        with open(curvature_path, "rb") as f:
            curvature_data = pickle.load(f)
            ######%%%%%%%%%%

        # Load meshes using concurrent processing
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(load_mesh, file) for file in files]
            for future in tqdm(futures):
                meshes.append(future.result())

        edge_index = get_edge_index(meshes[0].vertices, meshes[0].faces)

        # Shuffle meshes
        random.shuffle(meshes)
        count = int(0.8 * len(meshes))
        
        for i in range(len(meshes)):
            mesh_verts = torch.Tensor(meshes[i].vertices)
            
            # Add curvature to node features
            filename = osp.basename(files[i])
            curvatures = curvature_data[filename]["curvatures"]
            curvatures = torch.Tensor(curvatures)
            
            # ====== Add validation check ======
            if mesh_verts.shape[0] != curvatures.shape[0]:
                print(f"Error in {filename}:")
                print(f"Mesh vertices: {mesh_verts.shape[0]}")
                print(f"Curvatures: {curvatures.shape[0]}")
                raise ValueError("Vertex count mismatch between mesh and curvature data!")
            
            # Combine vertex coordinates with curvature
            node_features = torch.cat([mesh_verts, curvatures], dim=1)
            data = Data(x=mesh_verts, y=mesh_verts, edge_index=edge_index)
            
            if i < count:
                train_data.append(data)
                train_vertices.append(mesh_verts)
            else:
                eval_data.append(data)

        # Calculate mean and std
        mean_train = torch.Tensor(torch.mean(torch.stack(train_vertices), dim=0))
        std_train = torch.Tensor(torch.std(torch.stack(train_vertices), dim=0))
        # Handle any zero standard deviations to avoid division by zero
        std_train[std_train == 0] = 1.0
        
        # If mean_subtraction_only is True, set std to ones
        # In the process method:
        if self.mean_subtraction_only:
            norm_dict = {'mean': mean_train, 'std': None}
        else:
            std_train = torch.std(torch.stack(train_vertices), dim=0)
            std_train[std_train == 0] = 1.0
            norm_dict = {'mean': mean_train, 'std': std_train}
        
        # if self.mean_subtraction_only:
        #     std_train = torch.ones_like(std_train)
        
        # norm_dict = {'mean': mean_train, 'std': std_train}

        # Save template mesh
        mesh = Trimesh(vertices=mean_train, faces=meshes[0].faces)
        save_ply_explicit(mesh, self.config['template_file'])

        print("Transforming...")
        # Use the Normalize class with potential mean_subtraction_only option
        transform = Normalize(mean_train, std_train,   mean_subtraction_only=self.mean_subtraction_only)
        train_data = [transform(x) for x in train_data]
        eval_data = [transform(x) for x in eval_data]

        # Save processed data
        print("Saving...")
        torch.save(self.collate(train_data), self.processed_paths[0])
        torch.save(self.collate(eval_data), self.processed_paths[1])
        torch.save(norm_dict, self.processed_paths[2])
        torch.save(edge_index, self.processed_paths[3])
        torch.save(norm_dict, osp.join(self.config['dataset_dir'], "norm.pt"))





# ############## 18 Nov wo Principle #######
# import os.path as osp
# import random
# import numpy as np
# import torch
# from torch_geometric.data import Data, Dataset, InMemoryDataset
# from tqdm import tqdm
# from trimesh import Trimesh, load_mesh
# import os
# from utils.funcs import save_ply_explicit, get_edge_index
# from concurrent.futures import ProcessPoolExecutor
# from glob import glob

# class Normalize(object):
#     def __init__(self, mean=None, std=None):
#         self.mean = mean
#         self.std = std

#     def __call__(self, data):
#         assert self.mean is not None and self.std is not None, ('Initialize mean and std to normalize with')
#         self.mean = torch.as_tensor(self.mean, dtype=data.x.dtype, device=data.x.device)
#         self.std = torch.as_tensor(self.std, dtype=data.x.dtype, device=data.x.device)
#         data.x = (data.x - self.mean) / self.std
#         data.y = (data.y - self.mean) / self.std
#         return data

# #####VVVV
# def load_mesh_file(file):
#     return load_mesh(file)
# # def load_mesh_file(file, train_dir):
# #     return load_mesh(osp.join(train_dir, file))


# class MyDataset(InMemoryDataset):

#     def __init__(self, config, dtype='train'):
#         assert dtype in ['train', 'eval'], "Invalid dtype!"

#         self.config = config
#         self.root = config['dataset_dir']

#         super(MyDataset, self).__init__(self.root)

#         data_path = self.processed_paths[0]
#         if dtype == 'eval':
#             data_path = self.processed_paths[1]
#         norm_path = self.processed_paths[2]
#         edge_index_path = self.processed_paths[3]

#         norm_dict = torch.load(norm_path)
#         self.mean, self.std = norm_dict['mean'], norm_dict['std']
#         self.data, self.slices = torch.load(data_path)
#         self.edge_index = torch.load(edge_index_path)

#     @property
#     def processed_file_names(self):
#         processed_files = ['training.pt', 'eval.pt', 'norm.pt', "edge_index.pt"]
#         return processed_files

#     def process(self):
#         meshes = []
#         train_data, eval_data = [], []
#         train_vertices = []

#         train_dir = osp.join(self.root, "train")
        
        
#         ###############VVVV
#         # files = os.listdir(train_dir)
#         files = glob(osp.join(train_dir, '**', '*.ply'), recursive=True)

#         # for file in tqdm(files):
#         #     mesh = load_mesh(osp.join(train_dir, file))
#         #     meshes.append(mesh)

#         with ProcessPoolExecutor(max_workers=8) as executor:  # 调整max_workers的数量以达到最佳性能
#             ########VVV    
#             # futures = [executor.submit(load_mesh_file, file, train_dir) for file in files]
#             futures = [executor.submit(load_mesh_file, file) for file in files]
#             for future in tqdm(futures):
#                 meshes.append(future.result())

#         edge_index = get_edge_index(meshes[0].vertices, meshes[0].faces)

#         # shuffle
#         random.shuffle(meshes)
#         count = int(0.8 * len(meshes))
#         for i in range(len(meshes)):
#             mesh_verts = torch.Tensor(meshes[i].vertices)
#             data = Data(x=mesh_verts, y=mesh_verts, edge_index=edge_index)
#             if i < count:
#                 train_data.append(data)
#                 train_vertices.append(mesh_verts)
#             else:
#                 eval_data.append(data)

#         mean_train = torch.Tensor(np.mean(train_vertices, axis=0))
#         std_train = torch.Tensor(np.std(train_vertices, axis=0))
#         norm_dict = {'mean': mean_train, 'std': std_train}

#         # save template
#         mesh = Trimesh(vertices=mean_train, faces=meshes[0].faces)
#         save_ply_explicit(mesh, self.config['template_file'])

#         print("transforming...")
#         transform = Normalize(mean_train, std_train)
#         train_data = [transform(x) for x in train_data]
#         eval_data = [transform(x) for x in eval_data]

#         # save
#         print("saving...")
#         torch.save(self.collate(train_data), self.processed_paths[0])
#         torch.save(self.collate(eval_data), self.processed_paths[1])
#         torch.save(norm_dict, self.processed_paths[2])
#         torch.save(edge_index, self.processed_paths[3])
#         torch.save(norm_dict, osp.join(self.config['dataset_dir'], "norm.pt"))
