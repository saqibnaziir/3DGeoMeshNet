# #### with Global encoder decodr path only Date 06 April

# import torch
# import torch.nn.functional as F
# from torch_geometric.nn.conv import FeaStConv
# from torch_geometric.nn import BatchNorm
# from torch_geometric.data.batch import Batch

# class FMGenEncoder(torch.nn.Module):
#     def __init__(self, config, A, D):
#         super(FMGenEncoder, self).__init__()
#         self.A = [torch.tensor(a, requires_grad=False) for a in A]
#         self.D = [torch.tensor(a, requires_grad=False) for a in D]

#         self.batch_norm = config['batch_norm']
#         self.n_layers = config['n_layers']
#         self.z_length = config['z_length']
#         self.num_features_global = config['num_features_global']
#         self.num_features_local = config['num_features_local']

#         # Conv layers for global path
#         self.encoder_convs_global = torch.nn.ModuleList([
#             FeaStConv(in_channels=self.num_features_global[k], out_channels=self.num_features_global[k + 1])
#             for k in range(self.n_layers)
#         ])
        
#         # # Conv layers for local path - COMMENTED OUT
#         # self.encoder_convs_local = torch.nn.ModuleList([
#         #     FeaStConv(in_channels=self.num_features_local[k], out_channels=self.num_features_local[k + 1])
#         #     for k in range(self.n_layers)
#         # ])

#         # # Residual projection layers for local path to ensure dimension match - COMMENTED OUT
#         # self.local_residual_projections = torch.nn.ModuleList([
#         #     torch.nn.Linear(self.num_features_local[k], self.num_features_local[k + 1])
#         #     if self.num_features_local[k] != self.num_features_local[k + 1]
#         #     else torch.nn.Identity()
#         #     for k in range(self.n_layers)
#         # ])

#         # Global-related linear layer
#         self.encoder_lin = torch.nn.Linear(self.num_features_global[-1], self.z_length)
        
#         # # Modified linear layer for local path only (outputs directly to z) - COMMENTED OUT
#         # self.encoder_local_lin = torch.nn.Linear(self.D[0].shape[1] * self.num_features_local[-1], self.z_length)

#         self.reset_parameter()

#     def reset_parameter(self):
#         torch.nn.init.normal_(self.encoder_lin.weight, 0, 0.1)
#         # torch.nn.init.normal_(self.encoder_local_lin.weight, 0, 0.1)  # COMMENTED OUT

#     def forward(self, x, batch_size):
#         self.A = [a.to(x.device) for a in self.A]
#         self.D = [d.to(x.device) for d in self.D]

#         x_global = x
#         # x_local = x  # COMMENTED OUT

#         # Global path
#         for i in range(self.n_layers):
#             x_global = self.encoder_convs_global[i](x=x_global, edge_index=self.A[i])
#             x_global = F.leaky_relu(x_global)
#             x_global = x_global.reshape(batch_size, -1, self.num_features_global[i + 1])
#             y = torch.zeros(batch_size, self.D[i].shape[0], x_global.shape[2], device=x_global.device)
#             for j in range(batch_size):
#                 y[j] = torch.mm(self.D[i], x_global[j])
#             x_global = y
#             x_global = x_global.reshape(-1, self.num_features_global[i + 1])

#         x_global = x_global.reshape(batch_size, -1, self.num_features_global[-1])
#         x_global = torch.mean(x_global, dim=1)
#         x_global = F.leaky_relu(x_global)

#         # # Local path with residual connections - COMMENTED OUT
#         # for i in range(self.n_layers):
#         #     identity = x_local
#         #     x_local = self.encoder_convs_local[i](x=x_local, edge_index=self.A[0])
#         #     x_local = F.leaky_relu(x_local)
#         #     
#         #     # Add residual connection with dimension matching
#         #     x_local = x_local + self.local_residual_projections[i](identity)
#         # 
#         # x_local = x_local.reshape(batch_size, -1)
#         # z = self.encoder_local_lin(x_local)
#         # z = F.leaky_relu(z)

#         # Use only global representation for z
#         z = self.encoder_lin(x_global)
#         z = F.leaky_relu(z)

#         return z
# class FMGenDecoder(torch.nn.Module):
#     def __init__(self, config, A, U):
#         super(FMGenDecoder, self).__init__()
#         self.A = [torch.tensor(a, requires_grad=False) for a in A]
#         self.U = [torch.tensor(u, requires_grad=False) for u in U]

#         self.batch_norm = config['batch_norm']
#         self.n_layers = config['n_layers']
#         self.z_length = config['z_length']
#         self.num_features_global = config['num_features_global']
#         self.num_features_local = config['num_features_local']

#         # Conv layers for global path
#         self.decoder_convs_global = torch.nn.ModuleList([
#             FeaStConv(in_channels=self.num_features_global[-1 - k], out_channels=self.num_features_global[-2 - k])
#             for k in range(self.n_layers)
#         ])
        
#         # # Conv layers for local path - COMMENTED OUT
#         # self.decoder_convs_local = torch.nn.ModuleList([
#         #     FeaStConv(in_channels=self.num_features_local[-1 - k], out_channels=self.num_features_local[-2 - k])
#         #     for k in range(self.n_layers)
#         # ])

#         # # Add residual projection layers for local path - COMMENTED OUT
#         # self.local_residual_projections = torch.nn.ModuleList([
#         #     torch.nn.Linear(self.num_features_local[-1 - k], self.num_features_local[-2 - k])
#         #     if self.num_features_local[-1 - k] != self.num_features_local[-2 - k]
#         #     else torch.nn.Identity()
#         #     for k in range(self.n_layers)
#         # ])

#         # Global-related linear layer
#         self.decoder_lin = torch.nn.Linear(self.z_length, self.num_features_global[-1])
        
#         # # Modified linear layer for local path only (from z directly) - COMMENTED OUT
#         # self.decoder_local_lin = torch.nn.Linear(self.z_length, self.num_features_local[-1] * self.U[0].shape[0])

#         self.reset_parameter()

#     def reset_parameter(self):
#         torch.nn.init.normal_(self.decoder_lin.weight, 0, 0.1)
#         # torch.nn.init.normal_(self.decoder_local_lin.weight, 0, 0.1)  # COMMENTED OUT

#     def forward(self, z, batch_size):
#         self.A = [a.to(z.device) for a in self.A]
#         self.U = [u.to(z.device) for u in self.U]

#         # Global path
#         x_global = self.decoder_lin(z)
        
#         x_global = torch.unsqueeze(x_global, dim=1).repeat(1, self.U[-1].shape[1], 1)
#         for i in range(self.n_layers):
#             x_global = x_global.reshape(batch_size, -1, self.num_features_global[-1 - i])
#             y = torch.zeros(batch_size, self.U[-1 - i].shape[0], x_global.shape[2], device=x_global.device)
#             for j in range(batch_size):
#                 y[j] = torch.mm(self.U[-1 - i], x_global[j])
#             x_global = y
#             x_global = x_global.reshape(-1, self.num_features_global[-1 - i])
#             x_global = self.decoder_convs_global[i](x=x_global, edge_index=self.A[-2 - i])
#             if i < self.n_layers - 1:
#                 x_global = F.leaky_relu(x_global)
        
#         x_global = x_global.reshape(-1, self.num_features_global[0])

#         # # Local path with residual connections - COMMENTED OUT
#         # x_local = self.decoder_local_lin(z)
#         # x_local = x_local.reshape(-1, self.num_features_local[-1])
#         # for i in range(self.n_layers):
#         #     identity = x_local  # Store identity for residual connection
#         #     x_local = self.decoder_convs_local[i](x=x_local, edge_index=self.A[0])
#         #     if i < self.n_layers - 1:
#         #         x_local = F.leaky_relu(x_local)
#         #     # Add residual connection with dimension matching
#         #     x_local = x_local + self.local_residual_projections[i](identity)

#         # Just use global output directly
#         x = x_global

#         return x

# class FMGenModel(torch.nn.Module):
#     def __init__(self, config, A, D, U):
#         super(FMGenModel, self).__init__()

#         self.encoder = FMGenEncoder(config, A, D)
#         self.decoder = FMGenDecoder(config, A, U)

#     def forward(self, batch: Batch):
#         batch_size = batch.num_graphs
#         z = self.encoder(batch.x, batch_size)
#         x_reconstructed = self.decoder(z, batch_size)
#         return x_reconstructed, z

########## with local encoder decodr path only Date 02 April>>>>>>>>>>>>>>>>>>>>>>>>
# import torch
# import torch.nn.functional as F
# from torch_geometric.nn.conv import FeaStConv
# from torch_geometric.nn import BatchNorm
# from torch_geometric.data.batch import Batch

# class FMGenEncoder(torch.nn.Module):
#     def __init__(self, config, A, D):
#         super(FMGenEncoder, self).__init__()
#         self.A = [torch.tensor(a, requires_grad=False) for a in A]
#         self.D = [torch.tensor(a, requires_grad=False) for a in D]

#         self.batch_norm = config['batch_norm']
#         self.n_layers = config['n_layers']
#         self.z_length = config['z_length']
#         self.num_features_global = config['num_features_global']
#         self.num_features_local = config['num_features_local']

#         # # Conv layers for global path - COMMENTED OUT
#         # self.encoder_convs_global = torch.nn.ModuleList([
#         #     FeaStConv(in_channels=self.num_features_global[k], out_channels=self.num_features_global[k + 1])
#         #     for k in range(self.n_layers)
#         # ])
        
#         # Conv layers for local path
#         self.encoder_convs_local = torch.nn.ModuleList([
#             FeaStConv(in_channels=self.num_features_local[k], out_channels=self.num_features_local[k + 1])
#             for k in range(self.n_layers)
#         ])

#         # Residual projection layers for local path to ensure dimension match
#         self.local_residual_projections = torch.nn.ModuleList([
#             torch.nn.Linear(self.num_features_local[k], self.num_features_local[k + 1])
#             if self.num_features_local[k] != self.num_features_local[k + 1]
#             else torch.nn.Identity()
#             for k in range(self.n_layers)
#         ])

#         # # Global-related linear layer - COMMENTED OUT
#         # self.encoder_lin = torch.nn.Linear(self.z_length + self.num_features_global[-1], self.z_length)
        
#         # Modified linear layer for local path only (outputs directly to z)
#         self.encoder_local_lin = torch.nn.Linear(self.D[0].shape[1] * self.num_features_local[-1], self.z_length)

#         self.reset_parameter()

#     def reset_parameter(self):
#         # torch.nn.init.normal_(self.encoder_lin.weight, 0, 0.1)  # COMMENTED OUT
#         torch.nn.init.normal_(self.encoder_local_lin.weight, 0, 0.1)

#     def forward(self, x, batch_size):
#         self.A = [a.to(x.device) for a in self.A]
#         self.D = [d.to(x.device) for d in self.D]

#         # x_global = x  # COMMENTED OUT
#         x_local = x

#         # # Global path - COMMENTED OUT
#         # for i in range(self.n_layers):
#         #     x_global = self.encoder_convs_global[i](x=x_global, edge_index=self.A[i])
#         #     x_global = F.leaky_relu(x_global)
#         #     x_global = x_global.reshape(batch_size, -1, self.num_features_global[i + 1])
#         #     y = torch.zeros(batch_size, self.D[i].shape[0], x_global.shape[2], device=x_global.device)
#         #     for j in range(batch_size):
#         #         y[j] = torch.mm(self.D[i], x_global[j])
#         #     x_global = y
#         #     x_global = x_global.reshape(-1, self.num_features_global[i + 1])

#         # x_global = x_global.reshape(batch_size, -1, self.num_features_global[-1])
#         # x_global = torch.mean(x_global, dim=1)
#         # x_global = F.leaky_relu(x_global)

#         # Local path with residual connections
#         for i in range(self.n_layers):
#             identity = x_local
#             x_local = self.encoder_convs_local[i](x=x_local, edge_index=self.A[0])
#             x_local = F.leaky_relu(x_local)
            
#             # Add residual connection with dimension matching
#             x_local = x_local + self.local_residual_projections[i](identity)

#         x_local = x_local.reshape(batch_size, -1)
#         z = self.encoder_local_lin(x_local)
#         z = F.leaky_relu(z)

#         # # Concatenate global and local representations - COMMENTED OUT
#         # z = torch.concat((x_global, x_local), dim=1)
#         # z = self.encoder_lin(z)

#         return z

# class FMGenDecoder(torch.nn.Module):
#     def __init__(self, config, A, U):
#         super(FMGenDecoder, self).__init__()
#         self.A = [torch.tensor(a, requires_grad=False) for a in A]
#         self.U = [torch.tensor(u, requires_grad=False) for u in U]

#         self.batch_norm = config['batch_norm']
#         self.n_layers = config['n_layers']
#         self.z_length = config['z_length']
#         self.num_features_global = config['num_features_global']
#         self.num_features_local = config['num_features_local']

#         # # Conv layers for global path - COMMENTED OUT
#         # self.decoder_convs_global = torch.nn.ModuleList([
#         #     FeaStConv(in_channels=self.num_features_global[-1 - k], out_channels=self.num_features_global[-2 - k])
#         #     for k in range(self.n_layers)
#         # ])
        
#         # Conv layers for local path
#         self.decoder_convs_local = torch.nn.ModuleList([
#             FeaStConv(in_channels=self.num_features_local[-1 - k], out_channels=self.num_features_local[-2 - k])
#             for k in range(self.n_layers)
#         ])

#         # Add residual projection layers for local path
#         self.local_residual_projections = torch.nn.ModuleList([
#             torch.nn.Linear(self.num_features_local[-1 - k], self.num_features_local[-2 - k])
#             if self.num_features_local[-1 - k] != self.num_features_local[-2 - k]
#             else torch.nn.Identity()
#             for k in range(self.n_layers)
#         ])

#         # # Global-related linear layer - COMMENTED OUT
#         # self.decoder_lin = torch.nn.Linear(self.z_length, self.z_length + self.num_features_global[-1])
        
#         # Modified linear layer for local path only (from z directly)
#         self.decoder_local_lin = torch.nn.Linear(self.z_length, self.num_features_local[-1] * self.U[0].shape[0])

#         # # Attention network - COMMENTED OUT
#         # self.attention = torch.nn.Sequential(
#         #     torch.nn.Linear(self.num_features_global[0] + self.num_features_local[0], 64),
#         #     torch.nn.ReLU(),
#         #     torch.nn.Linear(64, 2),
#         #     torch.nn.Softmax(dim=-1)
#         # )

#         self.reset_parameter()

#     def reset_parameter(self):
#         # torch.nn.init.normal_(self.decoder_lin.weight, 0, 0.1)  # COMMENTED OUT
#         torch.nn.init.normal_(self.decoder_local_lin.weight, 0, 0.1)
#         # for layer in self.attention:  # COMMENTED OUT
#         #     if isinstance(layer, torch.nn.Linear):
#         #         torch.nn.init.normal_(layer.weight, 0, 0.1)

#     def forward(self, z, batch_size):
#         self.A = [a.to(z.device) for a in self.A]
#         self.U = [u.to(z.device) for u in self.U]

#         # # Global path - COMMENTED OUT
#         # x = self.decoder_lin(z)
#         # x_global = x[:, :self.num_features_global[-1]]
#         # x_local = x[:, self.num_features_global[-1]:]
#         # 
#         # x_global = torch.unsqueeze(x_global, dim=1).repeat(1, self.U[-1].shape[1], 1)
#         # for i in range(self.n_layers):
#         #     x_global = x_global.reshape(batch_size, -1, self.num_features_global[-1 - i])
#         #     y = torch.zeros(batch_size, self.U[-1 - i].shape[0], x_global.shape[2], device=x_global.device)
#         #     for j in range(batch_size):
#         #         y[j] = torch.mm(self.U[-1 - i], x_global[j])
#         #     x_global = y
#         #     x_global = x_global.reshape(-1, self.num_features_global[-1 - i])
#         #     x_global = self.decoder_convs_global[i](x=x_global, edge_index=self.A[-2 - i])
#         #     if i < self.n_layers - 1:
#         #         x_global = F.leaky_relu(x_global)
#         # 
#         # x_global = x_global.reshape(-1, self.num_features_global[0])

#         # Local path with residual connections
#         x_local = self.decoder_local_lin(z)
#         x_local = x_local.reshape(-1, self.num_features_local[-1])
#         for i in range(self.n_layers):
#             identity = x_local  # Store identity for residual connection
#             x_local = self.decoder_convs_local[i](x=x_local, edge_index=self.A[0])
#             if i < self.n_layers - 1:
#                 x_local = F.leaky_relu(x_local)
#             # Add residual connection with dimension matching
#             x_local = x_local + self.local_residual_projections[i](identity)

#         # # Adaptive merging - COMMENTED OUT
#         # concat_features = torch.cat([x_global, x_local], dim=-1)
#         # weights = self.attention(concat_features)
#         # x = weights[:, 0:1] * x_global + weights[:, 1:2] * x_local

#         # Just use local output directly
#         x = x_local

#         return x

# class FMGenModel(torch.nn.Module):
#     def __init__(self, config, A, D, U):
#         super(FMGenModel, self).__init__()

#         self.encoder = FMGenEncoder(config, A, D)
#         self.decoder = FMGenDecoder(config, A, U)

#     def forward(self, batch: Batch):
#         batch_size = batch.num_graphs
#         z = self.encoder(batch.x, batch_size)
#         x_reconstructed = self.decoder(z, batch_size)
#         return x_reconstructed, z
# class ModelSummary:
#     def __init__(self, model, config):
#         self.model = model
#         self.config = config

#     def print_summary(self, sample_batch: Batch):
#         print("FMGenModel Architecture Summary:")
#         print("================================")
        
#         self._print_encoder_summary(sample_batch)
#         self._print_latent_space_summary()
#         self._print_decoder_summary(sample_batch)
        
#         print("\nTotal parameters:", self._count_parameters())

#     def _print_encoder_summary(self, sample_batch: Batch):
#         print("\nEncoder:")
#         print("--------")
#         x = sample_batch.x
#         batch_size = sample_batch.num_graphs
#         num_vertices = x.shape[0] // batch_size

#         print("\nLocal Encoder:")
#         for i in range(self.model.encoder.n_layers):
#             in_features = self.model.encoder.num_features_local[i]
#             out_features = self.model.encoder.num_features_local[i + 1]
#             print(f"  Layer {i}:")
#             print(f"    FeaStConv: Input: ({num_vertices}, {in_features}) -> Output: ({num_vertices}, {out_features})")
#             print(f"    LeakyReLU activation")
#             print(f"    Residual connection with {'projection' if in_features != out_features else 'identity'}")

#         print(f"\nLocal Linear Layer: Input: ({batch_size}, {self.model.encoder.D[0].shape[1] * self.model.encoder.num_features_local[-1]}) -> Output: ({batch_size}, {self.model.encoder.z_length})")
#         print(f"    LeakyReLU activation")

#     def _print_latent_space_summary(self):
#         print("\nLatent Space:")
#         print("-------------")
#         z_length = self.model.encoder.z_length
#         print(f"Dimension: {z_length}")

#     def _print_decoder_summary(self, sample_batch: Batch):
#         print("\nDecoder:")
#         print("--------")
#         batch_size = sample_batch.num_graphs
#         final_num_vertices = self.model.decoder.U[0].shape[0]

#         print("Local Decoder:")
#         z_length = self.model.decoder.z_length
#         initial_local_features = self.model.decoder.num_features_local[-1]
#         print(f"  Linear Layer: Input: ({batch_size}, {z_length}) -> Output: ({batch_size}, {final_num_vertices * initial_local_features})")
#         print(f"  Reshape: ({batch_size}, {final_num_vertices * initial_local_features}) -> ({final_num_vertices}, {initial_local_features})")
        
#         for i in range(self.model.decoder.n_layers):
#             in_features = self.model.decoder.num_features_local[-1-i]
#             out_features = self.model.decoder.num_features_local[-2-i]
#             print(f"  Layer {i}:")
#             print(f"    FeaStConv: Input: ({final_num_vertices}, {in_features}) -> Output: ({final_num_vertices}, {out_features})")
#             if i < self.model.decoder.n_layers - 1:
#                 print(f"    LeakyReLU activation")
#             print(f"    Residual connection with {'projection' if in_features != out_features else 'identity'}")

#         print("\nFinal Output:")
#         out_features = sample_batch.x.shape[1]
#         print(f"  Local features: Output shape: ({final_num_vertices}, {out_features})")

#     def _count_parameters(self):
#         """Count the total number of trainable parameters in the model"""
#         return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

# # Add this line at the end of your model.py file
# FMGenModelSummary = ModelSummary


## 15 March with FEASTCONV Backup with resudail and attention network  ################# Local + Global with attention and residual connetions

import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import FeaStConv
from torch_geometric.nn import BatchNorm
from torch_geometric.data.batch import Batch

class FMGenEncoder(torch.nn.Module):
    def __init__(self, config, A, D):
        super(FMGenEncoder, self).__init__()
        self.A = [torch.tensor(a, requires_grad=False) for a in A]
        self.D = [torch.tensor(a, requires_grad=False) for a in D]

        self.batch_norm = config['batch_norm']
        self.n_layers = config['n_layers']
        self.z_length = config['z_length']
        self.num_features_global = config['num_features_global']
        self.num_features_local = config['num_features_local']

        # Conv layers for global and local paths
        self.encoder_convs_global = torch.nn.ModuleList([
            FeaStConv(in_channels=self.num_features_global[k], out_channels=self.num_features_global[k + 1])
            for k in range(self.n_layers)
        ])
        self.encoder_convs_local = torch.nn.ModuleList([
            FeaStConv(in_channels=self.num_features_local[k], out_channels=self.num_features_local[k + 1])
            for k in range(self.n_layers)
        ])

        # Residual projection layers for local path to ensure dimension match
        self.local_residual_projections = torch.nn.ModuleList([
            torch.nn.Linear(self.num_features_local[k], self.num_features_local[k + 1])
            if self.num_features_local[k] != self.num_features_local[k + 1]
            else torch.nn.Identity()
            for k in range(self.n_layers)
        ])

        # Linear layers
        self.encoder_lin = torch.nn.Linear(self.z_length + self.num_features_global[-1], self.z_length)
        self.encoder_local_lin = torch.nn.Linear(self.D[0].shape[1] * self.num_features_local[-1], self.z_length)

        self.reset_parameter()

    def reset_parameter(self):
        torch.nn.init.normal_(self.encoder_lin.weight, 0, 0.1)
        torch.nn.init.normal_(self.encoder_local_lin.weight, 0, 0.1)

    def forward(self, x, batch_size):
        self.A = [a.to(x.device) for a in self.A]
        self.D = [d.to(x.device) for d in self.D]

        x_global = x
        x_local = x

        # Global path
        for i in range(self.n_layers):
            x_global = self.encoder_convs_global[i](x=x_global, edge_index=self.A[i])
            x_global = F.leaky_relu(x_global)
            x_global = x_global.reshape(batch_size, -1, self.num_features_global[i + 1])
            y = torch.zeros(batch_size, self.D[i].shape[0], x_global.shape[2], device=x_global.device)
            for j in range(batch_size):
                y[j] = torch.mm(self.D[i], x_global[j])
            x_global = y
            x_global = x_global.reshape(-1, self.num_features_global[i + 1])

        x_global = x_global.reshape(batch_size, -1, self.num_features_global[-1])
        x_global = torch.mean(x_global, dim=1)
        x_global = F.leaky_relu(x_global)

        # Local path with residual connections
        for i in range(self.n_layers):
            identity = x_local
            x_local = self.encoder_convs_local[i](x=x_local, edge_index=self.A[0])
            x_local = F.leaky_relu(x_local)
            
            # Add residual connection with dimension matching   Comeneted
            x_local = x_local + self.local_residual_projections[i](identity)

        x_local = x_local.reshape(batch_size, -1)
        x_local = self.encoder_local_lin(x_local)
        x_local = F.leaky_relu(x_local)

        # Concatenate global and local representations
        z = torch.concat((x_global, x_local), dim=1)
        z = self.encoder_lin(z)

        return z



################################  FMGENDECODER with resiudal 
class FMGenDecoder(torch.nn.Module):
    def __init__(self, config, A, U):
        super(FMGenDecoder, self).__init__()
        self.A = [torch.tensor(a, requires_grad=False) for a in A]
        self.U = [torch.tensor(u, requires_grad=False) for u in U]

        self.batch_norm = config['batch_norm']
        self.n_layers = config['n_layers']
        self.z_length = config['z_length']
        self.num_features_global = config['num_features_global']
        self.num_features_local = config['num_features_local']

        # Conv layers for global and local paths
        self.decoder_convs_global = torch.nn.ModuleList([
            FeaStConv(in_channels=self.num_features_global[-1 - k], out_channels=self.num_features_global[-2 - k])
            for k in range(self.n_layers)
        ])
        self.decoder_convs_local = torch.nn.ModuleList([
            FeaStConv(in_channels=self.num_features_local[-1 - k], out_channels=self.num_features_local[-2 - k])
            for k in range(self.n_layers)
        ])

        # Add residual projection layers for local path (similar to encoder)
        self.local_residual_projections = torch.nn.ModuleList([
            torch.nn.Linear(self.num_features_local[-1 - k], self.num_features_local[-2 - k])
            if self.num_features_local[-1 - k] != self.num_features_local[-2 - k]
            else torch.nn.Identity()
            for k in range(self.n_layers)
        ])

        # Linear layers
        self.decoder_lin = torch.nn.Linear(self.z_length, self.z_length + self.num_features_global[-1])
        self.decoder_local_lin = torch.nn.Linear(self.z_length, self.num_features_local[-1] * self.U[0].shape[0])

        # Attention network for adaptive merging
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(self.num_features_global[0] + self.num_features_local[0], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
            torch.nn.Softmax(dim=-1)
        )

        self.reset_parameter()

    def reset_parameter(self):
        torch.nn.init.normal_(self.decoder_lin.weight, 0, 0.1)
        torch.nn.init.normal_(self.decoder_local_lin.weight, 0, 0.1)
        for layer in self.attention:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.normal_(layer.weight, 0, 0.1)

    def forward(self, z, batch_size):
        self.A = [a.to(z.device) for a in self.A]
        self.U = [u.to(z.device) for u in self.U]

        x = self.decoder_lin(z)
        x_global = x[:, :self.num_features_global[-1]]
        x_local = x[:, self.num_features_global[-1]:]

        # Global path (unchanged)
        x_global = torch.unsqueeze(x_global, dim=1).repeat(1, self.U[-1].shape[1], 1)
        for i in range(self.n_layers):
            x_global = x_global.reshape(batch_size, -1, self.num_features_global[-1 - i])
            y = torch.zeros(batch_size, self.U[-1 - i].shape[0], x_global.shape[2], device=x_global.device)
            for j in range(batch_size):
                y[j] = torch.mm(self.U[-1 - i], x_global[j])
            x_global = y
            x_global = x_global.reshape(-1, self.num_features_global[-1 - i])
            x_global = self.decoder_convs_global[i](x=x_global, edge_index=self.A[-2 - i])
            if i < self.n_layers - 1:
                x_global = F.leaky_relu(x_global)

        x_global = x_global.reshape(-1, self.num_features_global[0])

        # Local path with residual connections
        x_local = self.decoder_local_lin(x_local)
        x_local = x_local.reshape(-1, self.num_features_local[-1])
        for i in range(self.n_layers):
            identity = x_local  # Store identity for residual connection
            x_local = self.decoder_convs_local[i](x=x_local, edge_index=self.A[0])
            if i < self.n_layers - 1:
                x_local = F.leaky_relu(x_local)
            # Add residual connection with dimension matching
            x_local = x_local + self.local_residual_projections[i](identity)

        # Adaptive merging (unchanged)
        concat_features = torch.cat([x_global, x_local], dim=-1)
        weights = self.attention(concat_features)
        x = weights[:, 0:1] * x_global + weights[:, 1:2] * x_local

        return x

class FMGenModel(torch.nn.Module):
    def __init__(self, config, A, D, U):
        super(FMGenModel, self).__init__()

        self.encoder = FMGenEncoder(config, A, D)
        self.decoder = FMGenDecoder(config, A, U)

    def forward(self, batch: Batch):
        batch_size = batch.num_graphs
        z = self.encoder(batch.x, batch_size)
        x_reconstructed = self.decoder(z, batch_size)
        return x_reconstructed, z

######################### GAT CONV ###################VVVVVVVVVVVVVVVVV
# import torch
# import torch.nn.functional as F
# from torch_geometric.nn.conv import GATConv
# from torch_geometric.nn import BatchNorm
# from torch_geometric.data.batch import Batch

# class FMGenEncoder(torch.nn.Module):
#     def __init__(self, config, A, D):
#         super(FMGenEncoder, self).__init__()
#         self.A = [torch.tensor(a, requires_grad=False) for a in A]
#         self.D = [torch.tensor(a, requires_grad=False) for a in D]

#         self.batch_norm = config['batch_norm']
#         self.n_layers = config['n_layers']
#         self.z_length = config['z_length']
#         self.num_features_global = config['num_features_global']
#         self.num_features_local = config['num_features_local']

#         # Changed from FeaStConv to GATConv for global and local paths
#         self.encoder_convs_global = torch.nn.ModuleList([
#             GATConv(in_channels=self.num_features_global[k], out_channels=self.num_features_global[k + 1])
#             for k in range(self.n_layers)
#         ])
#         self.encoder_convs_local = torch.nn.ModuleList([
#             GATConv(in_channels=self.num_features_local[k], out_channels=self.num_features_local[k + 1])
#             for k in range(self.n_layers)
#         ])

#         # Residual projection layers for local path to ensure dimension match
#         self.local_residual_projections = torch.nn.ModuleList([
#             torch.nn.Linear(self.num_features_local[k], self.num_features_local[k + 1])
#             if self.num_features_local[k] != self.num_features_local[k + 1]
#             else torch.nn.Identity()
#             for k in range(self.n_layers)
#         ])

#         # Linear layers
#         self.encoder_lin = torch.nn.Linear(self.z_length + self.num_features_global[-1], self.z_length)
#         self.encoder_local_lin = torch.nn.Linear(self.D[0].shape[1] * self.num_features_local[-1], self.z_length)

#         self.reset_parameter()

#     def reset_parameter(self):
#         torch.nn.init.normal_(self.encoder_lin.weight, 0, 0.1)
#         torch.nn.init.normal_(self.encoder_local_lin.weight, 0, 0.1)

#     def forward(self, x, batch_size):
#         self.A = [a.to(x.device) for a in self.A]
#         self.D = [d.to(x.device) for d in self.D]

#         x_global = x
#         x_local = x

#         # Global path
#         for i in range(self.n_layers):
#             # Changed how GATConv is called (no x= parameter)
#             x_global = self.encoder_convs_global[i](x_global, self.A[i])
#             x_global = F.leaky_relu(x_global)
#             x_global = x_global.reshape(batch_size, -1, self.num_features_global[i + 1])
#             y = torch.zeros(batch_size, self.D[i].shape[0], x_global.shape[2], device=x_global.device)
#             for j in range(batch_size):
#                 y[j] = torch.mm(self.D[i], x_global[j])
#             x_global = y
#             x_global = x_global.reshape(-1, self.num_features_global[i + 1])

#         x_global = x_global.reshape(batch_size, -1, self.num_features_global[-1])
#         x_global = torch.mean(x_global, dim=1)
#         x_global = F.leaky_relu(x_global)

#         # Local path with residual connections
#         for i in range(self.n_layers):
#             identity = x_local
#             # Changed how GATConv is called (no x= parameter)
#             x_local = self.encoder_convs_local[i](x_local, self.A[0])
#             x_local = F.leaky_relu(x_local)
            
#             # Add residual connection with dimension matching
#             x_local = x_local + self.local_residual_projections[i](identity)

#         x_local = x_local.reshape(batch_size, -1)
#         x_local = self.encoder_local_lin(x_local)
#         x_local = F.leaky_relu(x_local)

#         # Concatenate global and local representations
#         z = torch.concat((x_global, x_local), dim=1)
#         z = self.encoder_lin(z)

#         return z
# class FMGenDecoder(torch.nn.Module):
#     def __init__(self, config, A, U):
#         super(FMGenDecoder, self).__init__()
#         self.A = [torch.tensor(a, requires_grad=False) for a in A]
#         self.U = [torch.tensor(u, requires_grad=False) for u in U]

#         self.batch_norm = config['batch_norm']
#         self.n_layers = config['n_layers']
#         self.z_length = config['z_length']
#         self.num_features_global = config['num_features_global']
#         self.num_features_local = config['num_features_local']

#         # Changed from FeaStConv to GATConv for global and local paths
#         self.decoder_convs_global = torch.nn.ModuleList([
#             GATConv(in_channels=self.num_features_global[-1 - k], out_channels=self.num_features_global[-2 - k])
#             for k in range(self.n_layers)
#         ])
#         self.decoder_convs_local = torch.nn.ModuleList([
#             GATConv(in_channels=self.num_features_local[-1 - k], out_channels=self.num_features_local[-2 - k])
#             for k in range(self.n_layers)
#         ])

#         # Add residual projection layers for local path (similar to encoder)
#         self.local_residual_projections = torch.nn.ModuleList([
#             torch.nn.Linear(self.num_features_local[-1 - k], self.num_features_local[-2 - k])
#             if self.num_features_local[-1 - k] != self.num_features_local[-2 - k]
#             else torch.nn.Identity()
#             for k in range(self.n_layers)
#         ])

#         # Linear layers
#         self.decoder_lin = torch.nn.Linear(self.z_length, self.z_length + self.num_features_global[-1])
#         self.decoder_local_lin = torch.nn.Linear(self.z_length, self.num_features_local[-1] * self.U[0].shape[0])

#         # Attention network for adaptive merging
#         self.attention = torch.nn.Sequential(
#             torch.nn.Linear(self.num_features_global[0] + self.num_features_local[0], 64),
#             torch.nn.ReLU(),
#             torch.nn.Linear(64, 2),
#             torch.nn.Softmax(dim=-1)
#         )

#         self.reset_parameter()

#     def reset_parameter(self):
#         torch.nn.init.normal_(self.decoder_lin.weight, 0, 0.1)
#         torch.nn.init.normal_(self.decoder_local_lin.weight, 0, 0.1)
#         for layer in self.attention:
#             if isinstance(layer, torch.nn.Linear):
#                 torch.nn.init.normal_(layer.weight, 0, 0.1)

#     def forward(self, z, batch_size):
#         self.A = [a.to(z.device) for a in self.A]
#         self.U = [u.to(z.device) for u in self.U]

#         x = self.decoder_lin(z)
#         x_global = x[:, :self.num_features_global[-1]]
#         x_local = x[:, self.num_features_global[-1]:]

#         # Global path (unchanged except for GATConv call)
#         x_global = torch.unsqueeze(x_global, dim=1).repeat(1, self.U[-1].shape[1], 1)
#         for i in range(self.n_layers):
#             x_global = x_global.reshape(batch_size, -1, self.num_features_global[-1 - i])
#             y = torch.zeros(batch_size, self.U[-1 - i].shape[0], x_global.shape[2], device=x_global.device)
#             for j in range(batch_size):
#                 y[j] = torch.mm(self.U[-1 - i], x_global[j])
#             x_global = y
#             x_global = x_global.reshape(-1, self.num_features_global[-1 - i])
#             # Changed how GATConv is called (no x= parameter)
#             x_global = self.decoder_convs_global[i](x_global, self.A[-2 - i])
#             if i < self.n_layers - 1:
#                 x_global = F.leaky_relu(x_global)

#         x_global = x_global.reshape(-1, self.num_features_global[0])

#         # Local path with residual connections
#         x_local = self.decoder_local_lin(x_local)
#         x_local = x_local.reshape(-1, self.num_features_local[-1])
#         for i in range(self.n_layers):
#             identity = x_local  # Store identity for residual connection
#             # Changed how GATConv is called (no x= parameter)
#             x_local = self.decoder_convs_local[i](x_local, self.A[0])
#             if i < self.n_layers - 1:
#                 x_local = F.leaky_relu(x_local)
#             # Add residual connection with dimension matching
#             x_local = x_local + self.local_residual_projections[i](identity)
#         # Adaptive merging (unchanged)
#         concat_features = torch.cat([x_global, x_local], dim=-1)
#         weights = self.attention(concat_features)
#         x = weights[:, 0:1] * x_global + weights[:, 1:2] * x_local
#         return x
# class FMGenModel(torch.nn.Module):
#     def __init__(self, config, A, D, U):
#         super(FMGenModel, self).__init__()
#         self.encoder = FMGenEncoder(config, A, D)
#         self.decoder = FMGenDecoder(config, A, U)
#     def forward(self, batch: Batch):
#         batch_size = batch.num_graphs
#         z = self.encoder(batch.x, batch_size)
#         x_reconstructed = self.decoder(z, batch_size)
#         return x_reconstructed, z
#################################### GAT COnv############^^^^^^^^^^^^
#################################### GC Conv###################VVVVVVVVVVVV
# import torch
# import torch.nn.functional as F
# from torch_geometric.nn.conv import GCNConv  # Changed from FeaStConv to GCNConv
# from torch_geometric.nn import BatchNorm
# from torch_geometric.data.batch import Batch

# class FMGenEncoder(torch.nn.Module):
#     def __init__(self, config, A, D):
#         super(FMGenEncoder, self).__init__()
#         self.A = [torch.tensor(a, requires_grad=False) for a in A]
#         self.D = [torch.tensor(a, requires_grad=False) for a in D]

#         self.batch_norm = config['batch_norm']
#         self.n_layers = config['n_layers']
#         self.z_length = config['z_length']
#         self.num_features_global = config['num_features_global']
#         self.num_features_local = config['num_features_local']

#         # Conv layers for global and local paths - Changed to GCNConv
#         self.encoder_convs_global = torch.nn.ModuleList([
#             GCNConv(in_channels=self.num_features_global[k], out_channels=self.num_features_global[k + 1])
#             for k in range(self.n_layers)
#         ])
#         self.encoder_convs_local = torch.nn.ModuleList([
#             GCNConv(in_channels=self.num_features_local[k], out_channels=self.num_features_local[k + 1])
#             for k in range(self.n_layers)
#         ])

#         # Residual projection layers for local path to ensure dimension match
#         self.local_residual_projections = torch.nn.ModuleList([
#             torch.nn.Linear(self.num_features_local[k], self.num_features_local[k + 1])
#             if self.num_features_local[k] != self.num_features_local[k + 1]
#             else torch.nn.Identity()
#             for k in range(self.n_layers)
#         ])

#         # Linear layers
#         self.encoder_lin = torch.nn.Linear(self.z_length + self.num_features_global[-1], self.z_length)
#         self.encoder_local_lin = torch.nn.Linear(self.D[0].shape[1] * self.num_features_local[-1], self.z_length)

#         self.reset_parameter()

#     def reset_parameter(self):
#         torch.nn.init.normal_(self.encoder_lin.weight, 0, 0.1)
#         torch.nn.init.normal_(self.encoder_local_lin.weight, 0, 0.1)

#     def forward(self, x, batch_size):
#         self.A = [a.to(x.device) for a in self.A]
#         self.D = [d.to(x.device) for d in self.D]

#         x_global = x
#         x_local = x

#         # Global path
#         for i in range(self.n_layers):
#             # GCNConv expects edge_index instead of adjacency matrix
#             x_global = self.encoder_convs_global[i](x=x_global, edge_index=self.A[i])
#             x_global = F.leaky_relu(x_global)
#             x_global = x_global.reshape(batch_size, -1, self.num_features_global[i + 1])
#             y = torch.zeros(batch_size, self.D[i].shape[0], x_global.shape[2], device=x_global.device)
#             for j in range(batch_size):
#                 y[j] = torch.mm(self.D[i], x_global[j])
#             x_global = y
#             x_global = x_global.reshape(-1, self.num_features_global[i + 1])

#         x_global = x_global.reshape(batch_size, -1, self.num_features_global[-1])
#         x_global = torch.mean(x_global, dim=1)
#         x_global = F.leaky_relu(x_global)

#         # Local path with residual connections
#         for i in range(self.n_layers):
#             identity = x_local
#             # GCNConv expects edge_index instead of adjacency matrix
#             x_local = self.encoder_convs_local[i](x=x_local, edge_index=self.A[0])
#             x_local = F.leaky_relu(x_local)
            
#             # Add residual connection with dimension matching
#             x_local = x_local + self.local_residual_projections[i](identity)

#         x_local = x_local.reshape(batch_size, -1)
#         x_local = self.encoder_local_lin(x_local)
#         x_local = F.leaky_relu(x_local)

#         # Concatenate global and local representations
#         z = torch.concat((x_global, x_local), dim=1)
#         z = self.encoder_lin(z)

#         return z


# class FMGenDecoder(torch.nn.Module):
#     def __init__(self, config, A, U):
#         super(FMGenDecoder, self).__init__()
#         self.A = [torch.tensor(a, requires_grad=False) for a in A]
#         self.U = [torch.tensor(u, requires_grad=False) for u in U]

#         self.batch_norm = config['batch_norm']
#         self.n_layers = config['n_layers']
#         self.z_length = config['z_length']
#         self.num_features_global = config['num_features_global']
#         self.num_features_local = config['num_features_local']

#         # Conv layers for global and local paths - Changed to GCNConv
#         self.decoder_convs_global = torch.nn.ModuleList([
#             GCNConv(in_channels=self.num_features_global[-1 - k], out_channels=self.num_features_global[-2 - k])
#             for k in range(self.n_layers)
#         ])
#         self.decoder_convs_local = torch.nn.ModuleList([
#             GCNConv(in_channels=self.num_features_local[-1 - k], out_channels=self.num_features_local[-2 - k])
#             for k in range(self.n_layers)
#         ])

#         # Add residual projection layers for local path (similar to encoder)
#         self.local_residual_projections = torch.nn.ModuleList([
#             torch.nn.Linear(self.num_features_local[-1 - k], self.num_features_local[-2 - k])
#             if self.num_features_local[-1 - k] != self.num_features_local[-2 - k]
#             else torch.nn.Identity()
#             for k in range(self.n_layers)
#         ])

#         # Linear layers
#         self.decoder_lin = torch.nn.Linear(self.z_length, self.z_length + self.num_features_global[-1])
#         self.decoder_local_lin = torch.nn.Linear(self.z_length, self.num_features_local[-1] * self.U[0].shape[0])

#         # Attention network for adaptive merging
#         self.attention = torch.nn.Sequential(
#             torch.nn.Linear(self.num_features_global[0] + self.num_features_local[0], 64),
#             torch.nn.ReLU(),
#             torch.nn.Linear(64, 2),
#             torch.nn.Softmax(dim=-1)
#         )

#         self.reset_parameter()

#     def reset_parameter(self):
#         torch.nn.init.normal_(self.decoder_lin.weight, 0, 0.1)
#         torch.nn.init.normal_(self.decoder_local_lin.weight, 0, 0.1)
#         for layer in self.attention:
#             if isinstance(layer, torch.nn.Linear):
#                 torch.nn.init.normal_(layer.weight, 0, 0.1)

#     def forward(self, z, batch_size):
#         self.A = [a.to(z.device) for a in self.A]
#         self.U = [u.to(z.device) for u in self.U]

#         x = self.decoder_lin(z)
#         x_global = x[:, :self.num_features_global[-1]]
#         x_local = x[:, self.num_features_global[-1]:]

#         # Global path
#         x_global = torch.unsqueeze(x_global, dim=1).repeat(1, self.U[-1].shape[1], 1)
#         for i in range(self.n_layers):
#             x_global = x_global.reshape(batch_size, -1, self.num_features_global[-1 - i])
#             y = torch.zeros(batch_size, self.U[-1 - i].shape[0], x_global.shape[2], device=x_global.device)
#             for j in range(batch_size):
#                 y[j] = torch.mm(self.U[-1 - i], x_global[j])
#             x_global = y
#             x_global = x_global.reshape(-1, self.num_features_global[-1 - i])
#             # GCNConv expects edge_index instead of adjacency matrix
#             x_global = self.decoder_convs_global[i](x=x_global, edge_index=self.A[-2 - i])
#             if i < self.n_layers - 1:
#                 x_global = F.leaky_relu(x_global)

#         x_global = x_global.reshape(-1, self.num_features_global[0])

#         # Local path with residual connections
#         x_local = self.decoder_local_lin(x_local)
#         x_local = x_local.reshape(-1, self.num_features_local[-1])
#         for i in range(self.n_layers):
#             identity = x_local  # Store identity for residual connection
#             # GCNConv expects edge_index instead of adjacency matrix
#             x_local = self.decoder_convs_local[i](x=x_local, edge_index=self.A[0])
#             if i < self.n_layers - 1:
#                 x_local = F.leaky_relu(x_local)
#             # Add residual connection with dimension matching
#             x_local = x_local + self.local_residual_projections[i](identity)

#         # Adaptive merging
#         concat_features = torch.cat([x_global, x_local], dim=-1)
#         weights = self.attention(concat_features)
#         x = weights[:, 0:1] * x_global + weights[:, 1:2] * x_local

#         return x

# class FMGenModel(torch.nn.Module):
#     def __init__(self, config, A, D, U):
#         super(FMGenModel, self).__init__()

#         self.encoder = FMGenEncoder(config, A, D)
#         self.decoder = FMGenDecoder(config, A, U)

#     def forward(self, batch: Batch):
#         batch_size = batch.num_graphs
#         z = self.encoder(batch.x, batch_size)
#         x_reconstructed = self.decoder(z, batch_size)
#         return x_reconstructed, z
    
# ################################# Summary overlall ^^^^^^^^^^^^^^^^
######################################################################
class ModelSummary:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def print_summary(self, sample_batch: Batch):
        print("FMGenModel Architecture Summary:")
        print("================================")
        
        self._print_encoder_summary(sample_batch)
        self._print_latent_space_summary()
        self._print_decoder_summary(sample_batch)
        
        print("\nTotal parameters:", self._count_parameters())

    def _print_encoder_summary(self, sample_batch: Batch):
        print("\nEncoder:")
        print("--------")
        x = sample_batch.x
        batch_size = sample_batch.num_graphs
        num_vertices = x.shape[0] // batch_size

        print("Global Encoder:")
        for i in range(self.model.encoder.n_layers):
            in_features = self.model.encoder.num_features_global[i]
            out_features = self.model.encoder.num_features_global[i + 1]
            print(f"  Layer {i}:")
            print(f"    FeaStConv: Input: ({num_vertices}, {in_features}) -> Output: ({num_vertices}, {out_features})")
            # if self.model.encoder.batch_norm:
            #     print(f"    BatchNorm")
            print(f"    LeakyReLU activation")
            print(f"    Downsampling: ({num_vertices}, {out_features}) -> ({self.model.encoder.D[i].shape[0]}, {out_features})")
            num_vertices = self.model.encoder.D[i].shape[0]

        print("\nLocal Encoder:")
        num_vertices = x.shape[0] // batch_size  # Reset to initial number of vertices
        for i in range(self.model.encoder.n_layers):
            in_features = self.model.encoder.num_features_local[i]
            out_features = self.model.encoder.num_features_local[i + 1]
            print(f"  Layer {i}:")
            print(f"    FeaStConv: Input: ({num_vertices}, {in_features}) -> Output: ({num_vertices}, {out_features})")
            # if self.model.encoder.batch_norm:
            #     print(f"    BatchNorm")
            print(f"    LeakyReLU activation")

        print(f"\nLocal Linear Layer: Input: ({batch_size}, {self.model.encoder.D[0].shape[1] * self.model.encoder.num_features_local[-1]}) -> Output: ({batch_size}, {self.model.encoder.z_length})")

    def _print_latent_space_summary(self):
        print("\nLatent Space:")
        print("-------------")
        z_length = self.model.encoder.z_length
        print(f"Dimension: {z_length}")

    def _print_decoder_summary(self, sample_batch: Batch):
        print("\nDecoder:")
        print("--------")
        batch_size = sample_batch.num_graphs
        final_num_vertices = self.model.decoder.U[0].shape[0]

        print("Global Decoder:")
        for i in range(self.model.decoder.n_layers):
            in_features = self.model.decoder.num_features_global[-1-i]
            out_features = self.model.decoder.num_features_global[-2-i]
            
            # Calculate the number of vertices before and after upsampling
            if i == 0:
                pre_upsample_vertices = self.model.decoder.U[-1].shape[1]
            else:
                pre_upsample_vertices = self.model.decoder.U[-i].shape[0]
            
            post_upsample_vertices = self.model.decoder.U[-1-i].shape[0]

            print(f"  Layer {i}:")
            print(f"    Upsampling: ({pre_upsample_vertices}, {in_features}) -> ({post_upsample_vertices}, {in_features})")
            print(f"    FeaStConv: Input: ({post_upsample_vertices}, {in_features}) -> Output: ({post_upsample_vertices}, {out_features})")
            if i < self.model.decoder.n_layers - 1:
                # if self.model.decoder.batch_norm:
                #     print(f"    BatchNorm")
                print(f"    LeakyReLU activation")

        print("\nLocal Decoder:")
        z_length = self.model.decoder.z_length
        initial_local_features = self.model.decoder.num_features_local[-1]
        print(f"  Linear Layer: Input: ({batch_size}, {z_length}) -> Output: ({batch_size}, {final_num_vertices * initial_local_features})")
        print(f"  Reshape: ({batch_size}, {final_num_vertices * initial_local_features}) -> ({final_num_vertices}, {initial_local_features})")
        
        for i in range(self.model.decoder.n_layers):
            in_features = self.model.decoder.num_features_local[-1-i]
            out_features = self.model.decoder.num_features_local[-2-i]
            print(f"  Layer {i}:")
            print(f"    FeaStConv: Input: ({final_num_vertices}, {in_features}) -> Output: ({final_num_vertices}, {out_features})")
            if i < self.model.decoder.n_layers - 1:
                # if self.model.decoder.batch_norm:
                #     print(f"    BatchNorm")
                print(f"    LeakyReLU activation")

        print("\nFinal Layer:")
        in_features = self.model.decoder.num_features_global[0] + self.model.decoder.num_features_local[0]
        out_features = sample_batch.x.shape[1]
        print(f"  Merge global and local features: Input: ({final_num_vertices}, {in_features}) -> Output: ({final_num_vertices}, {out_features})")

    def _count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

# Add this line at the end of your model.py file
FMGenModelSummary = ModelSummary

####################################################################
####################################################################



# #####12 November ####################### Local + Global #################### without residual
# import torch
# import torch.nn.functional as F
# from torch_geometric.nn.conv import FeaStConv
# from torch_geometric.nn import BatchNorm
# from torch_geometric.data.batch import Batch


# class FMGenEncoder(torch.nn.Module):
#     def __init__(self, config, A, D):
#         super(FMGenEncoder, self).__init__()
#         self.A = [torch.tensor(a, requires_grad=False) for a in A]
#         self.D = [torch.tensor(a, requires_grad=False) for a in D]

#         self.batch_norm = config['batch_norm']
#         self.n_layers = config['n_layers']
#         self.z_length = config['z_length']
#         self.num_features_global = config['num_features_global']
#         self.num_features_local = config['num_features_local']

#         # conv layers
#         self.encoder_convs_global = torch.nn.ModuleList([
#             FeaStConv(in_channels=self.num_features_global[k], out_channels=self.num_features_global[k + 1])
#             for k in range(self.n_layers)
#         ])
#         self.encoder_convs_local = torch.nn.ModuleList([
#             FeaStConv(in_channels=self.num_features_local[k], out_channels=self.num_features_local[k + 1])
#             for k in range(self.n_layers)
#         ])
       
#         ### Add residual projection layers for local path when input/output dimensions don't match
#         # self.local_residual_projections = torch.nn.ModuleList([
#         #     torch.nn.Linear(self.num_features_local[k], self.num_features_local[k + 1])
#         #     if self.num_features_local[k] != self.num_features_local[k + 1]
#         #     else torch.nn.Identity()
#         #     for k in range(self.n_layers)
#         # ])
        
        
#         # linear layers
#         self.encoder_lin = torch.nn.Linear(self.z_length + self.num_features_global[-1], self.z_length)
#         self.encoder_local_lin = torch.nn.Linear(self.D[0].shape[1] * self.num_features_local[-1], self.z_length)

#         self.reset_parameter()

#     def reset_parameter(self):
#         torch.nn.init.normal_(self.encoder_lin.weight, 0, 0.1)
#         torch.nn.init.normal_(self.encoder_local_lin.weight, 0, 0.1)
        

#     def forward(self, x, batch_size):
#         self.A = [a.to(x.device) for a in self.A]
#         self.D = [d.to(x.device) for d in self.D]

#         # Split into global and local paths
#         x_global = x
#         x_local = x

#         """
#             global
#         """
#         # x_global: [batch_size * D[0].shape[1], num_features_global[0]]
#         for i in range(self.n_layers):
#             x_global = self.encoder_convs_global[i](x=x_global, edge_index=self.A[i])
#             x_global = F.leaky_relu(x_global)
#             x_global = x_global.reshape(batch_size, -1, self.num_features_global[i + 1])
#             y = torch.zeros(batch_size, self.D[i].shape[0], x_global.shape[2], device=x_global.device)
#             for j in range(batch_size):
#                 y[j] = torch.mm(self.D[i], x_global[j])
#             x_global = y
#             x_global = x_global.reshape(-1, self.num_features_global[i + 1])

#         x_global = x_global.reshape(batch_size, -1, self.num_features_global[-1])

#         # (mean pool & relu)
#         x_global = torch.mean(x_global, dim=1)
#         x_global = F.leaky_relu(x_global)
#         # x_global: [batch_size, num_features_global[-1]]

#         """
#             local 
#         """
#         # begin x_local: [batch_size * D[0].shape[1], num_features_local[0]]
#         for i in range(self.n_layers):
#             # 卷积
#             # identity = x_local
            
#             x_local = self.encoder_convs_local[i](x=x_local, edge_index=self.A[0])
#             x_local = F.leaky_relu(x_local)

#         x_local = x_local.reshape(batch_size, -1)
#         # x_local: [batch_size, D[0].shape[1] * num_features_local[0]]

#         # (linear & relu)
#         x_local = self.encoder_local_lin(x_local)
#         x_local = F.leaky_relu(x_local)
#         # x_local: [batch_size, z_length]

#         """
#             get z
#         """
#         z = torch.concat((x_global, x_local), dim=1)
#         z = self.encoder_lin(z)

#         return z


# class FMGenDecoder(torch.nn.Module):
#     def __init__(self, config, A, U):
#         super(FMGenDecoder, self).__init__()
#         self.A = [torch.tensor(a, requires_grad=False) for a in A]
#         self.U = [torch.tensor(u, requires_grad=False) for u in U]

#         self.batch_norm = config['batch_norm']
#         self.n_layers = config['n_layers']
#         self.z_length = config['z_length']
#         self.num_features_global = config['num_features_global']
#         self.num_features_local = config['num_features_local']

#         # conv layers
#         self.decoder_convs_global = torch.nn.ModuleList([
#             FeaStConv(in_channels=self.num_features_global[-1 - k], out_channels=self.num_features_global[-2 - k])
#             for k in range(self.n_layers)
#         ])
#         self.decoder_convs_local = torch.nn.ModuleList([
#             FeaStConv(in_channels=self.num_features_local[-1 - k], out_channels=self.num_features_local[-2 - k])
#             for k in range(self.n_layers)
#         ])
        
#         # linear layers
#         self.decoder_lin = torch.nn.Linear(self.z_length, self.z_length + self.num_features_global[-1])
#         self.decoder_local_lin = torch.nn.Linear(self.z_length, self.num_features_local[-1] * self.U[0].shape[0])

#         # NEW: Attention network for adaptive merging
#         self.attention = torch.nn.Sequential(
#             torch.nn.Linear(self.num_features_global[0] + self.num_features_local[0], 64),
#             torch.nn.ReLU(),
#             torch.nn.Linear(64, 2),
#             torch.nn.Softmax(dim=-1)
#         )
               
#         self.reset_parameter()
# ########################################################## old fixed
#         # fixed ratios merge ratio
#         # self.global_ratio = 0.01
#         # self.local_ratio = 1 - self.global_ratio
#         ########## old VV
#     # def reset_parameter(self):
#     #     torch.nn.init.normal_(self.decoder_lin.weight, 0, 0.1)
#     #     torch.nn.init.normal_(self.decoder_local_lin.weight, 0, 0.1)
#     #     # Initialize residual projection layers
#     #     for layer in self.local_residual_projections:
#     #        if isinstance(layer, torch.nn.Linear):
#     #            torch.nn.init.normal_(layer.weight, 0, 0.1)
#     ############V new with attention
#     def reset_parameter(self):
#         torch.nn.init.normal_(self.decoder_lin.weight, 0, 0.1)
#         torch.nn.init.normal_(self.decoder_local_lin.weight, 0, 0.1)
#         # Initialize attention network layers
#         for layer in self.attention:
#             if isinstance(layer, torch.nn.Linear):
#                 torch.nn.init.normal_(layer.weight, 0, 0.1)
                
                
#     def forward(self, z, batch_size):
#         self.A = [a.to(z.device) for a in self.A]
#         self.U = [u.to(z.device) for u in self.U]

#         # decoder linear & 分叉
#         x = self.decoder_lin(z)
#         x_global = x[:, :self.num_features_global[-1]]
#         x_local = x[:, self.num_features_global[-1]:]

#         """
#             global
#         """
#         # x_global: [batch_size, num_features_global[-1]]
#         x_global = torch.unsqueeze(x_global, dim=1).repeat(1, self.U[-1].shape[1], 1)
#         # x_global: [batch_size, U[-1].shape[1], num_features_global[-1]]

#         for i in range(self.n_layers):
#             x_global = x_global.reshape(batch_size, -1, self.num_features_global[-1 - i])
#             y = torch.zeros(batch_size, self.U[-1 - i].shape[0], x_global.shape[2], device=x_global.device)
#             for j in range(batch_size):
#                 y[j] = torch.mm(self.U[-1 - i], x_global[j])
#             x_global = y
#             x_global = x_global.reshape(-1, self.num_features_global[-1 - i])
#             x_global = self.decoder_convs_global[i](x=x_global, edge_index=self.A[-2 - i])
#             if i < self.n_layers - 1:
#                 x_global = F.leaky_relu(x_global)
#         # x_global: [batch_size, U[0].shape[0], num_features_global[0]]
#         x_global = x_global.reshape(-1, self.num_features_global[0])
#         # x_global: [batch_size * U[0].shape[0], num_features_global[0]]

#         """
#             local 
#         """
#         # x_local: [batch_size, z_length]
#         x_local = self.decoder_local_lin(x_local)
#         # x_local: [batch_size, num_features_local[-1] * U[0].shape[0]]
#         x_local = x_local.reshape(-1, self.num_features_local[-1])
#         # x_local: [batch_size * U[0].shape[0], num_features_local[-1]]

#         for i in range(self.n_layers):
#             # 卷积
#             # Main path
#             x_local = self.decoder_convs_local[i](x=x_local, edge_index=self.A[0])
#             if i < self.n_layers - 1:
#                 x_local = F.leaky_relu(x_local)


#         """
#             merge NEW: attention-based merging
#         """
#         ### old
#         # x = self.global_ratio * x_global + self.local_ratio * x_local
        
#         # new Calculate adaptive weights
#         # Concatenate features for attention
#         concat_features = torch.cat([x_global, x_local], dim=-1)
        
#         # Compute attention weights
#         weights = self.attention(concat_features)
        
#         # Apply attention weights
#         x = weights[:, 0:1] * x_global + weights[:, 1:2] * x_local
        
#         return x


# class FMGenModel(torch.nn.Module):
#     def __init__(self, config, A, D, U):
#         super(FMGenModel, self).__init__()

#         self.encoder = FMGenEncoder(config, A, D)
#         self.decoder = FMGenDecoder(config, A, U)

#     def forward(self, batch: Batch):
#         batch_size = batch.num_graphs
#         z = self.encoder(batch.x, batch_size)
#         x_reconstructed = self.decoder(z, batch_size)
#         return x_reconstructed, z


# class ModelSummary:
#     def __init__(self, model, config):
#         self.model = model
#         self.config = config

#     def print_summary(self, sample_batch: Batch):
#         print("FMGenModel Architecture Summary:")
#         print("================================")
        
#         self._print_encoder_summary(sample_batch)
#         self._print_latent_space_summary()
#         self._print_decoder_summary(sample_batch)
        
#         print("\nTotal parameters:", self._count_parameters())

#     def _print_encoder_summary(self, sample_batch: Batch):
#         print("\nEncoder:")
#         print("--------")
#         x = sample_batch.x
#         batch_size = sample_batch.num_graphs
#         num_vertices = x.shape[0] // batch_size

#         print("Global Encoder:")
#         for i in range(self.model.encoder.n_layers):
#             in_features = self.model.encoder.num_features_global[i]
#             out_features = self.model.encoder.num_features_global[i + 1]
#             print(f"  Layer {i}:")
#             print(f"    FeaStConv: Input: ({num_vertices}, {in_features}) -> Output: ({num_vertices}, {out_features})")
#             # if self.model.encoder.batch_norm:
#             #     print(f"    BatchNorm")
#             print(f"    LeakyReLU activation")
#             print(f"    Downsampling: ({num_vertices}, {out_features}) -> ({self.model.encoder.D[i].shape[0]}, {out_features})")
#             num_vertices = self.model.encoder.D[i].shape[0]

#         print("\nLocal Encoder:")
#         num_vertices = x.shape[0] // batch_size  # Reset to initial number of vertices
#         for i in range(self.model.encoder.n_layers):
#             in_features = self.model.encoder.num_features_local[i]
#             out_features = self.model.encoder.num_features_local[i + 1]
#             print(f"  Layer {i}:")
#             print(f"    FeaStConv: Input: ({num_vertices}, {in_features}) -> Output: ({num_vertices}, {out_features})")
#             # if self.model.encoder.batch_norm:
#             #     print(f"    BatchNorm")
#             print(f"    LeakyReLU activation")

#         print(f"\nLocal Linear Layer: Input: ({batch_size}, {self.model.encoder.D[0].shape[1] * self.model.encoder.num_features_local[-1]}) -> Output: ({batch_size}, {self.model.encoder.z_length})")

#     def _print_latent_space_summary(self):
#         print("\nLatent Space:")
#         print("-------------")
#         z_length = self.model.encoder.z_length
#         print(f"Dimension: {z_length}")

#     def _print_decoder_summary(self, sample_batch: Batch):
#         print("\nDecoder:")
#         print("--------")
#         batch_size = sample_batch.num_graphs
#         final_num_vertices = self.model.decoder.U[0].shape[0]

#         print("Global Decoder:")
#         for i in range(self.model.decoder.n_layers):
#             in_features = self.model.decoder.num_features_global[-1-i]
#             out_features = self.model.decoder.num_features_global[-2-i]
            
#             # Calculate the number of vertices before and after upsampling
#             if i == 0:
#                 pre_upsample_vertices = self.model.decoder.U[-1].shape[1]
#             else:
#                 pre_upsample_vertices = self.model.decoder.U[-i].shape[0]
            
#             post_upsample_vertices = self.model.decoder.U[-1-i].shape[0]

#             print(f"  Layer {i}:")
#             print(f"    Upsampling: ({pre_upsample_vertices}, {in_features}) -> ({post_upsample_vertices}, {in_features})")
#             print(f"    FeaStConv: Input: ({post_upsample_vertices}, {in_features}) -> Output: ({post_upsample_vertices}, {out_features})")
#             if i < self.model.decoder.n_layers - 1:
#                 # if self.model.decoder.batch_norm:
#                 #     print(f"    BatchNorm")
#                 print(f"    LeakyReLU activation")

#         print("\nLocal Decoder:")
#         z_length = self.model.decoder.z_length
#         initial_local_features = self.model.decoder.num_features_local[-1]
#         print(f"  Linear Layer: Input: ({batch_size}, {z_length}) -> Output: ({batch_size}, {final_num_vertices * initial_local_features})")
#         print(f"  Reshape: ({batch_size}, {final_num_vertices * initial_local_features}) -> ({final_num_vertices}, {initial_local_features})")
        
#         for i in range(self.model.decoder.n_layers):
#             in_features = self.model.decoder.num_features_local[-1-i]
#             out_features = self.model.decoder.num_features_local[-2-i]
#             print(f"  Layer {i}:")
#             print(f"    FeaStConv: Input: ({final_num_vertices}, {in_features}) -> Output: ({final_num_vertices}, {out_features})")
#             if i < self.model.decoder.n_layers - 1:
#                 # if self.model.decoder.batch_norm:
#                 #     print(f"    BatchNorm")
#                 print(f"    LeakyReLU activation")

#         print("\nFinal Layer:")
#         in_features = self.model.decoder.num_features_global[0] + self.model.decoder.num_features_local[0]
#         out_features = sample_batch.x.shape[1]
#         print(f"  Merge global and local features: Input: ({final_num_vertices}, {in_features}) -> Output: ({final_num_vertices}, {out_features})")

#     def _count_parameters(self):
#         return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

# # Add this line at the end of your model.py file
# FMGenModelSummary = ModelSummary


