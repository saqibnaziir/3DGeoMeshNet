"""
3DGeoMeshNet model with selectable ablation variants.
- default: full model (global + local paths, FeaStConv, residual + attention)
- global_only: global path only (FeaStConv)
- local_only: local path only (FeaStConv, residual)
- gat: full model with GATConv instead of FeaStConv
- gcn: full model with GCNConv instead of FeaStConv
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import FeaStConv, GATConv, GCNConv
from torch_geometric.data.batch import Batch

# Supported model types for ablation
MODEL_TYPES = ('default', 'global_only', 'local_only', 'gat', 'gcn')


def _get_conv_layer(conv_type, in_ch, out_ch):
    """Return conv layer and whether it uses keyword args (FeaStConv uses x=, edge_index=)."""
    if conv_type == 'feast':
        return FeaStConv(in_channels=in_ch, out_channels=out_ch), True
    if conv_type == 'gat':
        return GATConv(in_channels=in_ch, out_channels=out_ch), False
    if conv_type == 'gcn':
        return GCNConv(in_channels=in_ch, out_channels=out_ch), False
    raise ValueError(f"conv_type must be feast|gat|gcn, got {conv_type}")


def _path_mode_and_conv(model_type):
    """Map model_type to (path_mode, conv_type). path_mode: 'full'|'global_only'|'local_only'."""
    if model_type == 'default':
        return 'full', 'feast'
    if model_type == 'global_only':
        return 'global_only', 'feast'
    if model_type == 'local_only':
        return 'local_only', 'feast'
    if model_type == 'gat':
        return 'full', 'gat'
    if model_type == 'gcn':
        return 'full', 'gcn'
    raise ValueError(f"model_type must be one of {MODEL_TYPES}, got {model_type}")


class FMGenEncoder(torch.nn.Module):
    def __init__(self, config, A, D, model_type='default'):
        super(FMGenEncoder, self).__init__()
        self.A = [torch.tensor(a, requires_grad=False) for a in A]
        self.D = [torch.tensor(a, requires_grad=False) for a in D]
        self.model_type = model_type
        path_mode, conv_type = _path_mode_and_conv(model_type)
        self.path_mode = path_mode
        self.conv_type = conv_type
        self._conv_use_kwargs = conv_type == 'feast'

        self.batch_norm = config['batch_norm']
        self.n_layers = config['n_layers']
        self.z_length = config['z_length']
        self.num_features_global = config['num_features_global']
        self.num_features_local = config['num_features_local']

        # Global path (full or global_only)
        if path_mode in ('full', 'global_only'):
            self.encoder_convs_global = torch.nn.ModuleList([
                _get_conv_layer(conv_type, self.num_features_global[k], self.num_features_global[k + 1])[0]
                for k in range(self.n_layers)
            ])
            if path_mode == 'global_only':
                self.encoder_lin = torch.nn.Linear(self.num_features_global[-1], self.z_length)
            else:
                self.encoder_lin = torch.nn.Linear(
                    self.z_length + self.num_features_global[-1], self.z_length
                )
        else:
            self.encoder_convs_global = None
            self.encoder_lin = None

        # Local path (full or local_only)
        if path_mode in ('full', 'local_only'):
            self.encoder_convs_local = torch.nn.ModuleList([
                _get_conv_layer(conv_type, self.num_features_local[k], self.num_features_local[k + 1])[0]
                for k in range(self.n_layers)
            ])
            self.local_residual_projections = torch.nn.ModuleList([
                torch.nn.Linear(self.num_features_local[k], self.num_features_local[k + 1])
                if self.num_features_local[k] != self.num_features_local[k + 1]
                else torch.nn.Identity()
                for k in range(self.n_layers)
            ])
            self.encoder_local_lin = torch.nn.Linear(
                self.D[0].shape[1] * self.num_features_local[-1], self.z_length
            )
        else:
            self.encoder_convs_local = None
            self.local_residual_projections = None
            self.encoder_local_lin = None

        self.reset_parameter()

    def reset_parameter(self):
        if self.encoder_lin is not None:
            torch.nn.init.normal_(self.encoder_lin.weight, 0, 0.1)
        if self.encoder_local_lin is not None:
            torch.nn.init.normal_(self.encoder_local_lin.weight, 0, 0.1)

    def _conv_forward(self, conv, x, edge_index):
        if self._conv_use_kwargs:
            return conv(x=x, edge_index=edge_index)
        return conv(x, edge_index)

    def forward(self, x, batch_size):
        self.A = [a.to(x.device) for a in self.A]
        self.D = [d.to(x.device) for d in self.D]

        if self.path_mode == 'global_only':
            x_global = x
            for i in range(self.n_layers):
                x_global = self._conv_forward(self.encoder_convs_global[i], x_global, self.A[i])
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
            z = self.encoder_lin(x_global)
            z = F.leaky_relu(z)
            return z

        if self.path_mode == 'local_only':
            x_local = x
            for i in range(self.n_layers):
                identity = x_local
                x_local = self._conv_forward(self.encoder_convs_local[i], x_local, self.A[0])
                x_local = F.leaky_relu(x_local)
                x_local = x_local + self.local_residual_projections[i](identity)
            x_local = x_local.reshape(batch_size, -1)
            z = self.encoder_local_lin(x_local)
            z = F.leaky_relu(z)
            return z

        # path_mode == 'full'
        x_global = x
        x_local = x
        # Global path
        for i in range(self.n_layers):
            x_global = self._conv_forward(self.encoder_convs_global[i], x_global, self.A[i])
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
        # Local path with residual
        for i in range(self.n_layers):
            identity = x_local
            x_local = self._conv_forward(self.encoder_convs_local[i], x_local, self.A[0])
            x_local = F.leaky_relu(x_local)
            x_local = x_local + self.local_residual_projections[i](identity)
        x_local = x_local.reshape(batch_size, -1)
        x_local = self.encoder_local_lin(x_local)
        x_local = F.leaky_relu(x_local)
        z = torch.concat((x_global, x_local), dim=1)
        z = self.encoder_lin(z)
        return z


class FMGenDecoder(torch.nn.Module):
    def __init__(self, config, A, U, model_type='default'):
        super(FMGenDecoder, self).__init__()
        self.A = [torch.tensor(a, requires_grad=False) for a in A]
        self.U = [torch.tensor(u, requires_grad=False) for u in U]
        self.model_type = model_type
        path_mode, conv_type = _path_mode_and_conv(model_type)
        self.path_mode = path_mode
        self.conv_type = conv_type
        self._conv_use_kwargs = conv_type == 'feast'

        self.batch_norm = config['batch_norm']
        self.n_layers = config['n_layers']
        self.z_length = config['z_length']
        self.num_features_global = config['num_features_global']
        self.num_features_local = config['num_features_local']

        if path_mode in ('full', 'global_only'):
            self.decoder_convs_global = torch.nn.ModuleList([
                _get_conv_layer(conv_type,
                    self.num_features_global[-1 - k], self.num_features_global[-2 - k])[0]
                for k in range(self.n_layers)
            ])
            self.decoder_lin = torch.nn.Linear(self.z_length, self.num_features_global[-1])
        else:
            self.decoder_convs_global = None
            self.decoder_lin = None

        if path_mode in ('full', 'local_only'):
            self.decoder_convs_local = torch.nn.ModuleList([
                _get_conv_layer(conv_type,
                    self.num_features_local[-1 - k], self.num_features_local[-2 - k])[0]
                for k in range(self.n_layers)
            ])
            self.local_residual_projections = torch.nn.ModuleList([
                torch.nn.Linear(self.num_features_local[-1 - k], self.num_features_local[-2 - k])
                if self.num_features_local[-1 - k] != self.num_features_local[-2 - k]
                else torch.nn.Identity()
                for k in range(self.n_layers)
            ])
            self.decoder_local_lin = torch.nn.Linear(
                self.z_length, self.num_features_local[-1] * self.U[0].shape[0]
            )
        else:
            self.decoder_convs_local = None
            self.local_residual_projections = None
            self.decoder_local_lin = None

        if path_mode == 'full':
            self.decoder_lin = torch.nn.Linear(
                self.z_length, self.z_length + self.num_features_global[-1]
            )
            self.decoder_local_lin = torch.nn.Linear(
                self.z_length, self.num_features_local[-1] * self.U[0].shape[0]
            )
            self.attention = torch.nn.Sequential(
                torch.nn.Linear(self.num_features_global[0] + self.num_features_local[0], 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 2),
                torch.nn.Softmax(dim=-1)
            )
        else:
            self.attention = None

        self.reset_parameter()

    def reset_parameter(self):
        if self.decoder_lin is not None:
            torch.nn.init.normal_(self.decoder_lin.weight, 0, 0.1)
        if self.decoder_local_lin is not None:
            torch.nn.init.normal_(self.decoder_local_lin.weight, 0, 0.1)
        if self.attention is not None:
            for layer in self.attention:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.normal_(layer.weight, 0, 0.1)

    def _conv_forward(self, conv, x, edge_index):
        if self._conv_use_kwargs:
            return conv(x=x, edge_index=edge_index)
        return conv(x, edge_index)

    def forward(self, z, batch_size):
        self.A = [a.to(z.device) for a in self.A]
        self.U = [u.to(z.device) for u in self.U]

        if self.path_mode == 'global_only':
            x_global = self.decoder_lin(z)
            x_global = torch.unsqueeze(x_global, dim=1).repeat(1, self.U[-1].shape[1], 1)
            for i in range(self.n_layers):
                x_global = x_global.reshape(batch_size, -1, self.num_features_global[-1 - i])
                y = torch.zeros(batch_size, self.U[-1 - i].shape[0], x_global.shape[2], device=x_global.device)
                for j in range(batch_size):
                    y[j] = torch.mm(self.U[-1 - i], x_global[j])
                x_global = y
                x_global = x_global.reshape(-1, self.num_features_global[-1 - i])
                x_global = self._conv_forward(self.decoder_convs_global[i], x_global, self.A[-2 - i])
                if i < self.n_layers - 1:
                    x_global = F.leaky_relu(x_global)
            x = x_global.reshape(-1, self.num_features_global[0])
            return x

        if self.path_mode == 'local_only':
            x_local = self.decoder_local_lin(z)
            x_local = x_local.reshape(-1, self.num_features_local[-1])
            for i in range(self.n_layers):
                identity = x_local
                x_local = self._conv_forward(self.decoder_convs_local[i], x_local, self.A[0])
                if i < self.n_layers - 1:
                    x_local = F.leaky_relu(x_local)
                x_local = x_local + self.local_residual_projections[i](identity)
            return x_local

        # path_mode == 'full'
        x = self.decoder_lin(z)
        x_global = x[:, :self.num_features_global[-1]]
        x_local = x[:, self.num_features_global[-1]:]
        x_global = torch.unsqueeze(x_global, dim=1).repeat(1, self.U[-1].shape[1], 1)
        for i in range(self.n_layers):
            x_global = x_global.reshape(batch_size, -1, self.num_features_global[-1 - i])
            y = torch.zeros(batch_size, self.U[-1 - i].shape[0], x_global.shape[2], device=x_global.device)
            for j in range(batch_size):
                y[j] = torch.mm(self.U[-1 - i], x_global[j])
            x_global = y
            x_global = x_global.reshape(-1, self.num_features_global[-1 - i])
            x_global = self._conv_forward(self.decoder_convs_global[i], x_global, self.A[-2 - i])
            if i < self.n_layers - 1:
                x_global = F.leaky_relu(x_global)
        x_global = x_global.reshape(-1, self.num_features_global[0])
        x_local = self.decoder_local_lin(x_local)
        x_local = x_local.reshape(-1, self.num_features_local[-1])
        for i in range(self.n_layers):
            identity = x_local
            x_local = self._conv_forward(self.decoder_convs_local[i], x_local, self.A[0])
            if i < self.n_layers - 1:
                x_local = F.leaky_relu(x_local)
            x_local = x_local + self.local_residual_projections[i](identity)
        concat_features = torch.cat([x_global, x_local], dim=-1)
        weights = self.attention(concat_features)
        x = weights[:, 0:1] * x_global + weights[:, 1:2] * x_local
        return x


class FMGenModel(torch.nn.Module):
    """Single model that delegates to the encoder/decoder variant based on config['model_type']."""

    def __init__(self, config, A, D, U, model_type=None):
        super(FMGenModel, self).__init__()
        if model_type is None:
            model_type = config.get('model_type', 'default')
        if model_type not in MODEL_TYPES:
            raise ValueError(f"model_type must be one of {MODEL_TYPES}, got {model_type}")
        self.model_type = model_type
        self.encoder = FMGenEncoder(config, A, D, model_type=model_type)
        self.decoder = FMGenDecoder(config, A, U, model_type=model_type)

    def forward(self, batch: Batch):
        batch_size = batch.num_graphs
        z = self.encoder(batch.x, batch_size)
        x_reconstructed = self.decoder(z, batch_size)
        return x_reconstructed, z


class ModelSummary:
    """Prints architecture summary; supports all ablation variants."""

    def __init__(self, model, config):
        self.model = model
        self.config = config

    def print_summary(self, sample_batch: Batch):
        mt = getattr(self.model, 'model_type', 'default')
        print(f"FMGenModel Architecture Summary (model_type={mt}):")
        print("================================")
        self._print_encoder_summary(sample_batch)
        self._print_latent_space_summary()
        self._print_decoder_summary(sample_batch)
        print("\nTotal parameters:", self._count_parameters())

    def _print_encoder_summary(self, sample_batch: Batch):
        print("\nEncoder:")
        print("--------")
        enc = self.model.encoder
        x = sample_batch.x
        batch_size = sample_batch.num_graphs
        num_vertices = x.shape[0] // batch_size
        conv_name = getattr(enc, 'conv_type', 'feast').upper()
        if enc.encoder_convs_global is not None:
            print("Global Encoder:")
            for i in range(enc.n_layers):
                in_f = enc.num_features_global[i]
                out_f = enc.num_features_global[i + 1]
                print(f"  Layer {i}: {conv_name}Conv ({num_vertices}, {in_f}) -> ({num_vertices}, {out_f}), LeakyReLU, Downsampling")
                num_vertices = enc.D[i].shape[0]
            print(f"  Mean pool -> Linear -> z (dim {enc.z_length})")
        if enc.encoder_convs_local is not None:
            num_vertices = x.shape[0] // batch_size
            print("Local Encoder:")
            for i in range(enc.n_layers):
                in_f = enc.num_features_local[i]
                out_f = enc.num_features_local[i + 1]
                print(f"  Layer {i}: {conv_name}Conv + residual ({num_vertices}, {in_f}) -> ({num_vertices}, {out_f}), LeakyReLU")
            print(f"  Linear -> z (dim {enc.z_length})")
        if enc.path_mode == 'full':
            print("  Concat(global, local) -> Linear -> z")

    def _print_latent_space_summary(self):
        print("\nLatent Space:")
        print("-------------")
        print(f"Dimension: {self.model.encoder.z_length}")

    def _print_decoder_summary(self, sample_batch: Batch):
        print("\nDecoder:")
        print("--------")
        dec = self.model.decoder
        batch_size = sample_batch.num_graphs
        final_num_vertices = dec.U[0].shape[0]
        conv_name = getattr(dec, 'conv_type', 'feast').upper()
        if dec.decoder_convs_global is not None:
            print("Global Decoder:")
            for i in range(dec.n_layers):
                in_f = dec.num_features_global[-1 - i]
                out_f = dec.num_features_global[-2 - i]
                print(f"  Layer {i}: Upsampling, {conv_name}Conv -> ({final_num_vertices}, {out_f}), LeakyReLU")
        if dec.decoder_convs_local is not None:
            print("Local Decoder:")
            print(f"  Linear -> Reshape -> {dec.n_layers} x {conv_name}Conv+residual")
        if dec.attention is not None:
            print("  Merge: attention(concat(global, local)) -> weighted sum")

    def _count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


FMGenModelSummary = ModelSummary
