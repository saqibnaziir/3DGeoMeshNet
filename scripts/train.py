import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import random
from config.config import read_config
from utils.my_dataset import MyDataset
from utils.models import FMGenModel, FMGenModelSummary
import argparse
from torch.nn import Conv1d, Parameter, ParameterList
from trimesh import Trimesh, load_mesh
from utils.funcs import get_mesh_matrices, spherical_regularization_loss
import warnings
# from pytorch3d.loss import chamfer_distance  # not used in training (L1 + MSE + reg only)
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

#from utils.funcs import save_curvatures_for_dataset

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def load_model(config, load_state_dict):
    pA, pD, pU = get_mesh_matrices(config)
    model = FMGenModel(config, pA, pD, pU)
    if load_state_dict:
        model.encoder.load_state_dict(torch.load(os.path.join(config['checkpoint_dir'], 'checkpoint_encoder.pt')))
        model.decoder.load_state_dict(torch.load(os.path.join(config['checkpoint_dir'], 'checkpoint_decoder.pt')))
    return model

def train_epoch(model, train_loader, optimizer, device, size, epoch, writer, lambda_reg=1.0):
    model.train()
    total_loss_l1 = 0
    total_loss_mse = 0
    total_loss_reg = 0

    for batch in tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        out, z = model(batch)

        loss_mse = F.mse_loss(out, batch.x)
        ## from FaceCom
        loss_l1 = F.l1_loss(out, batch.x)# new L1 loss
        
        loss_reg = spherical_regularization_loss(z)

        total_loss_mse += batch.num_graphs * loss_mse.item()
        total_loss_l1 += batch.num_graphs * loss_l1.item()
        total_loss_reg += batch.num_graphs * loss_reg.item()

        loss = loss_l1 + loss_mse + lambda_reg * loss_reg
        # loss = loss_mse 

        loss.backward()
        optimizer.step()

    avg_loss_l1 = total_loss_l1 / size
    avg_loss_mse = total_loss_mse / size
    avg_loss_reg = total_loss_reg / size

    # Log train losses to TensorBoard
    writer.add_scalar('Loss/train/L1', avg_loss_l1, epoch)
    writer.add_scalar('Loss/train/MSE', avg_loss_mse, epoch)
    writer.add_scalar('Loss/train/REG', avg_loss_reg, epoch)
    
    return avg_loss_l1, avg_loss_mse, avg_loss_reg
    # return  avg_loss_mse

def test_epoch(model, test_loader, device, size, epoch, writer):
    model.eval()
    total_loss_l1 = 0
    total_loss_mse = 0
    total_loss_reg = 0

    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = batch.to(device)

            out, z = model(batch)

            loss_mse = F.mse_loss(out, batch.x)
            loss_l1 = F.l1_loss(out, batch.x)
            loss_reg = spherical_regularization_loss(z)

            total_loss_mse += batch.num_graphs * loss_mse.item()
            total_loss_l1 += batch.num_graphs * loss_l1.item()
            total_loss_reg += batch.num_graphs * loss_reg.item()

    avg_loss_l1 = total_loss_l1 / size
    avg_loss_mse = total_loss_mse / size
    avg_loss_reg = total_loss_reg / size

    # Log test losses to TensorBoard
    writer.add_scalar('Loss/test/L1', avg_loss_l1, epoch)
    writer.add_scalar('Loss/test/MSE', avg_loss_mse, epoch)
    writer.add_scalar('Loss/test/REG', avg_loss_reg, epoch)
    
    return avg_loss_l1, avg_loss_mse, avg_loss_reg
    # return  avg_loss_mse

def train(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(config['checkpoint_dir']):
        os.makedirs(config['checkpoint_dir'])
    # Create a unique name for this training run
    
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{config['experiment_name']}_{current_time}"
    # Initialize TensorBoard writer with the unique run name
    writer = SummaryWriter(log_dir=os.path.join(config['checkpoint_dir'], 'tensorboard_logs', run_name))

    print(f"TensorBoard logs will be saved in: {writer.log_dir}")
    # Initialize TensorBoard writer
    #writer = SummaryWriter(log_dir=os.path.join(config['checkpoint_dir'], 'tensorboard_logs'))

    print("loading datasets...")
    dataset_train = MyDataset(config, 'train')
    dataset_test = MyDataset(config, 'eval')
    train_loader = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'], pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(dataset_test, batch_size=config['batch_size'], shuffle=False,
                              num_workers=config['num_workers'], pin_memory=True, persistent_workers=True)

    print("loading model...")
    model = load_model(config, False)
    model.to(device)

    model_summary = FMGenModelSummary(model, config)
    sample_batch = next(iter(train_loader))
    sample_batch = sample_batch.to(device)
    model_summary.print_summary(sample_batch)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    print("start training...")
    best_loss_item = float('inf')

    lambda_reg = config['lambda_reg']

    for epoch in range(scheduler.last_epoch + 1, config['epoch'] + 1):
        print("Epoch", epoch, "  lr:", scheduler.get_lr())

        loss_l1, loss_mse, loss_reg = train_epoch(model, train_loader, optimizer, device, len(dataset_train), epoch, writer, lambda_reg)
        print("Train    loss:    L1:", loss_l1, "MSE:", loss_mse, "REG:", loss_reg)

        loss_l1_test, loss_mse_test, loss_reg_test = test_epoch(model, test_loader, device, len(dataset_test), epoch, writer)
        print("Test     loss:    L1:", loss_l1_test, "MSE:", loss_mse_test, "REG:", loss_reg_test)


        scheduler.step()

        # Log learning rate
        writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], epoch)

        if loss_l1_test + loss_mse_test + lambda_reg * loss_reg_test < best_loss_item:
            best_loss_item = loss_l1_test + loss_mse_test + lambda_reg * loss_reg_test
            torch.save(model.encoder.state_dict(), os.path.join(config['checkpoint_dir'], 'checkpoint_encoder.pt'))
            torch.save(model.decoder.state_dict(), os.path.join(config['checkpoint_dir'], 'checkpoint_decoder.pt'))
            torch.save(scheduler.state_dict(), os.path.join(config['checkpoint_dir'], 'scheduler.pt'))
            print("\nsave!\n\n")
    # Close the TensorBoard writer
    writer.close()

def main():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--experiment_name", type=str, default=None, 
                        help="Name for this training run")
    parser.add_argument("--model", type=str, default=None,
                        choices=['default', 'global_only', 'local_only', 'gat', 'gcn'],
                        help="Model ablation: default (full global+local FeaSt), global_only, local_only, gat, gcn")
    parser.add_argument("--seed", type=int, default=1, 
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # Ensure deterministic behavior in CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for PyTorch's DataLoader workers
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    # Force PyTorch to use deterministic algorithms when possible
    torch.use_deterministic_algorithms(True, warn_only=True)

    config = read_config(args.config_file)
    if args.experiment_name:
        config['experiment_name'] = args.experiment_name
    if args.model is not None:
        config['model_type'] = args.model
    config['seed'] = args.seed
    
    # Add seed to experiment name for clarity
    config['experiment_name'] = f"{config['experiment_name']}_seed{args.seed}"
    
    train(config)

if __name__ == "__main__":
    main()