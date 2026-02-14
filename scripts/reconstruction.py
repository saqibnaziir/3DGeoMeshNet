"""
Batch mesh reconstruction: process all PLY files in an input folder and save
reconstructed meshes to an output folder. Entry point for 3DGeoMeshNet inference.
"""
import os
import sys
import argparse
import torch
from glob import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.reconstruction import facial_mesh_reconstruction
from config.config import read_config
from utils.funcs import load_generator


def process_ply_files(input_folder, output_folder, config, generator, lambda_reg, verbose):
    ply_files = glob(os.path.join(input_folder, '*.ply'))
    for in_file in ply_files:
        filename = os.path.basename(in_file)
        out_file = os.path.join(output_folder, f"completed_{filename}")
        facial_mesh_reconstruction(in_file, out_file, config, generator, lambda_reg, verbose)
        print(f"Processed: {filename} -> {out_file}")


def main():
    parser = argparse.ArgumentParser(description='Mesh reconstruction (batch)')
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--lambda_reg", type=float, default=0.1)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--rr", type=bool, default=False)
    parser.add_argument("--dis_percent", type=float, default=None)
    args = parser.parse_args()
    config = read_config(args.config_file)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = load_generator(config).to(device)
    os.makedirs(args.output_folder, exist_ok=True)
    process_ply_files(args.input_folder, args.output_folder, config, generator, args.lambda_reg, args.verbose)


if __name__ == "__main__":
    main()
