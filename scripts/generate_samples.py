"""
Generate random face meshes by sampling the latent space and decoding.
Entry point for 3DGeoMeshNet sample generation.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
from utils.reconstruction import generate_face_sample
from utils.funcs import load_generator
from config.config import read_config
from os.path import join
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='Generate random face samples from the model')
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--number", type=int, default=1)

    args = parser.parse_args()

    config = read_config(args.config_file)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = load_generator(config).to(device)

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    for i in tqdm(range(args.number)):
        generate_face_sample(join(args.out_dir, str(i + 1) + ".ply"), config, generator)


if __name__ == "__main__":
    main()
