import argparse
import json
from src.train import train

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--mode', type=str, help='[train/eval/encode]')
parser.add_argument('--inputs', type=str, help='for "encode" mode')


args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = json.load(f)

if args.mode == 'train':
    train(cfg)