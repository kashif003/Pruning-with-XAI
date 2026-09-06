# this file will be used to get the magnitude of each conv filter/channel.

from collections import defaultdict
import json
import torch
import torch.nn as nn

from .cnn import Custom_model

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--saving_dir", type=str, required=True)
parser.add_argument("--device", type=str, required=True)
args = parser.parse_args()

MODEL_NAME = args.model_name
GPU_NAME = args.device
SAVING_DIR = args.saving_dir

saving_dir_name = f"scores/{SAVING_DIR}/mag_score.json"

model = Custom_model(GPU_NAME, MODEL_NAME).model
score = defaultdict(list)

# No forward/backward pass needed -- magnitude is purely a property of
# the trained weights themselves.
for layer_name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        # weight shape: [out_channels, in_channels, kH, kW]
        # one filter per output channel -- flatten everything except
        # the out_channels dim and take the L2 norm per filter.
        weight = module.weight.detach()
        num_filters = weight.shape[0]
        filter_magnitudes = weight.view(num_filters, -1).norm(p=2, dim=1)

        score[layer_name] = filter_magnitudes.cpu().tolist()

with open(saving_dir_name, "w") as f:
    json.dump(score, f, indent=4)