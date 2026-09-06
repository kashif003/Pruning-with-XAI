# this file will be used to get the GMAR score for CNN (ResNet) models.

import json
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms

from ..utils import get_jpeg_images, get_img_tensor
from .cnn import Custom_model

# --
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--saving_dir", type=str, required=True)
parser.add_argument("--device", type=str, required=True)

args = parser.parse_args()

MODEL_NAME = args.model_name
GPU_NAME = args.device
SAVING_DIR = args.saving_dir

if SAVING_DIR == "resnet18":
    saving_dir_name = "scores/resnet18/GMAR_score.json"
elif SAVING_DIR == "resnet50":
    saving_dir_name = "scores/resnet50/GMAR_score.json"
else:
    saving_dir_name = f"scores/{SAVING_DIR}/GMAR_score.json"

'''
CNN_MODELS:
resnet50
resnet18
'''

NAME = MODEL_NAME
DEVICE = GPU_NAME if torch.cuda.is_available() else "cpu"

custom_model = Custom_model(device=DEVICE, name=NAME)

# Standard ImageNet preprocessing (torchvision models expect this,
# no AutoImageProcessor equivalent needed for ResNet)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

images = get_jpeg_images("imagenet_val_1000")


def compute_gmar_scores(gradients):
    """
    GMAR: absolute value of raw gradients, summed over spatial dims
    (H, W), averaged over batch -> one score per output channel,
    per conv layer.
    """
    gmar_scores = {}
    for layer_name, grad in gradients.items():
        # grad shape: [Batch, Channels, H, W]
        channel_scores = grad.abs().sum(dim=[-2, -1])   # -> [Batch, Channels]
        mean_channel_scores = channel_scores.mean(dim=0)  # -> [Channels]
        gmar_scores[layer_name] = mean_channel_scores.detach().cpu().numpy()
    return gmar_scores


def accumulate_gmar_scores(gmar_scores, final_score_dict=None):
    """
    Accumulates pre-calculated 1D layer-wise GMAR channel scores
    across the evaluation dataset.
    """
    if final_score_dict is None:
        final_score_dict = {}

    for layer_name, scores in gmar_scores.items():
        if layer_name in final_score_dict:
            final_score_dict[layer_name] += scores
        else:
            final_score_dict[layer_name] = np.copy(scores)

    return final_score_dict


# --- Main Loop ---
global_pruning_scores = {}

for idx, image in tqdm(enumerate(images)):
    torch.cuda.empty_cache()

    img_tensor = preprocess(image).unsqueeze(0)  # -> [1, 3, 224, 224]
    output, activations, gradients = custom_model.forward_pass(img_tensor.to(DEVICE))

    gmar_scores = compute_gmar_scores(gradients)
    global_pruning_scores = accumulate_gmar_scores(gmar_scores, global_pruning_scores)


print("[INFO] Converting tensors to native Python formats for JSON serialization...")

json_ready_scores = {
    str(layer_name): scores.tolist()
    for layer_name, scores in global_pruning_scores.items()
}

with open(saving_dir_name, "w") as file:
    json.dump(json_ready_scores, file, indent=4)