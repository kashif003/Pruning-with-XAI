# this file will be used to get the score from the LeGrad for CNN (ResNet) models.

import json
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from ..utils import get_jpeg_images
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

saving_dir_name = f"scores/{SAVING_DIR}/legrad_score.json"

custom_model = Custom_model(GPU_NAME, MODEL_NAME)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

images = get_jpeg_images("imagenet_val_1000")


def compute_legrad_scores(gradients):
    """
    LeGrad: ReLU-filter gradients (keep only positive), then sum over
    spatial dims (H, W), averaged over batch -> one score per channel.
    """
    legrad_scores = {}
    for layer_name, grad in gradients.items():
        # grad shape: [Batch, Channels, H, W]
        positive_grad = torch.clamp(grad, min=0)
        channel_scores = positive_grad.sum(dim=[-2, -1])   # -> [Batch, Channels]
        mean_channel_scores = channel_scores.mean(dim=0)   # -> [Channels]
        legrad_scores[layer_name] = mean_channel_scores.detach().cpu().numpy()
    return legrad_scores


def accumulate_legrad_scores(legrad_scores, final_score_dict=None):
    if final_score_dict is None:
        final_score_dict = {}
    for layer_name, scores in legrad_scores.items():
        if layer_name in final_score_dict:
            final_score_dict[layer_name] += scores
        else:
            final_score_dict[layer_name] = np.copy(scores)
    return final_score_dict


# --- Main Loop ---
global_pruning_scores = {}

for idx, image in tqdm(enumerate(images)):
    torch.cuda.empty_cache()

    img_tensor = preprocess(Image.open(image).convert("RGB")).unsqueeze(0)
    output, activations, gradients = custom_model.forward_pass(img_tensor.to(GPU_NAME))

    legrad_scores = compute_legrad_scores(gradients)
    global_pruning_scores = accumulate_legrad_scores(legrad_scores, global_pruning_scores)


print("[INFO] Converting tensors to native Python formats for JSON serialization...")

json_ready_scores = {
    str(layer_name): scores.tolist()
    for layer_name, scores in global_pruning_scores.items()
}

with open(saving_dir_name, "w") as file:
    json.dump(json_ready_scores, file, indent=4)

print("[INFO] Successfully saved LeGrad scores to legrad_score.json!")