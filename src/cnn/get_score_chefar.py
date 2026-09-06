# get_score_chefer_cnn.py
# CNN adaptation of Chefer et al. (2021) style importance scoring.
#
# NOTE ON THE SUBSTITUTION: Chefer's method for ViT combines the
# attention matrix (softmax weights) with its gradient to form an LRP-
# style relevance term. A CNN conv layer has no attention matrix -- the
# closest structural analog is the layer's own activation map (this is
# exactly the substitution Grad-CAM makes vs. attention-rollout methods).
# So here, `activations` plays the role `attentions` played in the ViT
# version; the rest of the formula (relevance = normalize(act * grad),
# combined = grad * relevance, ReLU, sum) is unchanged.

from .cnn import Custom_model
from ..utils import get_jpeg_images
import torch
import numpy as np
import json
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--saving_dir", type=str, required=True)
parser.add_argument("--device", type=str, required=True)
args = parser.parse_args()

MODEL_NAME = args.model_name
GPU_NAME = args.device
SAVING_DIR = args.saving_dir

saving_dir_name = f"scores/{SAVING_DIR}/chefer_score.json"

custom_model = Custom_model(name=MODEL_NAME, device=GPU_NAME)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

images = get_jpeg_images("imagenet_val_1000")


def compute_lrp_relevance(activations, gradients):
    """
    Computes LRP-style relevance per layer using Deep Taylor Decomposition.
    R = (activation * gradient) / sum(|activation * gradient|)
    Returns dict of relevance arrays, one per layer.
    """
    relevances = {}
    for layer_name in activations:
        act = activations[layer_name].detach().cpu().numpy()
        grad = gradients[layer_name].detach().cpu().numpy()

        # LRP: element-wise product of activation and gradient
        lrp = act * grad  # [Batch, Channels, H, W]

        # Normalize to maintain conservation rule
        denom = np.sum(np.abs(lrp), axis=(-2, -1), keepdims=True) + 1e-8
        lrp = lrp / denom

        relevances[layer_name] = lrp

    return relevances


def compute_chefer_channel_scores(activations, gradients):
    """
    Chefer-style channel importance score per layer:
    1. Compute LRP relevance R = normalize(activation * grad)
    2. Multiply gradient with relevance: grad x R
    3. ReLU -> keep positive values only
    4. Sum across spatial dims, mean across batch -> scalar per channel
    """
    relevances = compute_lrp_relevance(activations, gradients)
    channel_scores_per_layer = {}

    for layer_name, grad in gradients.items():
        grad = grad.detach().cpu().numpy()
        rel = relevances[layer_name]

        # Step 1 - gradient x LRP relevance: [Batch, Channels, H, W]
        combined = grad * rel

        # Step 2 - ReLU: keep only positive values
        positive = np.maximum(combined, 0)

        # Step 3 - sum across spatial dims -> [Batch, Channels]
        channel_scores = np.sum(positive, axis=(-2, -1))

        # Step 4 - mean across batch -> [Channels]
        mean_scores = np.mean(channel_scores, axis=0)

        channel_scores_per_layer[layer_name] = mean_scores

    return channel_scores_per_layer


def accumulate_chefer_scores(layer_scores, final_score_dict=None):
    """Accumulates channel scores across images."""
    if final_score_dict is None:
        final_score_dict = {}
    for layer_name, scores in layer_scores.items():
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

    layer_scores = compute_chefer_channel_scores(activations, gradients)
    global_pruning_scores = accumulate_chefer_scores(layer_scores, global_pruning_scores)

print("[INFO] Total images processed:", idx + 1)
print("[INFO] Saving scores...")

json_ready_scores = {
    str(layer_name): scores.tolist()
    for layer_name, scores in global_pruning_scores.items()
}

with open(saving_dir_name, "w") as f:
    json.dump(json_ready_scores, f, indent=4)

print("[INFO] Successfully saved Chefer scores to chefer_score.json!")