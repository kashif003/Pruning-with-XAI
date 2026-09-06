# this file computes a CAUSAL importance score for every conv channel:
# each channel is zero-ablated via a forward hook on its Conv2d layer,
# and the resulting drop in the model's top-1 confidence is measured,
# over the evaluation image set.
# Output JSON shape matches the GMAR-family scripts: {layer_name: [channel_scores...]}
#
# NOTE: ResNet50 has 26,560 output channels total. This script ablates
# ONE channel at a time and re-runs the full image set for each --
# that's 26,560 x len(images) forward passes. Expect this to take
# substantially longer than the ViT head-ablation version (144 heads).
# Consider restricting to a subset of layers/channels, or a smaller
# image sample, if runtime becomes impractical.

from collections import defaultdict
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.stats import ttest_rel
from tqdm import tqdm
from torchvision import transforms

from ..config import GPU_NAME, MODEL_NAME
from ..utils import get_jpeg_images
from .cnn import Custom_model

NAME = MODEL_NAME
DEVICE = GPU_NAME
BATCH_SIZE = 32

custom_model = Custom_model(device=DEVICE, name=NAME)
model = custom_model.model
images = get_jpeg_images("src.get_score.imagenet_val_1000")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

# Build the list of (layer_name, num_channels) to ablate, in forward order
conv_layers = [
    (name, module.out_channels)
    for name, module in model.named_modules()
    if isinstance(module, nn.Conv2d)
]


def load_batch(image_paths):
    """Loads and stacks a batch of images into one input tensor."""
    tensors = [preprocess(Image.open(p).convert("RGB")).unsqueeze(0) for p in image_paths]
    return torch.cat(tensors, dim=0)


def get_target_proba(logits, target_class):
    probs = torch.softmax(logits, dim=-1)
    batch_idx = torch.arange(logits.size(0), device=logits.device)
    return probs[batch_idx, target_class]


@torch.no_grad()
def compute_scores(targets, layer_name=None, channel_idx=None):
    """Runs the full image set through the model and returns a (N,) array
    of scores (probability assigned to `targets`), optionally with one
    conv channel zero-ablated."""
    all_scores = []
    for start in range(0, len(images), BATCH_SIZE):
        batch_paths = images[start:start + BATCH_SIZE]
        pixel_values = load_batch(batch_paths).to(DEVICE)
        logits = custom_model.causal_forward_pass(pixel_values, layer_name=layer_name, channel_idx=channel_idx)
        batch_targets = targets[start:start + len(batch_paths)]
        proba = get_target_proba(logits, batch_targets)
        all_scores.append(proba.cpu().numpy())
    return np.concatenate(all_scores)


# --- Step 1: baseline pass -- also fixes the "pseudo-label" per image ---
print("[INFO] Computing baseline scores and target classes...")
targets_list = []
with torch.no_grad():
    for start in range(0, len(images), BATCH_SIZE):
        batch_paths = images[start:start + BATCH_SIZE]
        pixel_values = load_batch(batch_paths).to(DEVICE)
        logits = custom_model.causal_forward_pass(pixel_values, layer_name=None, channel_idx=None)
        targets_list.append(logits.argmax(dim=-1))
targets = torch.cat(targets_list, dim=0).to(DEVICE)
baseline_scores = compute_scores(targets, layer_name=None, channel_idx=None)

# --- Diagnostic: confirm the hook actually changes the output ---
first_layer_name = conv_layers[0][0]
pixel_values = load_batch(images[:BATCH_SIZE]).to(DEVICE)
logits_masked = custom_model.causal_forward_pass(pixel_values, layer_name=first_layer_name, channel_idx=0)
logits_base = custom_model.causal_forward_pass(pixel_values, layer_name=None, channel_idx=None)
assert not torch.allclose(logits_masked, logits_base), "Hook is not modifying the forward pass!"
print("[INFO] Ablation hook verified working.")

# --- Step 2: ablate each channel, one at a time, over all images ---
global_causal_scores = defaultdict(list)
global_pvalues = defaultdict(list)

for layer_name, num_channels in tqdm(conv_layers, desc="layers"):
    for channel_idx in range(num_channels):
        masked_scores = compute_scores(targets, layer_name=layer_name, channel_idx=channel_idx)

        # Importance = how much confidence DROPS when the channel is removed.
        # High positive = important (keep). Near-zero/negative = safe to prune.
        importance = float((baseline_scores - masked_scores).mean())
        pvalue = float(ttest_rel(baseline_scores, masked_scores).pvalue)

        global_causal_scores[layer_name].append(importance)
        global_pvalues[layer_name].append(pvalue)

# --- Step 3: save, same JSON shape as the GMAR-family scripts ---
print("[INFO] Converting tensors to native Python formats for JSON serialization...")
json_ready_scores = {str(k): v for k, v in global_causal_scores.items()}
with open("scores/Causal_score.json", "w") as file:
    json.dump(json_ready_scores, file, indent=4)

json_ready_pvalues = {str(k): v for k, v in global_pvalues.items()}
with open("scores/Causal_pvalues.json", "w") as file:
    json.dump(json_ready_pvalues, file, indent=4)