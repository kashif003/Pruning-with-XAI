# this file computes a CAUSAL importance score for every attention head:
# each head is zero-ablated by hooking o_proj's input, and the resulting
# drop in the model's top-1 confidence is measured, over 1000 images.
# Output JSON shape matches get_score_GMARpp.py: {layer_idx: [head_scores...]}
# so it plugs straight into utils.get_layers_and_heads().

from collections import defaultdict
import json
import numpy as np
import torch
from scipy.stats import ttest_rel
from tqdm import tqdm
from transformers import AutoImageProcessor
from ..config  import  GPU_NAME, MODEL_NAME
from ..utils import get_jpeg_images, get_img_tensor
from ..vit import Custom_model

NAME = MODEL_NAME
DEVICE = GPU_NAME
BATCH_SIZE = 32

custom_model = Custom_model(device=DEVICE, name=NAME)
processor = AutoImageProcessor.from_pretrained(NAME)
images = get_jpeg_images("src.get_score.imagenet_val_1000")

model = custom_model.model
num_layers = model.config.num_hidden_layers
num_heads = model.config.num_attention_heads


def load_batch(image_paths):
    """Loads and stacks a batch of images into one pixel_values tensor."""
    tensors = [get_img_tensor(processor, p).pixel_values for p in image_paths]
    return torch.cat(tensors, dim=0)


def get_target_proba(logits, target_class):
    probs = torch.softmax(logits, dim=-1)
    batch_idx = torch.arange(logits.size(0), device=logits.device)
    return probs[batch_idx, target_class]


@torch.no_grad()
def compute_scores(targets, layer_idx=None, head_idx=None):
    """Runs the full image set through the model and returns a (N,) array
    of scores (probability assigned to `targets`), optionally with one
    attention head zero-ablated."""
    all_scores = []
    for start in range(0, len(images), BATCH_SIZE):
        batch_paths = images[start:start + BATCH_SIZE]
        pixel_values = load_batch(batch_paths).to(DEVICE)
        logits = custom_model.causal_forward_pass(pixel_values, layer_idx=layer_idx, head_idx=head_idx)
        batch_targets = targets[start:start + len(batch_paths)]
        proba = get_target_proba(logits, batch_targets)
        all_scores.append(proba.cpu().numpy())
    return np.concatenate(all_scores)


# --- Step 1: baseline pass — also fixes the "pseudo-label" per image ---
print("[INFO] Computing baseline scores and target classes...")
targets_list = []
with torch.no_grad():
    for start in range(0, len(images), BATCH_SIZE):
        batch_paths = images[start:start + BATCH_SIZE]
        pixel_values = load_batch(batch_paths).to(DEVICE)
        logits = custom_model.causal_forward_pass(pixel_values, layer_idx=None, head_idx=None)
        targets_list.append(logits.argmax(dim=-1))
targets = torch.cat(targets_list, dim=0).to(DEVICE)
baseline_scores = compute_scores(targets, layer_idx=None, head_idx=None)

# --- Diagnostic: confirm the hook actually changes the output ---
pixel_values = load_batch(images[:BATCH_SIZE]).to(DEVICE)
logits_masked = custom_model.causal_forward_pass(pixel_values, layer_idx=0, head_idx=0)
logits_base = custom_model.causal_forward_pass(pixel_values, layer_idx=None, head_idx=None)
assert not torch.allclose(logits_masked, logits_base), "Hook is not modifying the forward pass!"
print("[INFO] Ablation hook verified working.")

# --- Step 2: ablate each head, one at a time, over all images ---
global_causal_scores = defaultdict(list)
global_pvalues = defaultdict(list)

for layer_idx in tqdm(range(num_layers), desc="layers"):
    for head_idx in range(num_heads):
        masked_scores = compute_scores(targets, layer_idx=layer_idx, head_idx=head_idx)

        # Importance = how much confidence DROPS when the head is removed.
        # High positive = important (keep). Near-zero/negative = safe to prune.
        importance = float((baseline_scores - masked_scores).mean())
        pvalue = float(ttest_rel(baseline_scores, masked_scores).pvalue)

        global_causal_scores[layer_idx].append(importance)
        global_pvalues[layer_idx].append(pvalue)

# --- Step 3: save, same JSON shape as GMAR++'s output ---
print("[INFO] Converting tensors to native Python formats for JSON serialization...")
json_ready_scores = {str(k): v for k, v in global_causal_scores.items()}
with open("scores/Causal_score.json", "w") as file:
    json.dump(json_ready_scores, file, indent=4)

json_ready_pvalues = {str(k): v for k, v in global_pvalues.items()}
with open("scores/Causal_pvalues.json", "w") as file:
    json.dump(json_ready_pvalues, file, indent=4)