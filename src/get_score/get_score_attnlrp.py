# this file will be used to get the score from AttnLRP.
#
# NOTE: run this as its own standalone script/process. Do NOT import and
# call gmar/gmarpp/legrad/chefer scoring in the same process after this,
# since lxt's monkey_patch on modeling_vit is permanent for the process
# once applied (see the docstring above attnlrp_forward_pass in vit.py).

from collections import defaultdict
import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

from ..utils import get_jpeg_images, get_img_tensor
from ..vit import Custom_model

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

if SAVING_DIR == "tiny":
    saving_dir_name = "scores/tiny/attnLRP_score.json"
elif SAVING_DIR == "small":
    saving_dir_name = "scores/small/attnLRP_score.json"
else:
    saving_dir_name = f"scores/{SAVING_DIR}/attnLRP_score.json"

custom_model = Custom_model(GPU_NAME, MODEL_NAME )
processor = AutoImageProcessor.from_pretrained("google/vit-large-patch16-384")
images = get_jpeg_images("src/imagenet_val_1000")


def accumulate_attnlrp_scores(attnlrp_relevances, final_score_dict=None):
    """
    Accumulates pre-calculated 1D layer-wise AttnLRP head relevance scores
    across the evaluation dataset.
    """
    if final_score_dict is None:
        final_score_dict = {}

    for layer_idx, relevance_scores in attnlrp_relevances.items():
        if layer_idx in final_score_dict:
            final_score_dict[layer_idx] += relevance_scores
        else:
            final_score_dict[layer_idx] = np.copy(relevance_scores)

    return final_score_dict


# --- Main Loop ---
global_pruning_scores = {}

for idx, image in tqdm(enumerate(images)):
    torch.cuda.empty_cache()
    img_tensor = get_img_tensor(processor, image)
    output, attention, attnlrp_relevance = custom_model.attnlrp_forward_pass(
        img_tensor.pixel_values.to("cuda:7")
    )

    global_pruning_scores = accumulate_attnlrp_scores(attnlrp_relevance, global_pruning_scores)


print("[INFO] Converting tensors to native Python formats for JSON serialization...")

json_ready_scores = {
    str(layer_idx): scores.tolist()
    for layer_idx, scores in global_pruning_scores.items()
}

with open(saving_dir_name, "w") as file:
    json.dump(json_ready_scores, file, indent=4)