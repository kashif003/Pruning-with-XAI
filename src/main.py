from .utils import get_layers_and_heads, load_model_transformers
from transformers import AutoImageProcessor, AutoModelForImageClassification 
# from vit import Custom_model  # Commented out to keep imports clean since it's not used below
from thop import profile, clever_format
import torch
from .prune import FLOPS_and_PARAMS, prune_vit_heads
from .config import MODEL_NAME, GPU_NAME

import argparse
import os
from dotenv import load_dotenv
import wandb
from .get_score.get_baseline_accuracy import validate

load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_KEY")
wandb.login(key=WANDB_API_KEY)

print("[INFO] using model:", MODEL_NAME)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=int, required=True)
parser.add_argument("--score_path", type=str, required=True)
args = parser.parse_args()
cfg = args.config

print("[INFO] Using the following config:", cfg)
print("[INFO] Using the following score_path:", args.score_path)

# Extract just the filename from the path (e.g., "GMARPP_score.json" from "scores/small/GMARPP_score.json")
# This ensures your WandB run names are unique and descriptive!
score_filename = os.path.basename(args.score_path).replace(".json", "")

wandb.init(
    project="GMAR++ with pruning",
    # Dynamically name the run so you can tell them apart in the dashboard
    name=f"{score_filename}_tiny_{cfg}",
    config={
        "Pruning percentage": cfg,
        "Score File": score_filename
    }
)

layer_list, head_list = get_layers_and_heads(args.score_path, percentage=cfg)

device = torch.device(GPU_NAME if torch.cuda.is_available() else "cpu")
print("[INFO] Device:", device)

# Before pruning
dummy_input = torch.randn(1, 3, 224, 224).to(device)

processor, model = load_model_transformers(MODEL_NAME)
model.to(device)

bflops, bparams = FLOPS_and_PARAMS(model, dummy_input)

# Uncommented the actual pruning logic!
model_2 = prune_vit_heads(model, layer_indices=layer_list, heads_to_prune_list=head_list, device=device)

# After pruning
aflops, aparams = FLOPS_and_PARAMS(model_2, dummy_input)
print("BEFORE :", cfg)
print(bparams)
print("AFTER:", cfg)
print(aparams)

wandb.log({"FLOPS before Pruning": bflops, "PARAMS before Pruning": bparams})
wandb.log({"FLOPS after Pruning": aflops, "PARAMS after Pruning": aparams})

# Validation
accuracy = validate(model_2, device)

wandb.log({
    "Final Accuracy": accuracy["final_accuracy"],
    "harmonic_mean_accuracy": accuracy["harmonic_mean_accuracy"]
})

# per-class breakdown as a table
class_ids = list(range(len(accuracy["class_accuracy"])))
table = wandb.Table(
   columns=["class_id", "accuracy", "total_samples"],
   data=[[i, accuracy["class_accuracy"][i].item(), accuracy["class_total"][i].item()] for i in class_ids]
)
wandb.log({"class_wise_accuracy": table})

# Gracefully close the W&B run so the next loop starts clean
wandb.finish()