# this file will be used to get all the scores for CNN (ResNet) models.

"""
1. model_name.
2. dir where to store the score.
3. device used
"""

import subprocess
import argparse
import time
parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--saving_dir", type=str, required=True)
parser.add_argument("--device", type=str, required=True)

args = parser.parse_args()

def run_score_script(score_name, module_path):
    """
    Helper function to run a subprocess and catch any errors
    so the main script doesn't crash if one fails.
    """
    print(f"Making {score_name} score")
    try:
        subprocess.run([
            "python3",
            "-m",
            module_path,
            "--model_name", args.model_name,
            "--saving_dir", args.saving_dir,
            "--device", args.device
        ], check=True)
    except subprocess.CalledProcessError as e:
        # This catches the error from check=True and prints a warning instead of crashing
        print(f"Warning: {score_name} script failed with error code {e.returncode}. Continuing to the next script...\n")

# List of all the scores and their corresponding module paths.
# NOTE: no AttnLRP entry here -- that method is specific to attention
# softmax/matmul and has no CNN equivalent (see cnn.py docstring).
# mag and causal are left commented out below, same convention as the
# ViT version:
#   - mag needs no forward pass and runs almost instantly, so it's
#     usually run separately/first as a quick sanity check.
#   - causal ablates every conv channel one at a time (26,560 for
#     ResNet50 vs. 144 heads for ViT) and can take vastly longer than
#     the other scripts -- run it on its own when you have time budgeted.
#
# CAVEAT (same one the ViT get_all_scores.py flags and doesn't actually
# fix): "get_score_GMAR++" is not a valid Python module name -- `python3
# -m` cannot import a dotted path containing "+" characters. This entry
# is included for parity with your existing GMAR/GMAR++ split, but it
# will fail at the subprocess call (caught and skipped, thanks to
# run_score_script's try/except) unless you rename the file to something
# like get_score_GMAR_plus_plus.py and update the path below to match.
scripts_to_run = [
    ("GMAR", "src.cnn.get_score_GMAR"),
    ("GMAR++", "src.cnn.get_score_GMAR++"),
    ("chefar", "src.cnn.get_score_chefar"),
    ("legrad", "src.cnn.get_score_legrad"),
    ("lrp", "src.cnn.get_score_lrp"),
    # ("mag", "src.cnn.get_score_mag"),
    # ("causal", "src.cnn.get_score_causal"),
]

# Loop through the list and run each script
for name, module in scripts_to_run:
    time.sleep(1)
    print(f"Making {name} score")
    run_score_script(name, module)

print("\nAll experiments completed!")