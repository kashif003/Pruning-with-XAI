# this file will be used to get all the scores.

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
# Note: I changed 'get_score_GMAR++' to 'get_score_GMAR_plus_plus' to be a valid Python module name.
# Make sure to rename that specific Python file in your src/get_score/ directory!
scripts_to_run = [
    ("GMAR", "src.get_score.get_score_GMAR"),
    ("GMAR++", "src.get_score.get_score_GMAR++"), 
    ("chefar", "src.get_score.get_score_chefar"),
    ("attnlrp", "src.get_score.get_score_attlrp"),
    ("legrad", "src.get_score.get_score_legrad"),
    ("lrp", "src.get_score.get_score_lrp"),
    #("mag", "src.get_score.get_score_mag")
]

# Loop through the list and run each script
for name, module in scripts_to_run:
    time.sleep(1)
    print(f"Making {name} score")
    run_score_script(name, module)

print("\nAll experiments completed!")



