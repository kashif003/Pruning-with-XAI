import subprocess
import argparse
import os

parser = argparse.ArgumentParser()
# Changed argument name to reflect that it expects a directory, e.g., "scores/small"
parser.add_argument("--score_dir", type=str, required=True)
args = parser.parse_args()

score_dir = args.score_dir
configs = [10 , 20, 30, 40, 50, 60]

def run_all_experiments(score_file, configs):
    for cfg in configs:
        print(f"\n[INFO] Pruning configs: {cfg}, score_file: {score_file}")
        
        # os.path.join safely combines the folder path and the file name
        # e.g., "scores/small" + "GMARPP_score.json" -> "scores/small/GMARPP_score.json"
        full_path = os.path.join(score_dir, score_file)
        
        try:
            subprocess.run([
                "python3",
                "-m",
                "src.main",  # Removed the .py
                "--config",
                str(cfg),
                "--score_path",
                full_path
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Script failed for config {cfg} on {score_file}. Moving to next...")

# Read the directory
try:
    # This will grab all the JSON files inside "scores/small"
    score_list = os.listdir(score_dir)
    
    # Run the loop
    for score in score_list:
        # Optional: ensure we only run on JSON files, ignoring hidden files or folders
        if score.endswith(".json"):
            run_all_experiments(score, configs)
            
    # Success message placed at the very end
    print("\nAll experiments completed!")

except NotADirectoryError:
    print(f"Error: '{score_dir}' is a file. Please pass a directory (like 'scores/small').")
except FileNotFoundError:
    print(f"Error: The directory '{score_dir}' was not found.")