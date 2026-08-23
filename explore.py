 # this file will be use to read all the scoring methods and make confusion matrix.

import json
import glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

SCORE_DIR = "/media/Storage/Pruning + XAI/Pruning-with-XAI/scores"

# adjust this if filenames differ - dedupe typos/case-duplicates first
score_files = {
    "Magnitude": "mag_score.json",
    "GMAR": "GMAR_score.json",
    "GMAR++": "GMARPP_score.json",
    "LeGrad": "LeGrad_score.json",       # confirm this vs legrad_score.json
    "Chefer": "chefer_score.json",       # confirm this vs chefar_score.json
}

def load_flat_scores(path):
    with open(path, "r") as f:
        data = json.load(f)

    # assumes structure: {layer_idx (str/int): [score_head_0, ..., score_head_11]}
    sorted_layers = sorted(data.keys(), key=lambda x: int(x))
    flat = []
    for layer in sorted_layers:
        flat.extend(data[layer])
    return np.array(flat)

method_scores = {}
for name, fname in score_files.items():
    full_path = f"{SCORE_DIR}/{fname}"
    method_scores[name] = load_flat_scores(full_path)

# sanity check: all vectors must be same length (num_layers * 12)
lengths = {k: len(v) for k, v in method_scores.items()}
print("Vector lengths per method:", lengths)
assert len(set(lengths.values())) == 1, "Mismatched lengths - check layer/head counts across files"

methods = list(method_scores.keys())
n = len(methods)
corr_matrix = np.zeros((n, n))
pval_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        rho, pval = spearmanr(method_scores[methods[i]], method_scores[methods[j]])
        corr_matrix[i, j] = rho
        pval_matrix[i, j] = pval

corr_df = pd.DataFrame(corr_matrix, index=methods, columns=methods)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
            square=True, cbar_kws={"label": "Spearman ρ"})
plt.title("Spearman Rank Correlation Between XAI Scoring Methods")
plt.tight_layout()
plt.savefig("score_correlation_heatmap.png", dpi=300)
plt.show()

print(corr_df.round(3))