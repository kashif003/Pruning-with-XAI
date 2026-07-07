# XAI-Driven Structured Pruning of Vision Transformers

Structured pruning of Vision Transformer attention heads guided by explainability-based importance scoring, with evaluation on ImageNet-1k accuracy and computational efficiency.

---

## Overview

This project prunes entire attention heads from `google/vit-large-patch16-384` (ViT-Large) using importance scores derived from gradient-based XAI methods. Heads are ranked by their importance score and the lowest-scoring ones are removed using a structural `DependencyGraph` (via `torch-pruning`), which automatically propagates changes across all dependent layers (`Q`, `K`, `V`, `out_proj`).

**Model:** ViT-Large — 24 layers × 16 heads = 384 total heads  
**Baseline:** 304.1M parameters, 174.8G FLOPs, 85.81% Top-1 on ImageNet-1K

---

## Scoring Methods

Five head importance scoring methods are compared:

| Method | Formula |
|---|---|
| **Magnitude** | $s_h = \|W^Q_h\|_2 + \|W^K_h\|_2 + \|W^V_h\|_2$ |
| **GMAR** | $s_h = \frac{1}{N}\sum_n \sum_{i,j} \left\|\frac{\partial y_c}{\partial A_h[i,j]}\right\|$ |
| **GMAR++** | $s_h = \frac{1}{N}\sum_n \sqrt{\sum_{i,j} \left(\max\left(\frac{\partial y_c}{\partial A_h[i,j]}, 0\right)\right)^2}$ |
| **LeGrad** | $s_h = \frac{1}{N}\sum_n \sum_{i,j} \max\left(\frac{\partial y_c}{\partial A_h[i,j]}, 0\right)$ |
| **Chefer CAM** | $s_h = \frac{1}{N}\sum_n \sum_{i,j} \max\left(\nabla A_h[i,j] \cdot R_h[i,j], 0\right)$ |

---

## Results

Top-1 Accuracy (%) on ImageNet-1K — no fine-tuning after pruning.

### Model Size After Pruning

| | Baseline | 10% | 20% | 30% | 40% | 50% | 60% |
|---|---|---|---|---|---|---|---|
| **Params** | 304.1M | 294.2M | 284.2M | 274.0M | 264.8M | 255.6M | 246.4M |
| **FLOPs** | 174.8G | 169.1G | 163.3G | 157.4G | 152.1G | 146.8G | 141.5G |

### Top-1 Accuracy (%)

| Method | Baseline | 10% | 20% | 30% | 40% | 50% | 60% |
|---|---|---|---|---|---|---|---|
| LeGrad | 85.81 | 82.02 | 77.97 | 69.21 | 51.24 | 27.70 | 6.66 |
| Chefer | 85.81 | 82.48 | 78.28 | 70.16 | 51.64 | 24.33 | 6.65 |
| Magnitude | 85.81 | 82.39 | 71.94 | 54.90 | 38.70 | 11.24 | 6.62 |
| GMAR | 85.81 | **83.98** | 81.92 | **77.18** | 67.01 | **53.80** | **40.45** |
| GMAR++ | 85.81 | 83.85 | **81.95** | 76.57 | **67.76** | 53.28 | 38.33 |

---

## Project Structure

```
GMAR-Plus-with-pruning/
├── src/
│   ├── main.py                  # Entry point: prune model and evaluate accuracy
│   ├── run_experiment.py        # Loops over pruning ratios and calls main.py
│   ├── vit.py                   # Custom ViT wrapper with forward passes for each scoring method
│   ├── prune.py                 # Head pruning via torch-pruning DependencyGraph
│   ├── GMAR.py                  # GMAR heatmap computation and visualisation
│   ├── utils.py                 # Dataset, transforms, evaluation, and pruning utilities
│   ├── hooks.py                 # Gradient hooks
│   └── get_score/
│       ├── get_score_GMAR++.py  # Compute GMAR++ head importance scores
│       ├── get_score_GMAR.py    # Compute GMAR head importance scores
│       ├── get_score_legrad.py  # Compute LeGrad head importance scores
│       ├── get_score_chefar.py  # Compute Chefer CAM head importance scores
│       ├── get_score_mag.py     # Compute magnitude head importance scores
│       ├── get_baseline_accuracy.py  # ImageNet-1K streaming evaluation loop
│       └── get_dataset.py
├── scores/                      # Pre-computed JSON importance scores per method
├── logs/                        # Training/experiment output logs
├── plots/                       # Score distribution plots
├── images/                      # Test images and reference architecture diagrams
├── masks/                       # Segmentation masks
├── requirements.txt
└── .env                         # WANDB_KEY (create this yourself — not committed)
```

---

## Setup

**1. Clone and create a virtual environment**
```bash
git clone https://github.com/kashif003/GMAR-Plus-with-pruning.git
cd GMAR-Plus-with-pruning
python3 -m venv venv
source venv/bin/activate
```

**2. Install dependencies**

For GPU (replace `cu121` with your CUDA version):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**3. Create a `.env` file** with your Weights & Biases API key:
```
WANDB_KEY=your_wandb_api_key
```

**4. Log in to Hugging Face** (`ILSVRC/imagenet-1k` is a gated dataset — accept terms first):
```bash
huggingface-cli login
```

---

## Generating Importance Scores (Optional)

Pre-computed scores for all methods are already provided in the `scores/` directory. If you want to recompute them from scratch, follow these steps.

**Step 1 — Prepare the dataset**

To prepare for scoring, you need to download the 1,000 validation samples from the ImageNet-1K dataset. Run the following script to automate this:

```bash
python src/get_score/get_dataset_1k.py

**Step 2 — Run the scoring script for your chosen method**

From the project root:

```bash
# GMAR++
python3 -m src.get_score.get_score_GMAR++

# GMAR
python3 -m src.get_score.get_score_GMAR

# LeGrad
python3 -m src.get_score.get_score_legrad

# Chefer CAM
python3 -m src.get_score.get_score_chefar

# Magnitude
python3 -m src.get_score.get_score_mag
```

Each script writes its output JSON to `scores/` (e.g. `scores/GMARPP_score.json`). Once generated, pass that file to the experiment runner.

---

## Running the Experiments

Run all pruning ratios (10% → 60%) for a given scoring method:
```bash
cd src
python3 run_experiment.py --score_path ../scores/LeGrad_score.json
```

Run a single ratio:
```bash
cd src
python3 main.py --config 20 --score_path ../scores/LeGrad_score.json
```

Available score files in `scores/`:
- `LeGrad_score.json`
- `GMARPP_score.json`
- `GMAR_score.json`
- `chefer_score.json`
- `mag_score.json`

---

## References

- [1] P. Michel, O. Levy, G. Neubig — *Are Sixteen Heads Really Better than One?* NeurIPS, 2019.
- [2] G. Fang et al. — *DepGraph: Towards Any Structural Pruning.* CVPR, 2023.
