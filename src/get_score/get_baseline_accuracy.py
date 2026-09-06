import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import default_data_collator # Added collator
from ..utils import load_model_transformers
from tqdm import tqdm
from ..config import MODEL_NAME, GPU_NAME
import wandb

# 1. Setup Device
device = torch.device(GPU_NAME if torch.cuda.is_available() else "cpu")

# 2. Load Model and Processor
processor, model = load_model_transformers(MODEL_NAME) 
model.to(device) # Ensure model is on device

# 3. Load Streaming Dataset
dataset = load_dataset(
    "ILSVRC/imagenet-1k",
    split="validation",
    streaming=True
)

def transform(examples):
    rgb_images = [img.convert("RGB") for img in examples["image"]]
    inputs = processor(rgb_images, return_tensors="pt")
    inputs["labels"] = examples["label"] # Keep as list, collator handles tensor conversion
    return inputs

processed_dataset = dataset.map(
    transform,
    batched=True,
    remove_columns=["image", "label"]
)

# FIXED: Lowered batch size for streaming, added workers, added collator
batch_size = 512 
val_loader = DataLoader(
    processed_dataset, 
    batch_size=batch_size, 
    num_workers=14, # Multiprocessing for faster I/O max = 14
    collate_fn=default_data_collator 
)

NUM_CLASSES = 1000  # ImageNet-1K

def validate(model, device):
    model.eval()
    total_correct = 0
    total_samples = 0

    class_correct = torch.zeros(NUM_CLASSES, dtype=torch.long)
    class_total = torch.zeros(NUM_CLASSES, dtype=torch.long)

    print("Starting data stream...\n")
    print("[INFO] Current batch size:", val_loader.batch_size)
    pbar = tqdm(val_loader, desc="Evaluating")
    
    with torch.no_grad():
        for batch in pbar:
            images = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            if images.ndim == 5:
                images = images.squeeze(1)

            outputs = model(images)
            preds = outputs.logits.argmax(dim=-1)

            correct_mask = (preds == labels)

            batch_correct = correct_mask.sum().item()
            total_correct += batch_correct
            total_samples += labels.size(0)

            labels_cpu = labels.cpu()
            correct_cpu = correct_mask.cpu()

            class_total.index_add_(0, labels_cpu, torch.ones_like(labels_cpu))
            class_correct.index_add_(0, labels_cpu, correct_cpu.long())

            current_acc = (total_correct / total_samples) * 100

            wandb.log({"Per batch Accuracy": current_acc})
            pbar.set_postfix({"accuracy": f"{current_acc:.2f}%"})

    final_accuracy = (total_correct / total_samples) * 100

    seen_mask = class_total > 0
    class_accuracy = torch.zeros(NUM_CLASSES, dtype=torch.float)
    class_accuracy[seen_mask] = class_correct[seen_mask].float() / class_total[seen_mask].float()

    valid_acc = class_accuracy[seen_mask]
    if (valid_acc == 0).any():
        harmonic_mean = torch.tensor(0.0)
    else:
        N = valid_acc.numel()
        harmonic_mean = N / (1.0 / valid_acc).sum()

    return {
        "final_accuracy": final_accuracy,
        "class_accuracy": class_accuracy,       
        "class_total": class_total,             
        "harmonic_mean_accuracy": harmonic_mean.item() * 100
    }