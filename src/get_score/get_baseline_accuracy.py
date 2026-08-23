import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoImageProcessor
from tqdm import tqdm
import wandb

# 1. Setup Device
device = "cuda:6" if torch.cuda.is_available() else "cpu"

# 2. Load Model and Processor
processor = AutoImageProcessor.from_pretrained("google/vit-large-patch16-384")

# 3. Load Streaming Dataset
dataset = load_dataset(
    "ILSVRC/imagenet-1k",
    split="validation",
    streaming=True
)

def transform(examples):
    rgb_images = [img.convert("RGB") for img in examples["image"]]
    inputs = processor(rgb_images, return_tensors="pt")
    inputs["labels"] = torch.tensor(examples["label"])
    return inputs

processed_dataset = dataset.map(
    transform,
    batched=True,
    remove_columns=["image", "label"]
)

batch_size = 128
val_loader = DataLoader(processed_dataset, batch_size=batch_size)

NUM_CLASSES = 1000  # ImageNet-1K

def validate(model):
    model.eval()
    total_correct = 0
    total_samples = 0

    # per-class counters
    class_correct = torch.zeros(NUM_CLASSES, dtype=torch.long)
    class_total = torch.zeros(NUM_CLASSES, dtype=torch.long)

    print("Starting data stream...\n")
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

            # accumulate per-class counts (move to CPU for indexing)
            labels_cpu = labels.cpu()
            correct_cpu = correct_mask.cpu()

            class_total.index_add_(0, labels_cpu, torch.ones_like(labels_cpu))
            class_correct.index_add_(0, labels_cpu, correct_cpu.long())

            current_acc = (total_correct / total_samples) * 100

            wandb.log({"Per batch Accuracy": current_acc})
            pbar.set_postfix({"accuracy": f"{current_acc:.2f}%"})

    final_accuracy = (total_correct / total_samples) * 100

    # class-wise accuracy (only for classes actually seen)
    seen_mask = class_total > 0
    class_accuracy = torch.zeros(NUM_CLASSES, dtype=torch.float)
    class_accuracy[seen_mask] = class_correct[seen_mask].float() / class_total[seen_mask].float()

    # harmonic mean over classes that were seen and have nonzero accuracy
    valid_acc = class_accuracy[seen_mask]
    if (valid_acc == 0).any():
        harmonic_mean = torch.tensor(0.0)
    else:
        N = valid_acc.numel()
        harmonic_mean = N / (1.0 / valid_acc).sum()

    return {
        "final_accuracy": final_accuracy,
        "class_accuracy": class_accuracy,       # tensor of shape [NUM_CLASSES]
        "class_total": class_total,             # samples seen per class
        "harmonic_mean_accuracy": harmonic_mean.item() * 100
    }