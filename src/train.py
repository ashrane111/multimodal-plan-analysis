import socket
original_socket = socket.socket

class SafeSocket(original_socket):
    def connect(self, address):
        # Allow connecting to local system sockets (strings)
        # PyTorch multiprocessing uses these (e.g., '/tmp/pymp-...')
        if isinstance(address, str):
            return super().connect(address)
            
        # Allow connecting to localhost (127.0.0.1)
        if isinstance(address, tuple) and address[0] in ['127.0.0.1', 'localhost']:
            return super().connect(address)
            
        # BLOCK external connections (Hugging Face, etc.)
        raise ConnectionError(f"👮 CAUGHT YOU! A library tried to connect to external: {address}")

# Apply the patch
socket.socket = SafeSocket

import sys
import os
import torch
import numpy as np
from transformers import Trainer, TrainingArguments, LayoutLMv3Processor
import evaluate 

if not torch.cuda.is_available():
    print("❌ FATAL ERROR: PyTorch cannot find a GPU!")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Device Count: {torch.cuda.device_count()}")
    print("Exiting to prevent slow CPU training.")
    sys.exit(1)
else:
    print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")

# --- PATH PATCH START ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# --- PATH PATCH END ---

from src.data.dataset import FloorPlanDataset
from src.models.layoutlm_v3 import LayoutLMv3Model

def get_compute_metrics(id2label):
    metric = evaluate.load("seqeval")
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    return compute_metrics

def main():
    # 1. Setup Processor
    processor = LayoutLMv3Processor.from_pretrained("layoutlmv3_base", apply_ocr=False)

    # 2. Initialize Dataset
    print("Loading Arrow dataset...")
    # --- CHANGED HERE ---
    full_dataset = FloorPlanDataset(
        arrow_path="data/processed/cubicasa_arrow", # Point to new Arrow folder
        processor=processor
    )
    
    # Filter out "UNDEFINED" from the label map to avoid training on noise
    # (Simple approach: The dataset class handles mapping, so we trust it, 
    # but ideally we'd filter the CSV first. For now, we train on what we have.)
    
    # Split into Train (90%) and Val (10%)
    # We use a fixed seed for reproducibility
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    print(f"Training on {len(train_dataset)} samples, Validating on {len(eval_dataset)} samples.")

    # 3. Initialize Model
    labels = full_dataset.labels
    id2label = full_dataset.id2label 
    label2id = full_dataset.label2id
    
    model_wrapper = LayoutLMv3Model(
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )
    
    # 4. Training Arguments (PRODUCTION SETTINGS)
    training_args = TrainingArguments(
        output_dir="models_output/checkpoints",
        num_train_epochs=3,              # Iterate over data 3 times
        per_device_train_batch_size=8,   # Higher batch size for GPU (Try 8 or 16)
        per_device_eval_batch_size=4,
        eval_accumulation_steps=1,       # CRITICAL: Offload predictions to CPU immediately
        gradient_checkpointing=False,
        learning_rate=5e-5,
        eval_strategy="steps",
        eval_steps=500,                  # Evaluate every 500 steps
        save_steps=500,                  # Save checkpoint every 500 steps
        logging_steps=100,
        save_total_limit=2,              # Only keep last 2 checkpoints to save space
        remove_unused_columns=False,
        report_to="none",
        fp16=True,                       # Enable Mixed Precision (Much faster on V100/A100)
        dataloader_num_workers=4         # Preload data faster
    )

    # 5. Trainer
    trainer = Trainer(
        model=model_wrapper.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=get_compute_metrics(id2label),
    )

    print("Starting Full Training...")
    # Check if a valid checkpoint exists
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        from transformers.trainer_utils import get_last_checkpoint
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint:
            print(f"Found checkpoint at {last_checkpoint}. Resuming training...")

    # Pass the checkpoint (or None) to train()
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # 6. Save Final Model
    final_path = "models_output/layoutlmv3_finetuned_full"
    trainer.save_model(final_path)
    processor.save_pretrained(final_path)
    print(f"Model saved to {final_path}")

if __name__ == "__main__":
    main()