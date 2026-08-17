import os
import torch
import random
import numpy as np
import wandb 

from data_prep import prepare_data, get_dataloaders
from initialise_model import create_model
from training_eval import train_and_evaluate


EXPERIMENT_CONFIG = {
    "learning_rate": 1e-3,   # 1e-3 (faster learning) or 1e-5 (slower/cautious)
    "epochs": 25,            # How many learning iterations
    "batch_size": 32,        # How many samples it looks at at once
    "num_augmentations": 8,  # Augmenting only images with bugs in them for balance
    "threshold": 0.5,        # Model confidence for classification
    "frozen_layers": True,
    "architecture": "ResNet18",
    "notes": "Retaining learning rate, increasing augmentations. Model 3."
}

BASE_DIR = os.getcwd()
CSV_CANDIDATES = [os.path.join(BASE_DIR, "Filtered_Bugs_Min4_Frac0.55_Gold-x.csv")]
CSV_FILE = next((p for p in CSV_CANDIDATES if os.path.exists(p)), None)
SOURCE_DIR = os.path.join(BASE_DIR, "images")
RESULTS_DIR = "model_results"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    if not CSV_FILE:
        raise FileNotFoundError("Could not find CSV.")

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # INITIALIZE WEIGHTS & BIASES
    # ==========================================
    run_name = f"LR-{EXPERIMENT_CONFIG['learning_rate']}_Aug-{EXPERIMENT_CONFIG['num_augmentations']}_Thresh-{EXPERIMENT_CONFIG['threshold']}"
    
    wandb.init(
        project="rhs-wisley-bug-watch", 
        name=run_name,
        notes=EXPERIMENT_CONFIG["notes"],
        config=EXPERIMENT_CONFIG
    )

    print(f"\nStarting experiment: {run_name}")
    print(f"Notes: {EXPERIMENT_CONFIG['notes']}\n")

    prepare_data(CSV_FILE, SOURCE_DIR, BASE_DIR, EXPERIMENT_CONFIG["num_augmentations"])
    train_loader, test_loader, train_ds, test_ds = get_dataloaders(BASE_DIR, EXPERIMENT_CONFIG["batch_size"])
    
    model, criterion, optimizer, insect_idx, other_idx = create_model(
        train_ds, device, EXPERIMENT_CONFIG["learning_rate"]
    )
    
    train_and_evaluate(
        model=model, 
        criterion=criterion, 
        optimizer=optimizer, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        test_ds=test_ds, 
        insect_class_id=insect_idx, 
        other_class_id=other_idx, 
        device=device,
        threshold=EXPERIMENT_CONFIG["threshold"],
        epochs=EXPERIMENT_CONFIG["epochs"],
        results_dir=RESULTS_DIR
    )

    wandb.finish()

if __name__ == "__main__":
    main()