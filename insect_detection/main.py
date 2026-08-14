import os
import torch
import random
import numpy as np
import wandb

from data_prep import prepare_data, get_dataloaders
from initialise_model import create_model
from training_eval import train_and_evaluate

# ==========================================
# 1. CONFIGURATION & PARAMETERS
# ==========================================
BASE_DIR = os.getcwd()
CSV_CANDIDATES = [os.path.join(BASE_DIR, "Filtered_Bugs_Min4_Frac0.55_Gold-x.csv")]
CSV_FILE = next((p for p in CSV_CANDIDATES if os.path.exists(p)), None)
SOURCE_DIR = os.path.join(BASE_DIR, "images")
RESULTS_DIR = "model_results"

EPOCHS = 25
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_AUGMENTATIONS = 6
THRESHOLD = 0.5 

# ==========================================
# RANDOM SEED
# ==========================================
def set_seed(seed=67):
    print(f"Random seed set to {seed}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    if not CSV_FILE:
        raise FileNotFoundError(f"Could not find a valid filtered bugs CSV file in {BASE_DIR}")

    set_seed(67)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="rhs-wisley-bug-watch",
        name=f"ResNet18_LR-{LEARNING_RATE}_Aug-{NUM_AUGMENTATIONS}",
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "augmentations": NUM_AUGMENTATIONS,
            "threshold": THRESHOLD,
            "architecture": "ResNet18"
        }
    )


    # ==========================================
    # DATA PREP -> MODEL INIT -> TRAIN & EVAL
    # ==========================================
    prepare_data(CSV_FILE, SOURCE_DIR, BASE_DIR, NUM_AUGMENTATIONS)
    train_loader, test_loader, train_ds, test_ds = get_dataloaders(BASE_DIR, BATCH_SIZE)
    model, criterion, optimizer, insect_idx, other_idx = create_model(train_ds, device, LEARNING_RATE)
    
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
        threshold=THRESHOLD,
        epochs=EPOCHS,
        results_dir=RESULTS_DIR
    )

    wandb.finish()
    
if __name__ == "__main__":
    main()