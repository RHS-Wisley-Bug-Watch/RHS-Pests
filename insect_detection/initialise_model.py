import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np

def create_model(train_ds, device, learning_rate=1e-4, frozen_layers=True):
    print("Initializing ResNet18 model...")
    
    insect_class_id = train_ds.class_to_idx['Insect']
    other_class_id = train_ds.class_to_idx['Other']

    targets = np.array(train_ds.targets)
    insect_count = np.sum(targets == insect_class_id)
    other_count = np.sum(targets == other_class_id)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    if frozen_layers:
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    weights = torch.tensor([1.0, 1.0], dtype=torch.float)
    weights[insect_class_id] = other_count / max(1, insect_count)
    weights[other_class_id] = 1.0
    weights = weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    return model, criterion, optimizer, insect_class_id, other_class_id