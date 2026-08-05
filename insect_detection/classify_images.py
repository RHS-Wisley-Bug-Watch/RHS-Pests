
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import random
import shutil
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# STRONGER AUG
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.7,1.0)),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# loading datasetds
train_root = "train_balanced"
train_ds = datasets.ImageFolder(train_root, transform=train_transform)
test_ds  = datasets.ImageFolder("test", transform=test_transform)

# extreme oversampling
insect_class_id = train_ds.class_to_idx['Insect']
other_class_id = train_ds.class_to_idx['Other']

insect_idx = [i for i, (_, label) in enumerate(train_ds.samples) if label == insect_class_id]
other_idx  = [i for i, (_, label) in enumerate(train_ds.samples) if label == other_class_id]

# repeat insect 10x --- Revisit this
insect_datasets = [torch.utils.data.Subset(train_ds, insect_idx)] * 10
other_datasets = [torch.utils.data.Subset(train_ds, other_idx)]
# Combine with Other once
balanced_ds = ConcatDataset([*insect_datasets, *other_datasets])
train_loader = DataLoader(balanced_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

# freeze pretrained layers
model = models.resnet18(pretrained=True)
for name, param in model.named_parameters():
    if "fc" not in name:
        param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# weighted loss
class_counts = np.array([len(other_idx), len(insect_idx)])
weights = torch.tensor([1.0, 1.0], dtype=torch.float)
weights[insect_class_id] = class_counts[0] / max(1, class_counts[1]) 
weights[other_class_id] = 1.0
weights = weights.to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)  # only fc parameters train

# training
EPOCHS = 15
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

#eval w low threshold
model.eval()
preds, labels_list, all_probs = [], [], []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        out = model(x)
        probs = torch.softmax(out, dim=1)
        insect_preds = (probs[:, insect_class_id] > 0.5).int().cpu().numpy()
        preds.extend(insect_preds)
        labels_list.extend(y.numpy())
        all_probs.append(probs.cpu())

all_probs = torch.cat(all_probs)
print("\nSample probabilities:")
print(all_probs[:10])
print(test_ds.classes)


print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(labels_list, preds, target_names=test_ds.classes))

cm = confusion_matrix(labels_list, preds)
plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=test_ds.classes,
            yticklabels=test_ds.classes)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
