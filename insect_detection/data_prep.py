import os
import shutil
import pandas as pd
import random
import torch
from torchvision.transforms import v2
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image

def class_from_label(label):
    lbl = str(label).strip().lower()
    if any(bug in lbl for bug in ["thrip", "pirate bug", "insect", "bug"]):
        return "Insect"
    return "Other"

def get_classification(row):
    for col in ("Split_Labels", "Chosen_Label", "InsectIDs"):
        if col in row.index:
            value = row[col]
            if pd.notna(value) and str(value).strip() and str(value).strip().lower() != 'nan':
                if col == "InsectIDs":
                    val_str = str(value)
                    if 'Thrip' in val_str or 'Pirate Bug' in val_str:
                        return "Insect"
                    elif 'Other/Unknown' in val_str or val_str == '':
                        return "Other"
                else:
                    return class_from_label(value)
    return "Other"

def prepare_data(csv_file, source_dir, base_dir, num_augmentations):
    output_dirs = [
        os.path.join(base_dir, "train_balanced", "Insect"),
        os.path.join(base_dir, "train_balanced", "Other"),
        os.path.join(base_dir, "test", "Insect"),
        os.path.join(base_dir, "test", "Other"),
    ]

    for folder in output_dirs:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

    print(f"Loading CSV: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"Total rows found: {len(df)}")

    file_classes = {}
    for _, row in df.iterrows():
        filename = str(row.get('Filename', '')).strip()
        if not filename or filename.lower() == 'nan':
            continue
        
        classification = get_classification(row)
        if filename not in file_classes:
            file_classes[filename] = classification
        elif classification == "Insect":
            file_classes[filename] = "Insect"

    insects = [f for f, c in file_classes.items() if c == 'Insect']
    others = [f for f, c in file_classes.items() if c == 'Other']

    augmenter = v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomChoice([
            v2.RandomRotation([0, 0]),     
            v2.RandomRotation([90, 90]),   
            v2.RandomRotation([180, 180]), 
            v2.RandomRotation([270, 270])  
        ]),
        v2.Pad(padding=75, padding_mode='reflect'),
        v2.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        v2.CenterCrop(224),
        v2.RandomChoice([
            v2.ColorJitter(brightness=(0.8, 1.2), contrast=0.3, saturation=(0.5, 1.5), hue=0.05),
            v2.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 1.0)),
            v2.Lambda(lambda img: img) 
        ]),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.ToPILImage() 
    ])

    resizer = v2.Resize((224, 224), antialias=True)

    def split_and_augment(files, class_name):
        print(f"Processing {len(files)} {class_name} images...")
        
        random.shuffle(files)
        train_count = int(len(files) * 0.8)
        train_raw = files[:train_count]
        test_raw = files[train_count:]
        
        def process_and_save(raw_list, subset_folder):
            count = 0
            for f in raw_list:
                src = os.path.join(source_dir, f)
                if not os.path.exists(src):
                    continue
                    
                img = Image.open(src).convert('RGB')
                
                out_img = resizer(img)
                out_name = f"{os.path.splitext(f)[0]}_orig.jpg"
                out_path = os.path.join(base_dir, subset_folder, class_name, out_name)
                out_img.save(out_path)
                count += 1
                
                if class_name == 'Insect':
                    for i in range(num_augmentations):
                        out_img = augmenter(img)
                        out_name = f"{os.path.splitext(f)[0]}_aug_{i}.jpg"
                        out_path = os.path.join(base_dir, subset_folder, class_name, out_name)
                        out_img.save(out_path)
                        count += 1
            return count

        copied_train = process_and_save(train_raw, 'train_balanced')
        copied_test = process_and_save(test_raw, 'test')
        return copied_train, copied_test

    train_ins, test_ins = split_and_augment(insects, 'Insect')
    train_oth, test_oth = split_and_augment(others, 'Other')

    print(f"TRAIN SET: {train_ins} Insects | {train_oth} Others")
    print(f"TEST SET:  {test_ins} Insects | {test_oth} Others")

def get_dataloaders(base_dir, batch_size):
    train_root = os.path.join(base_dir, "train_balanced")
    test_root = os.path.join(base_dir, "test")

    standard_transform = v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(train_root, transform=standard_transform)
    test_ds  = datasets.ImageFolder(test_root, transform=standard_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_ds, test_ds