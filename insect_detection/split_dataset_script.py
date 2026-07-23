import os
import shutil
import pandas as pd
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "Filtered_Bugs_Min4_Frac0.55_Gold.csv")
SOURCE_DIR = os.path.join(BASE_DIR, "images")
OUTPUT_DIRS = [
    os.path.join(BASE_DIR, "train_balanced", "Insect"),
    os.path.join(BASE_DIR, "train_balanced", "Other"),
    os.path.join(BASE_DIR, "test", "Insect"),
    os.path.join(BASE_DIR, "test", "Other"),
]


def class_from_label(label):
    label = str(label).strip()
    if label in {"Thrip", "Pirate Bug", "Other Insect", "Possible Insect"}:
        return "Insect"
    return "Other"

for folder in OUTPUT_DIRS:
    os.makedirs(folder, exist_ok=True)

print(f"Created folders: {', '.join(OUTPUT_DIRS)}")

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"Could not find CSV file: {CSV_FILE}")

df = pd.read_csv(CSV_FILE)

file_classes = {}

for _, row in df.iterrows():
    filename = str(row['Filename']).strip()
    if not filename or filename == 'nan':
        continue

    label = row.get('Chosen_Label', '')

    is_insect = class_from_label(label) == 'Insect'

    if filename not in file_classes:
        file_classes[filename] = 'Insect' if is_insect else 'Other'
    else:
        if is_insect:
            file_classes[filename] = 'Insect'

insects = [f for f, c in file_classes.items() if c == 'Insect']
others = [f for f, c in file_classes.items() if c == 'Other']

random.seed(42)
random.shuffle(insects)
random.shuffle(others)

def split_and_copy(files, class_name):

    train_count = int(len(files) * 0.8)
    
    train_files = files[:train_count]
    test_files = files[train_count:]
    
    copied_train = 0
    copied_test = 0

    for f in train_files:
        src = os.path.join(SOURCE_DIR, f)
        dst = os.path.join(BASE_DIR, 'train_balanced', class_name, f)
        if os.path.exists(src):
            shutil.copy(src, dst)
            copied_train += 1
            
    for f in test_files:
        src = os.path.join(SOURCE_DIR, f)
        dst = os.path.join(BASE_DIR, 'test', class_name, f)
        if os.path.exists(src):
            shutil.copy(src, dst)
            copied_test += 1
            
    return copied_train, copied_test

print("\nSorting images...")
train_ins, test_ins = split_and_copy(insects, 'Insect')
train_oth, test_oth = split_and_copy(others, 'Other')

print("\n--- SPLIT COMPLETE ---")
print(f"Total Unique Images Processed: {train_ins + test_ins + train_oth + test_oth}")
print(f"TRAIN SET (80%): {train_ins} Insects | {train_oth} Others")
print(f"TEST SET  (20%): {test_ins} Insects | {test_oth} Others")