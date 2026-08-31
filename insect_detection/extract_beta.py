"""
Little script that helps with updating the filtered spreadsheet with "No Bug" images. 
It will print the filenames and their corresponding row numbers in the original CSV, 
which can be used to update the filtered spreadsheet.
"""

import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(script_dir, "Final_insect_numbers.csv")

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"Could not find {CSV_FILE}. Make sure it is in {script_dir}!")

print(f"Reading data from {CSV_FILE}...")
df = pd.read_csv(CSV_FILE)

gold_files = [f for f in os.listdir(script_dir) if f.startswith("Filtered_Bugs_Min4") and f.endswith(".csv")]
gold_df = None

if gold_files:
    gold_files.sort(key=lambda x: os.path.getmtime(os.path.join(script_dir, x)), reverse=True)
    gold_path = os.path.join(script_dir, gold_files[0])
    print(f"Detected Gold Standard file: {gold_files[0]}")
    gold_df = pd.read_csv(gold_path)
else:
    print("No Gold Standard CSV found in directory. Proceeding without deduplication.")

mask_empty = df['Participants_with_PositiveDetection'] == 0

df['InsectIDs'] = df['InsectIDs'].fillna('')
mask_other_only = (
    (df['Participants_with_PositiveDetection'] > 0) & 
    (df['InsectIDs'].str.contains('Other/Unknown', na=False)) & 
    (~df['InsectIDs'].str.contains('Thrip', na=False)) & 
    (~df['InsectIDs'].str.contains('Pirate Bug', na=False))
)

no_bugs_df = df[mask_empty | mask_other_only].copy()

if gold_df is not None and 'Filename' in gold_df.columns:
    gold_filenames = set(gold_df['Filename'].dropna().unique())
    initial_count = len(no_bugs_df)
    
    no_bugs_df = no_bugs_df[~no_bugs_df['Filename'].isin(gold_filenames)]
    removed_count = initial_count - len(no_bugs_df)
    
    if removed_count > 0:
        print(f"-> Removed {removed_count} 'No Bug' images that were already present in the Gold Standard.")

if 'TotalParticipants' in no_bugs_df.columns:
    no_bugs_df = no_bugs_df.sort_values(by='TotalParticipants', ascending=False)
    
no_bugs_df = no_bugs_df.drop_duplicates(subset=['Filename'])

target_amount = 300
no_bugs_df = no_bugs_df.head(target_amount)

OUT_FILE = os.path.join(script_dir, "High_Confidence_No_Bugs_Top300.csv")
no_bugs_df.to_csv(OUT_FILE, index=False)

print("\n--- EXTRACTION COMPLETE ---")
print(f"Total raw rows scanned: {len(df)}")
print(f"-> Success! Extracted the top {len(no_bugs_df)} highest-confidence 'No Bug' images.")
print(f"-> File saved to: {OUT_FILE}")