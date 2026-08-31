import pandas as pd
import re
import os
from collections import Counter

SEARCH_LIST = [
    111738156,
    "plant_images_block2_130625_sub_1.jpeg",
    111738157,	
    "plant_images_block6_130625_sub_12.jpeg",
    111738128,	
    "plant_images_block1_130625_sub_21.jpeg",
    111738139,	
    "plant_images_block8_130625_sub_12.jpeg",
    119224026,	
    "plant_images_block5_160725_sub_40.jpeg",
    111738183,	
    "plant_images_block5_130625_sub_48.jpeg",
    119223446,	
    "plant_images_block2_210825_sub_6.jpeg",
    111738116,	
    "plant_images_block4_130625_sub_46.jpeg",
    111738645,	
    "plant_images_block3_270625_sub_8.jpeg",
    111738897,	
    "plant_images_block8_190625_sub_10.jpeg",
    119224609,	
    "plant_images_block2_180825_sub_7.jpeg",
    119223833,	
    "plant_images_block1_160725_sub_27.jpeg",
    111738910,	
    "plant_images_block5_190625_sub_7.jpeg",
    111738336,	
    "plant_images_block4_130625_sub_15.jpeg",
    111738886,	
    "plant_images_block2_190625_sub_6.jpeg",
    111738130,	
    "plant_images_block6_130625_sub_48.jpeg",
    111738945,	
    "plant_images_block5_190625_sub_9.jpeg",
    111738549,	
    "plant_images_block5_250725_sub_12.jpeg",
    119223685,	
    "plant_images_block6_160725_sub_13.jpeg",
    111738938,	
    "plant_images_block1_190625_sub_4.jpeg",
    119223909,	
    "plant_images_block8_160725_sub_42.jpeg",
    119223745,	
    "plant_images_block8_160725_sub_25.jpeg",
    111738904,	
    "plant_images_block3_190625_sub_1.jpeg",
    111738926,	
    "plant_images_block2_190625_sub_2.jpeg",
    119224541,	
    "plant_images_block7_180825_sub_3.jpeg",
    111738958,	
    "plant_images_block4_190625_sub_3.jpeg",
    111738182,	
    "plant_images_block8_130625_sub_48.jpeg",
    119224604,	
    "plant_images_block3_180825_sub_6.jpeg",
    111738798,	
    "plant_images_block8_280725_sub_6.jpeg",
    119223776,	
    "plant_images_block4_160725_sub_30.jpeg",
    119224571,	
    "plant_images_block5_180825_sub_3.jpeg",
    111738453,	
    "plant_images_block5_130625_sub_46.jpeg",
    111738154,	
    "plant_images_block1_130625_sub_40.jpeg",
    111813361,	
    "plant_images_block8_050825_sub_3.jpeg",
    111738471,	
    "plant_images_block1_130625_sub_9.jpeg",
    111738962,	
    "plant_images_block3_190625_sub_4.jpeg",
    119224560,	
    "plant_images_block2_180825_sub_9.jpeg",
    111979296,	
    "plant_images_block8_080825_sub_5.jpeg",
    119224063,	
    "plant_images_block4_160725_sub_38.jpeg",
    111738872,	
    "plant_images_block6_280725_sub_1.jpeg",
    111738918,	
    "plant_images_block8_190625_sub_9.jpeg",
    119224080,	
    "plant_images_block3_160725_sub_47.jpeg",
    111738436,	
    "plant_images_block5_130625_sub_47.jpeg",
    111738915,	
    "plant_images_block3_190625_sub_11.jpeg",
    119223526,	
    "plant_images_block1_210825_sub_9.jpeg",
    119223550,	
    "plant_images_block6_210825_sub_9.jpeg",
    111979325,	
    "plant_images_block2_080825_sub_6.jpeg",
    119223918,	
    "plant_images_block3_160725_sub_1.jpeg",
    111738499,	
    "plant_images_block4_250725_sub_3.jpeg",
    119224581,	
    "plant_images_block1_180825_sub_2.jpeg",
    111738799,	
    "plant_images_block4_280725_sub_9.jpeg",
    111738836,	
    "plant_images_block6_280725_sub_5.jpeg",
    111738238,	
    "plant_images_block6_130625_sub_23.jpeg",
    111738405,	
    "plant_images_block8_130625_sub_30.jpeg",
    111738957,	
    "plant_images_block3_190625_sub_5.jpeg",
    111738686,	
    "plant_images_block1_270625_sub_12.jpeg",
    119224085,	
    "plant_images_block7_160725_sub_42.jpeg",
    111813302,	
    "plant_images_block2_050825_sub_3.jpeg",
    111738887,	
    "plant_images_block4_190625_sub_1.jpeg",
    119223923,	
    "plant_images_block8_160725_sub_23.jpeg",
    119223760,	
    "plant_images_block4_160725_sub_47.jpeg",
    111738305,	
    "plant_images_block7_130625_sub_34.jpeg",
    111738444,	
    "plant_images_block4_130625_sub_48.jpeg",
    119224176,	
    "plant_images_block4_160725_sub_45.jpeg",
    111738370,	
    "plant_images_block3_130625_sub_9.jpeg",
    111813293,	
    "plant_images_block1_050825_sub_10.jpeg",
    119223956,	
    "plant_images_block1_160725_sub_25.jpeg",
    119223882,	
    "plant_images_block6_160725_sub_38.jpeg",
    119224201,	
    "plant_images_block1_160725_sub_22.jpeg",
    119224502,	
    "plant_images_block5_180825_sub_8.jpeg",
    111738663,	
    "plant_images_block6_270625_sub_10.jpeg",
    119223511,	
    "plant_images_block4_210825_sub_9.jpeg",
    119224089,	
    "plant_images_block1_160725_sub_9.jpeg",
    119224109,	
    "plant_images_block3_160725_sub_37.jpeg",
    119224549,	
    "plant_images_block5_180825_sub_2.jpeg",
    119224017,	
    "plant_images_block5_160725_sub_5.jpeg",
    119223799,	
    "plant_images_block4_160725_sub_26.jpeg",
    111738964,	
    "plant_images_block7_190625_sub_5.jpeg",
    119223486,	
    "plant_images_block7_210825_sub_10.jpeg",
    119223487,	
    "plant_images_block4_210825_sub_6.jpeg",
    111813356,	
    "plant_images_block8_050825_sub_10.jpeg",
    119223790,	
    "plant_images_block3_160725_sub_22.jpeg",
    111979311,	
    "plant_images_block5_080825_sub_6.jpeg",
    111738851,	
    "plant_images_block4_280725_sub_1.jpeg",
    111738828,	
    "plant_images_block6_280725_sub_11.jpeg",
    119224603,	
    "plant_images_block4_180825_sub_1.jpeg",
    111813377,	
    "plant_images_block3_050825_sub_11.jpeg",
    119223914,	
    "plant_images_block8_160725_sub_15.jpeg",
    111813369,	
    "plant_images_block1_050825_sub_12.jpeg",
    111738944,	
    "plant_images_block1_190625_sub_8.jpeg",
    119224120,	
    "plant_images_block7_160725_sub_32.jpeg",
    111738865,	
    "plant_images_block7_280725_sub_1.jpeg",
    119223978,	
    "plant_images_block7_160725_sub_38.jpeg",
    119223893,	
    "plant_images_block6_160725_sub_43.jpeg",
    119223965,	
    "plant_images_block4_160725_sub_4.jpeg",
    111738838,	
    "plant_images_block5_280725_sub_2.jpeg",
    111738492,	
    "plant_images_block7_130625_sub_29.jpeg",
    111979261,	
    "plant_images_block4_080825_sub_3.jpeg",
    120586361,	
    "plant_images_block5_280825_sub_5.jpeg"
]

script_dir = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(script_dir, "Final_insect_numbers.csv")

def extract_rows():
    if not os.path.exists(CSV_FILE):
        print(f"Error: Could not find '{CSV_FILE}'")
        print("Make sure it is in the exact same folder as this script!")
        return

    print(f"Reading data from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)

    output_data = []

    # 2 & 3. Process each target in the EXACT order of your SEARCH_LIST (keeping Duplicates)
    for target in SEARCH_LIST:
        if isinstance(target, int) or (isinstance(target, str) and target.isdigit()):
            mask = df['SubID'] == int(target)
        else:
            mask = df['Filename'] == target
            
        df_filtered = df[mask].copy()

        if df_filtered.empty:
            print(f"Warning: No match found for '{target}'")
            continue

        # Process all matched rows (including Zooniverse batch duplicates)
        for _, row in df_filtered.iterrows():
            subid = str(row['SubID']).replace('.0', '')
            filename = str(row['Filename']).strip()
            total_participants = row.get('TotalParticipants', 15)
            
            # --- Calculate Marked_File ---
            base_name, ext = os.path.splitext(filename)
            marked_file = f"{base_name}_marked_{subid}{ext}"
            
            # --- Calculate Chosen_Label & Consensus_Count ---
            insects_str = str(row.get('InsectIDs', ''))
            labels = re.findall(r"['\"](.*?)['\"]", insects_str)
            
            label_counts = Counter(labels)
            if label_counts:
                chosen_label, consensus_count = label_counts.most_common(1)[0]
            else:
                chosen_label, consensus_count = "Other/Unknown", 0

            # --- Build Output Dictionary with SubID & Filename FIRST ---
            output_data.append({
                'SubID': subid,
                'Filename': filename,
                'Marked_File': marked_file,
                'Chosen_Label': chosen_label,
                'Observed_Labels': '',
                'Sam_labels': '',
                'Split_Labels': 'Other',
                'Concensus_Count': consensus_count,
                'Total_Participants': total_participants
            })

    if not output_data:
        print("No matching rows found. Check your SubIDs or Filenames.")
        return

    # 4. Save to a new CSV
    out_file = os.path.join(script_dir, "Extracted_Gold_Targets.csv")
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(out_file, index=False)
    
    print("\n--- EXTRACTION COMPLETE ---")
    print(f"Successfully found and extracted {len(output_df)} total rows (including duplicates).")
    print(f"Saved formatted data to: {out_file}")

if __name__ == "__main__":
    extract_rows()