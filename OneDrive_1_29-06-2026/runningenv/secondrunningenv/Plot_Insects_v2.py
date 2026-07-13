import pandas as pd
import json
import argparse
import numpy as np
import os
import re
import ast
from PIL import Image, ImageDraw
from collections import Counter
import math

parser = argparse.ArgumentParser(description='This program plots bugs found by citizen scientists on subimages')
parser.add_argument('inputfile',type=str,
                    help= 'Input .csv file')
parser.add_argument('-makeimages',type=str,default='yes',
                    help= 'Make Images: default = yes')
parser.add_argument('-min_participants', type=int, default=2, 
                    help='Absolute floor: Min number of participants needed (default: 2)')
parser.add_argument('-min_fraction', type=float, default=0.5, 
                    help='Relative floor: Min consensus fraction needed (default: 0.5)')

args = parser.parse_args()

# Reference: column names to choose from

columns_in = ['SubID', 'TotalParticipants','Participants_with_PositiveDetection','NoInsects','NoParticipants','MatchedInsects','InsectIDs','AllXpos', 'AllYpos','Filename'] 


classifications = pd.read_csv(args.inputfile)


if args.makeimages =='yes':
    os.makedirs("ResultImages", exist_ok=True)
f_number_insect=[]
f_most_common_label=[]
f_most_common_count=[]
f_total_no_labels=[]
f_total_participants=[]
f_subIDs=[]
f_filename=[]

mask = classifications["Retirement_reason"].astype(str).str.contains("classification_count", na=False)


total_insects=0

#for i,row in classifications.iterrows():
for i,row in classifications[mask].iterrows():

    Xpos_str = str(row['AllXpos']).replace('\n', ' ').replace(',', ' ').strip("[]")
    Xpos = [float(x) for x in Xpos_str.split() if x.strip()]
    
    Ypos_str = str(row['AllYpos']).replace('\n', ' ').replace(',', ' ').strip("[]")
    Ypos = [float(x) for x in Ypos_str.split() if x.strip()]
    
    Insects_str = str(row['InsectIDs'])
    Insects = re.findall(r"['\"](.*?)['\"]", Insects_str)

    if args.makeimages =='yes':
        imagefile = str(row['Filename']).strip()
        
        if os.path.exists(imagefile):
            img = Image.open(imagefile)
            width, height = img.size
            draw = ImageDraw.Draw(img)

            # scale_x, scale_y = 1.0, 1.0 # ensures no change to scaling
            
            # if str(row['SubID']).strip().replace('.0', '') == '111738486':
            #     print(f"Applying manual scale for mismatched image: {imagefile}")
            #     # manually scale the size difference
            #     scale_x = 554 / 605
            #     scale_y = 554 / 605

            r = 30  # radius        
            colour_map = {
                'Thrip': "blue",
                'Pirate Bug': "green",
                'Other/Unknown': "red"
            }

            for i in range(len(Xpos)):
                x=Xpos[i] #* scale_x
                y=Ypos[i] #* scale_y
    
                colour = colour_map.get(Insects[i][:], "yellow") if i < len(Insects) else "yellow"
                # Draw circle (bounding box)    
                draw.ellipse((x - r, y - r, x + r, y + r), outline=colour, width=2)

            base_name, ext = os.path.splitext(imagefile)
            outfile = f"{base_name}_marked_{row['SubID']}{ext}"
            outfile = os.path.basename(outfile) # Prevents folder path bugs
            img.save(os.path.join("ResultImages", outfile))


    # --- parse labels ---
    labels = re.findall(r"['\"](.*?)['\"]", str(row['InsectIDs']))

    # --- combine all data ---
    points = list(zip(labels, Xpos, Ypos))  # (label, x, y)
    
    # --- group + analyze ---
    counts_str = str(row['NoParticipants'])
    indices_str = str(row['MatchedInsects'])
    
    # Safely parse the fixed string arrays
    try:
        indices = ast.literal_eval(indices_str)
        counts = ast.literal_eval(counts_str.replace(".0",''))  
    except (ValueError, SyntaxError):
        indices = []
        counts = []

    #count total number reporting as an insect of any type (except possible)
    insects = re.findall(r"['\"]([^'\"]+)['\"]", str(row['InsectIDs']))
    number_insects = sum(x in ['Thrip', 'Pirate Bug', 'Other/Unknown'] for x in insects)

    label_counts = Counter(insects)
    if label_counts:
        winning_label, winning_votes = label_counts.most_common(1)[0]
    else:
        winning_label, winning_votes = "None", 0

    true_consensus_fraction = winning_votes / row['TotalParticipants'] if row['TotalParticipants'] > 0 else 0

    if winning_votes > args.min_participants and true_consensus_fraction > args.min_fraction:
        f_number_insect.append(number_insects)
        # ... (the rest of your append statements) ...

    #Counts total number of insects spotted by 3 or more people    
    for l in range(len(counts)):
        if counts[l] > 2:
            total_insects+=1

    for idx, c in enumerate(counts):
        if idx < len(indices):
            group_idx = indices[idx]
            group = [points[j] for j in group_idx if j < len(points)]

            # extract labels
            group_labels = [item[0] for item in group]
                
            # most common insect            
            if len(group_labels)>0:
                most_common = Counter(group_labels).most_common(1)[0]
            else:
                most_common = group_labels
                
            print("\nSubimage", row['SubID'])
            if most_common and isinstance(most_common, tuple):
                winning_votes_group = most_common[1]
                true_consensus_fraction_group = winning_votes_group / row['TotalParticipants'] if row['TotalParticipants'] > 0 else 0
                
                print("Most common:", most_common[0], f"(count={winning_votes_group})", f"(total={len(group_labels)})",f"(fraction={true_consensus_fraction_group})",f"(No of Participants={row['TotalParticipants']})")

                if winning_votes_group > args.min_participants and true_consensus_fraction_group > args.min_fraction:
                    f_most_common_label.append(most_common[0])
                    f_most_common_count.append(most_common[1])
                    f_total_no_labels.append(len(group_labels))
                    f_total_participants.append(row['TotalParticipants'])
                    f_subIDs.append(row['SubID'])
                    f_filename.append(str(row['Filename']).strip())

detections=0
most_count_thrip = sum(1 for arr in f_most_common_label if arr.count("Thrip") > detections)
most_count_pirate = sum(1 for arr in f_most_common_label if arr.count("Pirate Bug") > detections)
most_count_other = sum(1 for arr in f_most_common_label if arr.count("Other/Unknown") > detections)

print("\n\nThe numbers below are counting the most common label for an insect. With a min of "+str(args.min_participants+1)+" participants in agreement and fraction labelling insect with same label is "+str(args.min_fraction)+" \n")
print("Insects to be labelled as the same have to be within 10 pixels in x and y direction")
print("Number of Thips:",most_count_thrip)
print("Number of Pirate Bugs:",most_count_pirate)
print("Number of Other/Unknown Insects:",most_count_other)
print("Total most common bug:",most_count_thrip + most_count_pirate + most_count_other)
print("Total most common bug (excluding unknown insects):",most_count_thrip + most_count_pirate)

print("Total Bugs:",total_insects)

print("\nFilenames and most common insect of all subimages with a min of "+str(args.min_participants+1)+" participants in agreement and fraction labelling insect with same label is "+str(args.min_fraction)+" \n")
for i in range(len(f_filename)):
    print(str(f_filename[i]) + "           " + str(f_most_common_label[i]))

# --- Print images containing Thrips (blue circles) ---
print("\n--- Images containing Thrips (blue circles) ---")
thrip_files = set()
for i in range(len(f_filename)):
    if f_most_common_label[i] == "Thrip":
        thrip_files.add(f_filename[i].replace ('.jpeg', '_marked_' + str(f_subIDs[i]) + '.jpeg'))

print(f"Total Unique Images with Thrips: {len(thrip_files)}")
for f in sorted(list(thrip_files)):
    print(f)
print("-----------------------------------------------\n")

print('The numbers above were computed only using the '+ str(len(classifications[mask]))+' images with retirement reason \'classification count\'.')
    
print("Retirement reason\n", classifications['Retirement_reason'].value_counts())
print("Total number of subjects: ",len(classifications))

df = pd.DataFrame({
    'SubID': f_subIDs,
    'Filename': f_filename,
    'Marked_File': [f.replace('.jpeg', '_marked_' + str(subid) + '.jpeg') for f, subid in zip(f_filename, f_subIDs)],
    'Chosen_Label': f_most_common_label,
    'Consensus_Count': f_most_common_count,
    'Total_Participants': f_total_participants
})

output_filename = f"Filtered_Bugs_Min{args.min_participants}_Frac{args.min_fraction}.csv"
df.to_csv(output_filename, index=False)
print(f"\nSuccessfully saved filtered data to: {output_filename}")