import pandas as pd
import json
import argparse
import numpy as np
import json
import os
import re
import ast
from PIL import Image, ImageDraw
from collections import Counter
import numpy as np
import math

parser = argparse.ArgumentParser(description='This program plots bugs found by citizen scientists on subimages')
parser.add_argument('inputfile',type=str,
                    help= 'Input .csv file')
parser.add_argument('-makeimages',type=str,default='yes',
                    help= 'Make Images: default = yes')
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

mask=classifications["Retirement_reason"]=="{\'classification_count\'}"


total_insects=0

#for i,row in classifications.iterrows():
for i,row in classifications[mask].iterrows():

    #    print (row['SubID'])
    #    print (row['Filename'].replace("\'",'').replace('[','').replace(']',''))
    Xpos= row['AllXpos']
    Xpos = [float(x) for x in Xpos.strip("[]").split()]
    Ypos= row['AllYpos']
    Ypos = [float(x) for x in Ypos.strip("[]").split()]
    #        print(Xpos[0])
    Insects=row['InsectIDs']
    Insects = re.findall(r"'(.*?)'", Insects)

    if args.makeimages =='yes':
        # Load image
        imagefile=row['Filename'].replace("\'",'').replace('[','').replace(']','').replace('{','').replace('}','')
#        print (imagefile)
        if os.path.exists(imagefile):
            img = Image.open(imagefile)
            width, height = img.size
            #        print(width, height)
            # Create drawing object
            draw = ImageDraw.Draw(img)

            r = 30  # radius        
            #        print (Insects)
            colour_map = {
                'Thrip': "blue",
                'Pirate Bug': "green",
                'Other Insect' : 'red',
                'Possible Insect':'purple'
            }

            for i in range(len(Xpos)):
                x=Xpos[i]
                y=Ypos[i]
                #            print (Insects[i])
                colour = colour_map.get(Insects[i][:], "yellow") # default = red
                #            print (colour)
                # Draw circle (bounding box)    
                draw.ellipse((x - r, y - r, x + r, y + r), outline=colour, width=2)

            # Save result
            outfile=imagefile.replace(".jpeg","_marked_"+str(row['SubID'])+".jpeg")
#            print (outfile)         

            img.save("./ResultImages/"+outfile)


    # --- parse labels ---
    labels = re.findall(r"'(.*?)'", row['InsectIDs'])

    # --- combine all data ---
    points = list(zip(labels, Xpos, Ypos))  # (label, x, y)
    
    # --- group + analyze ---
    start = 0
    counts=row['NoParticipants']
    indices=row['MatchedInsects']
    IDs=row['InsectIDs']
    indices=ast.literal_eval(indices)
    counts=ast.literal_eval(counts.replace(".0",''))  # some entries are floats so remove .0 from each to make integer.

    #count total number reporting as an insect of any type (except possible)
    insects=re.findall(r"'([^']+)'", IDs)
    #        number_insects={k: counts[k] for k in ['Thrip', 'Pirate Bug', 'Other Insect']}
    print (insects)
    number_insects = sum(x in ['Thrip', 'Pirate Bug', 'Other Insect','Possible Insect'] for x in insects)
    print(number_insects)
        # Put in to array only if the number of people identifying a particular insect is greater than participants and the fraction is larger than fraction

    participants=2  
    fraction=0.5

    #Counts total number of insects spotted by 3 or more people and with the fraction of people labelling it >50%
    if number_insects > participants and  number_insects/row['TotalParticipants'] >fraction:
        f_number_insect.append(number_insects)

    #Counts total number of insects spotted by 3 or more people    
    for l in range(len(counts)):
        if counts[l] >2:
            total_insects+=1

        
    for i, c in enumerate(counts):
        group_idx = indices[start:start+c]
        group = [points[j] for j in group_idx]

        # extract labels
        group_labels = [item[0] for item in group]
            
        # most common insect            
        if len(group_labels)>0:
            most_common = Counter(group_labels).most_common(1)[0]
        else:
            most_common =group_labels
            
            
        print("\nSubimage", row['SubID'])
#        print("Labels:", group_labels)
        print("Most common:", most_common[0], f"(count={most_common[1]})", f"(total={len(group_labels)})",f"(fraction={most_common[1]/len(group_labels)})",f"(No of Participants={row['TotalParticipants']})")

        


            
        if len(group_labels) > participants and  most_common[1]/len(group_labels) >fraction:
            f_most_common_label.append(most_common[0])
            f_most_common_count.append(most_common[1])
            f_total_no_labels.append(len(group_labels))
            f_total_participants.append(row['TotalParticipants'])
            f_subIDs.append(row['SubID'])
            f_filename.append(row['Filename'])
                             
        start += c


            
#detections=2
#count_thrip = sum(1 for arr in classifications['InsectIDs'] if arr.count("Thrip") > detections)
#count_pirate = sum(1 for arr in classifications['InsectIDs'] if arr.count("Pirate Bug") > detections)
#count_possible = sum(1 for arr in classifications['InsectIDs'] if arr.count("Possible Insect") > detections)
#count_other = sum(1 for arr in classifications['InsectIDs'] if arr.count("Other Insect") > detections)
#
#print("The numbers below are counting the number of times "+str(detections+1)+" or more people record an insect with the same label. The same insect could be repeated here if two other people say its another type of bug\n")
#print("Number of Thips:",count_thrip)
#print("Number of Pirate Bugs:",count_pirate)
#print("Number of Other Insects:",count_other)
#print("Number of Possible Insect:",count_possible)
#print("Total Bugs:",count_thrip+count_pirate+count_other+count_possible)
#print("Total Bugs (excluding:",count_thrip+count_pirate+count_other)
#print("Total insects: ",sum(classifications['NoInsects']))

detections=0
most_count_thrip = sum(1 for arr in f_most_common_label if arr.count("Thrip") > detections)
most_count_pirate = sum(1 for arr in f_most_common_label if arr.count("Pirate Bug") > detections)
most_count_possible = sum(1 for arr in f_most_common_label if arr.count("Possible Insect") > detections)
most_count_other = sum(1 for arr in f_most_common_label if arr.count("Other Insect") > detections)


print("\n\nThe numbers below are counting the most common label for an insect. With a min of "+str(participants+1)+" participants in agreement and fraction labelling insect with same label is "+str(fraction)+" \n")
print("Insects to be labelled as the same have to be within 10 pixels in x and y direction")
print("Number of Thips:",most_count_thrip)
print("Number of Pirate Bugs:",most_count_pirate)
print("Number of Other Insects:",most_count_other)
print("Number of Possible Insect:",most_count_possible)
print("Total Most common Bug:",most_count_thrip+most_count_pirate+most_count_other+most_count_possible)
print("Total Most commong Bug (excluding possible insects):",most_count_thrip+most_count_pirate+most_count_other)

print("Total Bugs:",total_insects)

print("Filenames and most common insect of all subimages with a min of "+str(participants+1)+" participants in agreement and fraction labelling insect with same label is "+str(fraction)+" \n")
for i in range(len(f_filename)):
    print(f_filename[i].replace("\'",'').replace('[','').replace(']','').replace('{','').replace('}','')+"           "+f_most_common_label[i])

print('The numbers above were computed only using the '+ str(len(classifications[mask]))+' images with retirement reason \'classification count\'.')

    
print("Retirement reason ", classifications['Retirement_reason'].value_counts())
print("Total number of subjects: ",len(classifications))
