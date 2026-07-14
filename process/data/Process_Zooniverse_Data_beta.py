import pandas as pd
import json
import argparse
import numpy as np
import ast
import math

def unique_insects(pairs, total_points):
    parent = np.arange(total_points)
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])  # path compression
        return parent[x]

    def union(a, b):
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    # Perform unions
    for a, b in pairs:
        union(a, b)

    # Group components
    clusters = {}
    for i in range(total_points):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    # Convert to list of arrays
    result = [np.array(group) for group in clusters.values()]
    total = len(result)
    sizes = [len(cluster) for cluster in result]
    
    return result, total, sizes

# Euclidean distance calculation with tolerance at 10px
def find_close_pairs(arr_x, arr_y, tolerance=10):
    result = []                                                                                                                                                                                                                                                                                                                      
    index = []
    n = len(arr_x)
    
    for i in range(n):
        for j in range(i + 1, n):
            distance = math.sqrt((arr_x[i] - arr_x[j])**2 + (arr_y[i] - arr_y[j])**2)
            if distance <= tolerance:
                result.append((arr_x[i], arr_x[j]))
                index.append([i, j])
    return result, index

parser = argparse.ArgumentParser(description='This program analyses the Zooniverse data')
parser.add_argument('inputfile', type=str,
                    help='Input Zooniverse .csv file')
args = parser.parse_args()

outputfile = "Zooniverse_results.csv"
filename_classifications = args.inputfile
filename_output = outputfile

columns_out = [
    'classification_id', 'created_at', 'user_name', 'user_id',
    'workflow_id', 'workflow_version', 'subject_ids', 
    'taskvalue_text', 'taskvalue_survey', 'taskvalue_x', 
    'taskvalue_y', 'taskvalue_count'
]

# Reference: column names to choose from
columns_in = [
    'classification_id', 'user_name', 'user_id', 'user_ip', 
    'workflow_id', 'workflow_name', 'workflow_version', 'created_at', 
    'gold_standard', 'expert', 'metadata', 'annotations', 
    'subject_data', 'subject_ids'
]
       
columns_new = [
    'metadata_json', 'annotations_json', 'subject_data_json', 
    'taskvalue_text', 'taskvalue_survey', 'taskvalue_x', 
    'taskvalue_y', 'taskvalue_count', 'taskvalue_filename', 
    'taskvalue_retirement_reason', 'taskvalue_classifications_count'
]
     
# Load Data
classifications = pd.read_csv(filename_classifications)

# Safe flattening of JSON for raw Zooniverse data
classifications['annotations_json'] = [json.loads(q) if isinstance(q, str) else q for q in classifications.get('annotations', pd.Series([None]*len(classifications)))]
classifications['subject_data_json'] = [json.loads(q) if isinstance(q, str) else q for q in classifications.get('subject_data', pd.Series([None]*len(classifications)))]

taskvalue_survey = []
taskvalue_text = []
taskvalue_x = []
taskvalue_y = []
taskvalue_count = []
taskvalue_filename = []
taskvalue_classifications_count = []
taskvalue_retirement_reason = []

insects = ['Thrip', 'Pirate Bug', 'Other/Unknown']

for i, row in classifications.iterrows():
    data = row.get('subject_data_json', {})
    
    inner = {}
    if isinstance(data, dict) and data:
        first_key = next(iter(data))
        if isinstance(data[first_key], dict):
            inner = data[first_key]
        else:
            inner = data
    
    # Get filename
    fname = ""
    for k in ["Filename", "filename", "image_name", "image_name_1", "image_name_2", "Image", "image", "file_name"]:
        if k in inner and isinstance(inner[k], str):
            fname = inner[k]
            break

    if not fname:
        for k, v in inner.items():
            if isinstance(v, str) and any(v.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                fname = v
                break
                
    taskvalue_filename.append(fname)

    retired = inner.get("retired", {})
    if isinstance(retired, dict):
        taskvalue_retirement_reason.append(str(retired.get("retirement_reason", "")))
        taskvalue_classifications_count.append(str(retired.get("classifications_count", "")))
    else:
        taskvalue_retirement_reason.append("")
        taskvalue_classifications_count.append("")
    
    ann_json = row.get('annotations_json', [])
    
    row_text = ''
    row_survey = ''
    row_x = []
    row_y = []
    row_count = 0

    label_map = {}
    if isinstance(ann_json, list):
        for tt in ann_json:
            if isinstance(tt, dict) and tt.get('task') == 'T0.0.0':
                label_map[tt.get('markIndex')] = tt.get('value')
                
        for t in ann_json:
            if not isinstance(t, dict): continue
            
            if t.get('task') == 'T0':
                if isinstance(t.get('value'), list) and len(t['value']) > 0:
                    x = []
                    y = []
                    toollab = []
                    insectcount = 0
                    
                    for pt_idx in range(len(t['value'])):
                        pt = t['value'][pt_idx]
                        if not isinstance(pt, dict): continue
                        
                        x.append(pt.get('x'))
                        y.append(pt.get('y'))

                        if pt_idx in label_map:
                            try:
                                idx = int(label_map[pt_idx])
                                toollab.append(insects[idx])
                            except (IndexError, ValueError, TypeError):
                                toollab.append("Other/Unknown")
                        else:
                            toollab.append("Other/Unknown")

                        insectcount += 1

                    row_survey = toollab
                    row_x = x
                    row_y = y
                    row_count = insectcount
                    
            if t.get('task') == 'T1':
                val = t.get('value', '')
                row_text = val.rstrip() if isinstance(val, str) else ''

    taskvalue_text.append(row_text)
    taskvalue_survey.append(row_survey)
    taskvalue_x.append(row_x)
    taskvalue_y.append(row_y)
    taskvalue_count.append(row_count)

classifications['taskvalue_text'] = taskvalue_text
classifications['taskvalue_survey'] = taskvalue_survey
classifications['taskvalue_x'] = taskvalue_x
classifications['taskvalue_y'] = taskvalue_y
classifications['taskvalue_count'] = taskvalue_count
classifications['taskvalue_filename'] = taskvalue_filename
classifications['taskvalue_retirement_reason'] = taskvalue_retirement_reason
classifications['taskvalue_classifications_count'] = taskvalue_classifications_count


# --- CLUSTERING ALGORITHM ---
res = classifications   
unique_subids = res['subject_ids'].dropna().unique()

total_positives = 0
for j in range(len(res)):
    if res["taskvalue_count"].iloc[j] != 0:
        total_positives += 1

# Loop through subids to find insects
f_total_perID = []
f_total_positivesperID = []
f_xpos = []
f_ypos = []
f_unique_insects = []
f_no_participants_per_unique_insect = []
f_insect_pos_match = []
f_insect_label = []
f_filename = []
f_class_count = []
f_ret_reason = []

for subid in unique_subids:
    total_positives_perID = 0
    total_perID = 0
    x_values = []
    y_values = []
    insect_label = []
    filename = []
    class_count = []
    ret_reason = []
    
    sub_df = classifications[classifications['subject_ids'] == subid]
    
    for _, row in sub_df.iterrows():
        filename.append(row['taskvalue_filename'])
        ret_reason.append(row['taskvalue_retirement_reason'])
        class_count.append(row['taskvalue_classifications_count'])
        
        if row["taskvalue_count"] != 0:
            x_values.append(row['taskvalue_x'])
            y_values.append(row['taskvalue_y'])
            insect_label.append(row['taskvalue_survey'])
            total_positives_perID += 1
            total_perID += 1
        elif row["taskvalue_count"] == 0:
            total_perID += 1

    f_total_perID.append(total_perID)
    f_total_positivesperID.append(total_positives_perID)

    x_cut = np.array(sum(x_values, []))
    y_cut = np.array(sum(y_values, []))
    insect_label_cut = np.array(sum(insect_label, []))
    
    # Filter distinct valid strings
    valid_filenames = [f for f in set(filename) if str(f).strip()]
    filename = valid_filenames[0] if valid_filenames else ''
    
    valid_ret = [r for r in set(ret_reason) if str(r).strip()]
    ret_reason = valid_ret[0] if valid_ret else ''
    
    valid_class = [c for c in set(class_count) if str(c).strip()]
    class_count = valid_class[0] if valid_class else ''

    if len(x_cut) > 1:
        condition, index = find_close_pairs(x_cut, y_cut, tolerance=25)
        if len(index) > 0:
            insect_matches, total_insects, no_citizens_finding_insect = unique_insects(index, len(x_cut))
            
            # Conversions to avoid string errors in the output
            insect_matches_str = str([[int(item) for item in subarr] for subarr in insect_matches])
            
            f_xpos.append(x_cut)
            f_ypos.append(y_cut)
            f_insect_label.append(insect_label_cut)
            f_unique_insects.append(total_insects)
            f_no_participants_per_unique_insect.append(no_citizens_finding_insect)
            f_insect_pos_match.append(insect_matches_str)
            f_filename.append(filename)
            f_ret_reason.append(ret_reason)
            f_class_count.append(class_count)
        else:
            f_xpos.append(x_cut)
            f_ypos.append(y_cut)
            f_insect_label.append(insect_label_cut)
            f_unique_insects.append(len(x_cut))
            f_no_participants_per_unique_insect.append(str([1] * len(x_cut)))
            f_insect_pos_match.append(str([[i] for i in range(len(x_cut))]))
            f_filename.append(filename)
            f_ret_reason.append(ret_reason)
            f_class_count.append(class_count)
    else:
        f_xpos.append(x_cut)
        f_ypos.append(y_cut)
        f_insect_label.append(insect_label_cut)
        f_unique_insects.append(len(x_cut))
        f_no_participants_per_unique_insect.append(str([1] * len(x_cut)))
        f_insect_pos_match.append(str([[i] for i in range(len(x_cut))]))
        f_filename.append(filename)
        f_ret_reason.append(ret_reason)
        f_class_count.append(class_count)

        
final_results = pd.DataFrame()
final_results['NoInsects'] = f_unique_insects
final_results['NoParticipants'] = f_no_participants_per_unique_insect
final_results['AllXpos'] = f_xpos
final_results['AllYpos'] = f_ypos
final_results['InsectIDs'] = f_insect_label
final_results['SubID'] = unique_subids
final_results['SubID'] = final_results['SubID'].astype(int)
final_results['MatchedInsects'] = f_insect_pos_match
final_results['TotalParticipants'] = f_total_perID
final_results['Participants_with_PositiveDetection'] = f_total_positivesperID
final_results['Retirement_reason'] = f_ret_reason
final_results['Filename'] = f_filename
final_results['Classification_count'] = f_class_count


f_columns_out = [
    'SubID', 'TotalParticipants', 'Participants_with_PositiveDetection', 
    'NoInsects', 'NoParticipants', 'MatchedInsects', 'InsectIDs', 
    'AllXpos', 'AllYpos', 'Filename', 'Retirement_reason', 'Classification_count'
]

# Output 1: Clustered dataset
foutput = final_results[f_columns_out]
foutput.to_csv('Final_insect_numbers.csv', index=False)

# Output 2: Compact Zooniverse dataset
final_zooniverse_cols = [col for col in columns_out if col in classifications.columns]
output = classifications[final_zooniverse_cols]
output.to_csv(filename_output, index=False)