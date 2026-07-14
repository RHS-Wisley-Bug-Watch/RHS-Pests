import pandas as pd
import numpy as np
import json
import math
import argparse

def unique_insects(pairs, total_points):
    parent = np.arange(total_points)
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_a] = root_b
    for a, b in pairs:
        union(a, b)
    clusters = {}
    for i in range(total_points):
        root = find(i)
        clusters.setdefault(root, []).append(i)
    result = [np.array(group) for group in clusters.values()]
    return result, len(result), [len(cluster) for cluster in result]

def find_close_pairs(arr_x, arr_y, tolerance):
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


# Replace these SubIDs and counts with your manual visual checks
ground_truth = {
    111738550: 3,
    111813347: 2,
    111738558: 4,
    111738505: 3,
    111738833: 2,
}

tolerances_to_test = [3, 5, 8, 10, 12, 15, 20, 25, 30]


def run_sweep(input_csv):
    print("Loading Zooniverse data...")
    df = pd.read_csv(input_csv)
    
    # Pre-parse the JSON data specifically for our ground truth images to save time
    df['annotations_json'] = [json.loads(q) if isinstance(q, str) else q for q in df.get('annotations', pd.Series([None]*len(df)))]
    
    # Filter only the rows corresponding to our test images
    test_df = df[df['subject_ids'].isin(ground_truth.keys())]
    
    # Store the extracted coordinates for each image so we only parse JSON once
    image_data = {}
    
    print("Extracting coordinates for test images...")
    for subid in ground_truth.keys():
        sub_df = test_df[test_df['subject_ids'] == subid]
        all_x = []
        all_y = []
        
        for _, row in sub_df.iterrows():
            ann_json = row.get('annotations_json', [])
            if isinstance(ann_json, list):
                for t in ann_json:
                    if isinstance(t, dict) and t.get('task') == 'T0':
                        if isinstance(t.get('value'), list):
                            for pt in t['value']:
                                if isinstance(pt, dict) and 'x' in pt and 'y' in pt:
                                    all_x.append(pt['x'])
                                    all_y.append(pt['y'])
                                    
        image_data[subid] = (np.array(all_x), np.array(all_y))

    print("\n--- BEGINNING TOLERANCE SWEEP ---\n")
    
    results_scoreboard = []

    for tol in tolerances_to_test:
        total_absolute_error = 0
        print(f"Testing Tolerance: {tol}px")
        
        for subid, true_count in ground_truth.items():
            x_cut, y_cut = image_data[subid]
            
            if len(x_cut) > 1:
                _, index = find_close_pairs(x_cut, y_cut, tolerance=tol)
                
                if len(index) > 0:
                    _, calculated_insects, _ = unique_insects(index, len(x_cut))
                else:
                    calculated_insects = len(x_cut)
            else:
                calculated_insects = len(x_cut)
                
            # Calculate how far off the algorithm is from reality
            error = abs(calculated_insects - true_count)
            total_absolute_error += error
            
        avg_error = total_absolute_error / len(ground_truth)
        results_scoreboard.append((tol, total_absolute_error, avg_error))
        
        print(f"  > Total Error across all test images: {total_absolute_error} (Avg: {avg_error:.2f} off per image)")

    print("\n--- FINAL OPTIMIZATION RESULTS ---")
    print("Tolerance | Total Error | Avg Error per Image")
    print("-" * 45)
    
    # Sort the scoreboard by the lowest error to find the winner
    results_scoreboard.sort(key=lambda x: x[1])
    
    for rank, (tol, total_err, avg_err) in enumerate(results_scoreboard):
        marker = "WINNER" if rank == 0 else ""
        print(f"   {tol:2d}px   |     {total_err:3d}     |      {avg_err:.2f} {marker}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a sweep to find the optimal clustering tolerance.')
    parser.add_argument('inputfile', type=str, help='Raw Zooniverse .csv file')
    args = parser.parse_args()
    
    run_sweep(args.inputfile)