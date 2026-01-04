import numpy as np
import os
import sys

# Define your task order
TASK_ORDER = ['NF-BoT-IoT-v2', 'NF-ToN-IoT-v2', 'NF-CSE-CIC-IDS2018-v2', 'NF-UNSW-NB15-v2']

def check_classes():
    repo_root = os.getcwd()
    max_id_found = -1
    unique_labels = set()
    
    print(f"--- SCANNING DATASETS ---")
    
    for task in TASK_ORDER:
        # Construct path: benchmark_data/TaskName/train.npy
        data_path = os.path.join(repo_root, 'benchmark_data', task, 'train.npy')
        
        if not os.path.exists(data_path):
            print(f"[ERROR] Could not find file: {data_path}")
            continue
            
        try:
            # Load only the data we need (lazy loading if possible, but npy loads all)
            data = np.load(data_path)
            
            # Assuming the Label is the LAST column
            labels = data[:, -1]
            
            # Get stats
            unique_in_task = np.unique(labels)
            current_max = int(np.max(labels))
            
            # Update globals
            unique_labels.update(unique_in_task)
            if current_max > max_id_found:
                max_id_found = current_max
                
            print(f"✅ {task}")
            print(f"   - Unique Categories: {len(unique_in_task)}")
            print(f"   - Labels found: {unique_in_task}")
            print(f"   - Max Label ID: {current_max}")
            
        except Exception as e:
            print(f"❌ Error reading {task}: {e}")

    print("\n" + "="*40)
    print("       FINAL CONFIGURATION RESULT")
    print("="*40)
    print(f"Total Unique Categories (Guests): {len(unique_labels)}")
    print(f"Highest Label ID found (Room Key): {max_id_found}")
    print("-" * 40)
    print(f"👉 UPDATE 'train_ewc.py' WITH THIS:")
    print(f"   NUM_CLASSES = {max_id_found + 1}")
    print("="*40)

if __name__ == "__main__":
    check_classes()