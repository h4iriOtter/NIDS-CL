import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURATION
# ==========================================
TASK_ORDER = [
    'NF-BoT-IoT-v2', 
    'NF-ToN-IoT-v2', 
    'NF-CSE-CIC-IDS2018-v2', 
    'NF-UNSW-NB15-v2'
]

def load_global_map(filepath):
    """
    Reads the 'global_class_mapping.txt' file and returns a dictionary
    Format in file: "ID : Attack Name"
    """
    mapping = {}
    if not os.path.exists(filepath):
        print(f"[WARNING] Mapping file not found at: {filepath}")
        return mapping

    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if ":" in line and "ID" not in line: # Skip header
                parts = line.split(":")
                try:
                    # Clean whitespace and convert
                    class_id = int(parts[0].strip())
                    attack_name = parts[1].strip()
                    mapping[class_id] = attack_name
                except ValueError:
                    continue
    return mapping

def check_classes_and_export():
    repo_root = os.getcwd()
    benchmark_dir = os.path.join(repo_root, 'benchmark_data')
    
    # 1. Load the Name Mapping
    mapping_path = os.path.join(benchmark_dir, 'global_class_mapping.txt')
    id_to_name = load_global_map(mapping_path)
    print(f"Loaded {len(id_to_name)} class names from global mapping.")

    max_id_found = -1
    unique_labels_global = set()

    print(f"\n--- SCANNING & GENERATING REPORTS ---")

    for task in TASK_ORDER:
        task_dir = os.path.join(benchmark_dir, task)
        data_path = os.path.join(task_dir, 'train.npy')
        
        if not os.path.exists(data_path):
            print(f"[ERROR] Missing data for {task}: {data_path}")
            continue
            
        print(f"\n>>> Processing: {task}")
        try:
            # 2. Load Data
            data = np.load(data_path)
            labels = data[:, -1].astype(int) # Last column is label
            
            # 3. Calculate Stats
            unique_ids, counts = np.unique(labels, return_counts=True)
            total_samples = len(labels)
            
            # 4. Build DataFrame
            stats_data = []
            for uid, count in zip(unique_ids, counts):
                # Get Name from mapping, or use "Unknown" if missing
                name = id_to_name.get(uid, f"Unknown-ID-{uid}")
                percent = (count / total_samples) * 100
                stats_data.append({
                    "Class ID": uid,
                    "Attack Name": name,
                    "Count": count,
                    "Percentage": f"{percent:.2f}%"
                })
            
            df = pd.DataFrame(stats_data)
            df = df.sort_values(by="Class ID")

            # Update Global Check vars
            current_max = int(np.max(unique_ids))
            unique_labels_global.update(unique_ids)
            if current_max > max_id_found:
                max_id_found = current_max

            # 5. EXPORT CSV
            csv_path = os.path.join(task_dir, 'class_distribution.csv')
            df.to_csv(csv_path, index=False)
            print(f"   ✅ Saved CSV: {csv_path}")

            # 6. EXPORT PICTURE (Bar Chart)
            # Add a numeric column for sorting the plot
            df['SortCount'] = df['Count'] 
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x="Count", y="Attack Name", hue="Attack Name", palette="viridis", legend=False)
            
            plt.title(f"Class Distribution: {task}\nTotal Samples: {total_samples:,}")
            plt.xlabel("Number of Samples")
            plt.ylabel("Attack Type")
            plt.tight_layout()
            
            img_path = os.path.join(task_dir, 'class_distribution.png')
            plt.savefig(img_path, dpi=300)
            plt.close() # Close plot to free memory
            print(f"   ✅ Saved Plot: {img_path}")

        except Exception as e:
            print(f"   ❌ Error processing {task}: {e}")

    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    print("\n" + "="*40)
    print("       FINAL CONFIGURATION RESULT")
    print("="*40)
    print(f"Total Unique Categories Found: {len(unique_labels_global)}")
    print(f"Highest Label ID Found:        {max_id_found}")
    print("-" * 40)
    print(f"👉 UPDATE 'train.py' NUM_CLASSES to:")
    print(f"   NUM_CLASSES = {max_id_found + 1}")
    print("="*40)

if __name__ == "__main__":
    check_classes_and_export()