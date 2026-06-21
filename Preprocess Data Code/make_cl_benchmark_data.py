import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ==========================================
# 1. SETUP PATHS
# ==========================================
repo_root = os.getcwd()
INPUT_CSV = os.path.join(repo_root, 'data', 'NF-UQ-NIDS-v2_10Percent.csv')
OUTPUT_DIR = os.path.join(repo_root, 'benchmark_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Columns to Remove (Cheating/Metadata)
CHEAT_COLS = [
    'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 
    'L4_SRC_PORT', 'L4_DST_PORT', 
    'DNS_QUERY_ID', 'Label', 'Dataset', 'Attack'
]

def process_cl_data():
    print(f"--- CONTINUAL LEARNING PIPELINE (Overflow Fix V2) ---")
    
    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] Could not find: {INPUT_CSV}")
        return

    print(f"[1] Loading Data from {INPUT_CSV}...")
    # Low_memory=False helps preventing mixed type warnings on big files
    df = pd.read_csv(INPUT_CSV, low_memory=False)

    # [2] FILTER TINY CLASSES
    min_samples = 20
    class_counts = df['Attack'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index
    
    df = df[df['Attack'].isin(valid_classes)].copy()

    # [3] GLOBAL MAPPING
    print("[2] Building Global Label Map...")
    le = LabelEncoder()
    le.fit(df['Attack'])
    global_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    
    # Save Map Text
    with open(os.path.join(OUTPUT_DIR, 'global_class_mapping.txt'), 'w') as f:
        f.write("ID : Attack Name\n")
        for name, id_num in global_mapping.items():
            f.write(f"{id_num} : {name}\n")

    # Save Map Image
    map_df = pd.DataFrame(list(global_mapping.items()), columns=['Attack Name', 'Class ID'])
    map_df = map_df.sort_values('Class ID')
    fig, ax = plt.subplots(figsize=(6, len(map_df)*0.4))
    ax.axis('tight'); ax.axis('off')
    table = ax.table(cellText=map_df.values, colLabels=map_df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1, 1.5)
    plt.title("Global Attack Class Mapping", fontsize=14, pad=20)
    plt.savefig(os.path.join(OUTPUT_DIR, 'global_class_mapping.png'), bbox_inches='tight', dpi=300)

    # [4] SPLIT BY DATASET
    unique_datasets = df['Dataset'].unique()
    print(f"[3] Found {len(unique_datasets)} Distinct Datasets (Tasks):")
    print(f"    {unique_datasets}")

    for task_name in unique_datasets:
        print(f"\n    >>> Processing Task: {task_name}")
        task_folder = os.path.join(OUTPUT_DIR, task_name)
        os.makedirs(task_folder, exist_ok=True)
        
        task_df = df[df['Dataset'] == task_name].copy()
        
        # Save Task Stats
        stats = task_df['Attack'].value_counts().reset_index()
        stats.columns = ['Attack Type', 'Count']
        stats['Global ID'] = stats['Attack Type'].map(global_mapping)
        stats.to_csv(os.path.join(task_folder, 'task_stats.csv'), index=False)

        # Prepare Labels
        y = le.transform(task_df['Attack'])
        X_df = task_df.drop(columns=CHEAT_COLS, errors='ignore')
        
        # === CRITICAL FIX START ===
        # 1. Keep as float64 first (Pandas default) to hold massive numbers
        # Fill NaNs with 0 just in case
        X = X_df.values
        X = np.nan_to_num(X) 
        
        # 2. Apply Log Transform to shrink numbers
        # We use np.maximum(X, 0) to avoid log of negative numbers
        X = np.log1p(np.maximum(X, 0))
        
        # 3. NOW it is safe to cast to float32
        X = X.astype(np.float32)
        # === CRITICAL FIX END ===

        # [5] SPLITTING
        try:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.15, stratify=y, random_state=42
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42
            )
        except ValueError:
            print(f"        [WARNING] Stratify failed. Using Random Split.")
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)

        # [6] SCALING
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # [7] SAVE
        np.save(os.path.join(task_folder, 'train.npy'), np.column_stack((X_train, y_train)).astype(np.float32))
        np.save(os.path.join(task_folder, 'val.npy'), np.column_stack((X_val, y_val)).astype(np.float32))
        np.save(os.path.join(task_folder, 'test.npy'), np.column_stack((X_test, y_test)).astype(np.float32))
        
        print(f"        [SUCCESS] Saved: {len(X_train)} Train samples")

    print(f"\n--- ALL DONE ---")
    print(f"Data is ready in: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_cl_data()