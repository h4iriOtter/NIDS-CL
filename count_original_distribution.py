import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import Counter
import sys

# ==========================================
# 1. SETUP PATHS
# ==========================================
repo_root = os.getcwd()

# --- INPUT: YOUR D: DRIVE DATASET ---
dataset_dir = r"D:\Dataset\9810e03bba4983da_MOHANAD_A4706\9810e03bba4983da_MOHANAD_A4706\data"
input_filename = "NF-UQ-NIDS-v2.csv"
input_path = os.path.join(dataset_dir, input_filename)

# --- OUTPUT: WHERE TO SAVE THE COUNT REPORT ---
evidence_folder = os.path.join(repo_root, 'experiment_results')
os.makedirs(evidence_folder, exist_ok=True)

report_csv_path = os.path.join(evidence_folder, 'original_dataset_counts.csv')
report_img_path = os.path.join(evidence_folder, 'original_dataset_counts.png')

# ==========================================
# 2. COUNTING PROCESS (Memory Safe)
# ==========================================
if not os.path.exists(input_path):
    print(f"[ERROR] File not found: {input_path}")
    sys.exit()

print(f"--- COUNTING ORIGINAL DATASET ---")
print(f"Reading: {input_path}")
print("This will take a few minutes (reading 12GB)...")

chunk_size = 1_000_000  # 1 Million rows per batch
total_counts = Counter()
total_rows = 0

try:
    with pd.read_csv(input_path, chunksize=chunk_size, low_memory=False) as reader:
        for i, chunk in enumerate(reader):
            # We only need the 'Attack' column to count
            chunk.dropna(subset=['Attack'], inplace=True)
            
            # Update our running total
            counts = chunk['Attack'].value_counts().to_dict()
            total_counts.update(counts)
            
            total_rows += len(chunk)
            print(f"Processed Chunk {i+1} | Total Rows counted: {total_rows:,}", end='\r')

    print(f"\n\n[SUCCESS] Counting complete.")
    
    # ==========================================
    # 3. SAVE RESULTS
    # ==========================================
    print("Generating Report...")
    
    # Convert to DataFrame
    df_counts = pd.DataFrame.from_dict(total_counts, orient='index', columns=['Count'])
    df_counts.index.name = 'Attack Type'
    df_counts = df_counts.sort_values(by='Count', ascending=False)
    
    # Calculate Percentage
    df_counts['Percentage'] = (df_counts['Count'] / total_rows) * 100
    df_counts['Percentage'] = df_counts['Percentage'].map('{:.4f}%'.format)

    # 1. Save CSV
    df_counts.to_csv(report_csv_path)
    print(f"Saved CSV to: {report_csv_path}")

    # 2. Save Image (Table)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df_counts.reset_index().values, 
                     colLabels=df_counts.reset_index().columns, 
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title(f"Original Dataset Distribution\nTotal Rows: {total_rows:,}", fontsize=14, pad=20)
    plt.savefig(report_img_path, bbox_inches='tight', dpi=300)
    print(f"Saved Image to: {report_img_path}")

    print("--- DONE ---")

except Exception as e:
    print(f"\n[CRITICAL ERROR] {e}")