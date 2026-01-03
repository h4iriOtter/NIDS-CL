import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

# ==========================================
# 1. SETUP PATHS
# ==========================================
repo_root = os.getcwd()

# --- INPUT ---
dataset_dir = r"D:\Dataset\9810e03bba4983da_MOHANAD_A4706\9810e03bba4983da_MOHANAD_A4706\data"
input_filename = "NF-UQ-NIDS-v2.csv"
input_path = os.path.join(dataset_dir, input_filename)

# --- OUTPUTS ---
data_output_dir = os.path.join(repo_root, 'data')
os.makedirs(data_output_dir, exist_ok=True)
data_output_path = os.path.join(data_output_dir, 'NF-UQ-NIDS-v2_10Percent.csv')

evidence_folder = os.path.join(repo_root, 'experiment_results')
os.makedirs(evidence_folder, exist_ok=True)
report_csv_path = os.path.join(evidence_folder, 'stratification_report.csv')
report_img_path = os.path.join(evidence_folder, 'stratification_evidence.png')

# ==========================================
# 2. CHUNK PROCESSING (The RAM Saver)
# ==========================================
if not os.path.exists(input_path):
    print(f"[ERROR] File not found: {input_path}")
    sys.exit()

chunk_size = 1_000_000  # Process 1 million rows at a time
sampled_chunks = []
total_rows_processed = 0

print(f"--- STARTING CHUNKED PROCESSING ---")
print(f"Input: {input_path}")
print(f"Chunk Size: {chunk_size:,} rows")

try:
    # Read file in chunks
    with pd.read_csv(input_path, chunksize=chunk_size, low_memory=False) as reader:
        for i, chunk in enumerate(reader):
            # 1. Basic Cleanup
            chunk.dropna(subset=['Dataset', 'Attack'], inplace=True)
            
            # 2. Create Stratify Key
            chunk['stratify_key'] = chunk['Dataset'].astype(str) + "_" + chunk['Attack'].astype(str)
            
            # 3. Filter classes with too few samples to split
            # (If a chunk has only 1 example of an attack, we keep it to be safe)
            v_counts = chunk['stratify_key'].value_counts()
            valid_classes = v_counts[v_counts > 1].index
            
            # Split data: safely handle rare classes
            if len(valid_classes) > 0:
                mask = chunk['stratify_key'].isin(valid_classes)
                chunk_clean = chunk[mask]
                
                # Stratified split for this chunk
                try:
                    chunk_sample, _ = train_test_split(
                        chunk_clean, 
                        train_size=0.1, 
                        stratify=chunk_clean['stratify_key'], 
                        random_state=42
                    )
                    sampled_chunks.append(chunk_sample)
                except ValueError:
                    # Fallback: if stratify fails (e.g. rare class distribution), just take random 10%
                    chunk_sample = chunk.sample(frac=0.1, random_state=42)
                    sampled_chunks.append(chunk_sample)
            else:
                # If chunk is super weird/small, just take 10% random
                chunk_sample = chunk.sample(frac=0.1, random_state=42)
                sampled_chunks.append(chunk_sample)

            total_rows_processed += len(chunk)
            print(f"Processed Chunk {i+1} | Total Rows: {total_rows_processed:,}", end='\r')

    print(f"\n\n[SUCCESS] All chunks processed. combining...")
    
    # Combine all small 10% chunks into one DataFrame
    df_final = pd.concat(sampled_chunks, ignore_index=True)
    
    # Remove the helper column
    if 'stratify_key' in df_final.columns:
        df_final.drop(columns=['stratify_key'], inplace=True)

    print(f"Final Dataset Size: {len(df_final):,} rows")

    # ==========================================
    # 3. GENERATE EVIDENCE
    # ==========================================
    print("Generating Evidence Report...")
    # Note: We can't compare perfectly to original count easily without a second pass,
    # so we will just show the distribution of the Sampled data.
    
    sample_counts = df_final['Attack'].value_counts().rename("Sample Count")
    sample_percent = (df_final['Attack'].value_counts(normalize=True) * 100).rename("Distribution (%)")
    
    audit_df = pd.concat([sample_counts, sample_percent], axis=1)
    audit_df['Distribution (%)'] = audit_df['Distribution (%)'].map('{:.2f}%'.format)

    # Save Image
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=audit_df.values, colLabels=audit_df.columns, rowLabels=audit_df.index, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title("Stratified Sample Distribution (10%)", fontsize=14, pad=20)
    plt.savefig(report_img_path, bbox_inches='tight', dpi=300)
    
    audit_df.to_csv(report_csv_path)

    # ==========================================
    # 4. SAVE DATA
    # ==========================================
    print(f"Saving final CSV to: {data_output_path}")
    df_final.to_csv(data_output_path, index=False)
    print("--- DONE ---")

except Exception as e:
    print(f"\n[CRITICAL ERROR] {e}")