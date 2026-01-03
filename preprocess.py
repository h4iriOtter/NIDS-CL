import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

# ==========================================
# 1. SETUP PATHS
# ==========================================
# A. External Storage (For the huge data files)
home = str(Path.home())
downloads_folder = os.path.join(home, 'Downloads')

input_filename = 'NF-UQ-NIDS-v2.csv'
data_output_filename = 'NF-UQ-NIDS-v2_10Percent.csv'

input_path = os.path.join(downloads_folder, input_filename)
data_output_path = os.path.join(downloads_folder, data_output_filename)

# B. Internal Repo Storage (For the Evidence/Results)
# We get the folder where this script is currently running
repo_root = os.getcwd()
evidence_folder = os.path.join(repo_root, 'experiment_results')

# Create the folder if it doesn't exist
os.makedirs(evidence_folder, exist_ok=True)

report_csv_path = os.path.join(evidence_folder, 'stratification_report.csv')
report_img_path = os.path.join(evidence_folder, 'stratification_evidence.png')

print(f"--- STRATIFICATION & ORGANIZED EVIDENCE ---")
print(f"Data Input:      {input_path}")
print(f"Data Output:     {data_output_path} (Save in Downloads)")
print(f"Evidence Output: {evidence_folder} (Save Inside Repo)")

# ==========================================
# 2. LOAD & PROCESS
# ==========================================
if not os.path.exists(input_path):
    print(f"\n[ERROR] File not found: {input_path}")
    sys.exit()

print(f"\n[1/6] Loading full dataset...")
df = pd.read_csv(input_path, low_memory=False)
df.dropna(subset=['Dataset', 'Attack'], inplace=True)

print(f"\n[2/6] Auditing original distribution...")
original_counts = df['Attack'].value_counts().rename("Original Count")

print(f"\n[3/6] Performing 10% Stratified Split...")
stratify_key = df['Dataset'].astype(str) + "_" + df['Attack'].astype(str)
valid_classes = stratify_key.value_counts()[stratify_key.value_counts() > 1].index
mask = stratify_key.isin(valid_classes)

df_clean = df[mask].copy()
stratify_key_clean = stratify_key[mask]

df_sample, _ = train_test_split(
    df_clean, train_size=0.1, stratify=stratify_key_clean, random_state=42
)

# ==========================================
# 3. BUILD THE AUDIT TABLE
# ==========================================
print(f"\n[4/6] Building Statistics Table...")
new_counts = df_sample['Attack'].value_counts().rename("Sample Count")
audit_df = pd.concat([original_counts, new_counts], axis=1).fillna(0).astype(int)

audit_df['Retention (%)'] = (audit_df['Sample Count'] / audit_df['Original Count']) * 100
audit_df['Retention (%)'] = audit_df['Retention (%)'].map('{:.2f}%'.format)
audit_df = audit_df.sort_values(by='Original Count', ascending=False)

# ==========================================
# 4. GENERATE & SAVE IMAGE
# ==========================================
print(f"\n[5/6] Saving Image to '{evidence_folder}'...")

fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=audit_df.values,
                 colLabels=audit_df.columns,
                 rowLabels=audit_df.index,
                 cellLoc='center',
                 loc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.title("Stratified Sampling Audit: NF-UQ-NIDS-v2", fontsize=14, pad=20)

# Save into the REPO folder
plt.savefig(report_img_path, bbox_inches='tight', dpi=300)

# ==========================================
# 5. SAVE DATA
# ==========================================
print(f"\n[6/6] Saving Data files...")
# Save CSV report to REPO folder
audit_df.to_csv(report_csv_path)

# Save Huge Data to DOWNLOADS folder
df_sample.to_csv(data_output_path, index=False)

print("\n--- DONE! ---")
print(f"Evidence saved in: {evidence_folder}")
print(f"Stratified Data saved in: {data_output_path}")