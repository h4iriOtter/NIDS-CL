import pandas as pd
import os
from sklearn.model_selection import train_test_split
from pathlib import Path

# ==========================================
# 1. SETUP PATHS (AUTO-DETECT DOWNLOADS)
# ==========================================
# This automatically finds your Windows Downloads folder
home = str(Path.home())
downloads_folder = os.path.join(home, 'Downloads')

# Define input and output filenames
input_filename = 'NF-UQ-NIDS-v2.csv'
output_filename = 'NF-UQ-NIDS-v2_10Percent.csv'

# Full paths
input_path = os.path.join(downloads_folder, input_filename)
output_path = os.path.join(downloads_folder, output_filename)

print(f"--- LOCAL STRATIFICATION SCRIPT ---")
print(f"Target Input: {input_path}")
print(f"Target Output: {output_path}")

# ==========================================
# 2. LOAD DATA
# ==========================================
if not os.path.exists(input_path):
    print(f"\n[ERROR] File not found at: {input_path}")
    print("Please make sure 'NF-UQ-NIDS-v2.csv' is actually in your Downloads folder.")
    exit()

print(f"\nLoading {input_filename}... (This uses your local RAM)")

# We use standard loading (no type optimization) since you have 32GB RAM
# low_memory=False prevents warnings for mixed types
df = pd.read_csv(input_path, low_memory=False)

print(f"Successfully loaded {len(df):,} rows.")

# ==========================================
# 3. PREPARE STRATIFICATION
# ==========================================
# We need to drop rows where critical columns are NaN to avoid errors
df.dropna(subset=['Dataset', 'Attack'], inplace=True)

# Create the key to ensure we get 10% from EVERY dataset and EVERY attack type
print("Creating stratification keys...")
stratify_key = df['Dataset'].astype(str) + "_" + df['Attack'].astype(str)

# Filter out classes with only 1 sample (cannot be split)
class_counts = stratify_key.value_counts()
valid_classes = class_counts[class_counts > 1].index
mask = stratify_key.isin(valid_classes)

# Apply filter
df_clean = df[mask].copy()
stratify_key_clean = stratify_key[mask]

dropped = len(df) - len(df_clean)
if dropped > 0:
    print(f"Dropped {dropped} rows (singletons) to ensure valid split.")

# ==========================================
# 4. EXECUTE SPLIT (10%)
# ==========================================
print("Splitting data (Taking 10%)...")

# train_size=0.1 means we KEEP 10%
# We perform the split and ignore the 90%
df_sample, _ = train_test_split(
    df_clean,
    train_size=0.1,  # 10%
    stratify=stratify_key_clean,
    random_state=42
)

# ==========================================
# 5. SAVE RESULT
# ==========================================
print(f"Saving to {output_path}...")
df_sample.to_csv(output_path, index=False)

print("\n--- PROCESS COMPLETE ---")
print(f"Original Count: {len(df):,}")
print(f"Final Count:    {len(df_sample):,}")
print(f"Saved file is ready in your Downloads folder.")
