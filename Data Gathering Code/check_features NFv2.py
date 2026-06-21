import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. Define paths precisely as requested
input_csv = r"D:\Projects\NIDS-CL\data\NF-UQ-NIDS-v2_10Percent.csv"
output_dir = r"D:\Projects\NIDS-CL\data"
output_csv = os.path.join(output_dir, "extracted_features.csv")
output_png = os.path.join(output_dir, "extracted_features.png")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

print("⚡ Reading headers from the 1.28GB dataset (Memory-Optimized)...")

# nrows=0 guarantees we only scan the column names without loading data bytes
df_header = pd.read_csv(input_csv, nrows=0)
feature_list = df_header.columns.tolist()
total_features = len(feature_list)

print(f"✅ Extracted {total_features} features successfully.")

# ==========================================
# 2. SAVE AS CSV
# ==========================================
# Saving as a clear index-mapped table
features_df = pd.DataFrame({
    'Feature_Index': range(1, total_features + 1),
    'Feature_Name': feature_list
})
features_df.to_csv(output_csv, index=False)
print(f"💾 CSV Saved to: {output_csv}")

# ==========================================
# 3. SAVE AS PNG (Visual Feature Map Grid)
# ==========================================
print("🎨 Rendering features list to PNG...")

# Split the features into 2 columns for a clean visual layout
midpoint = (total_features + 1) // 2
col1 = feature_list[:midpoint]
col2 = feature_list[midpoint:]

# Set up matplotlib figure
fig, ax = plt.subplots(figsize=(12, 10), facecolor='#f7f9fa')
ax.axis('off') # Hide graph lines/ticks

# Title formatting
plt.title(f"NF-UQ-NIDS-v2 Feature Map Schema (Total: {total_features} Columns)", 
          fontsize=16, fontweight='bold', pad=20, color='#1a202c')

# Draw Column 1
y_pos = 0.95
for i, feat in enumerate(col1):
    text_str = f"{i+1:02d}. {feat}"
    ax.text(0.05, y_pos, text_str, fontsize=11, family='monospace',
            transform=ax.transAxes, verticalalignment='top', color='#2d3748')
    y_pos -= 0.042

# Draw Column 2
y_pos = 0.95
for i, feat in enumerate(col2):
    text_str = f"{i+midpoint+1:02d}. {feat}"
    ax.text(0.55, y_pos, text_str, fontsize=11, family='monospace',
            transform=ax.transAxes, verticalalignment='top', color='#2d3748')
    y_pos -= 0.042

# Footer Note
ax.text(0.5, 0.02, "Generated for NIDS-CL Continual Learning Framework Research", 
        fontsize=9, style='italic', horizontalalignment='center', transform=ax.transAxes, color='#a0aec0')

# Save visual representation with high resolution
plt.tight_layout()
plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()

print(f"🖼️  PNG Image Saved to: {output_png}")
print("\n🎉 Done! Ready to review your feature space files.")