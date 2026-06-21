import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. SETUP TARGET AND OUTPUT PATHS
input_large_csv = r"D:\Dataset\malayagt1480features.csv"
output_dir = r"D:\Projects\NIDS-CL\data"
output_sample_csv = os.path.join(output_dir, "malaya_1percent_sample.csv")
output_features_csv = os.path.join(output_dir, "malaya_features.csv")
output_features_png = os.path.join(output_dir, "malaya_features.png")

# Ensure the project output folder exists
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(input_large_csv):
    print(f"❌ Error: Cannot find the 3GB file at: {input_large_csv}")
    print("Please verify the filename and file path match your D: drive precisely.")
else:
    print("⚡ Step 1: Extracting column headers to mapping schema...")
    # Read only row zero to grab feature names cleanly
    df_header = pd.read_csv(input_large_csv, nrows=0)
    features = [f.strip() for f in df_header.columns.tolist()]
    total_features = len(features)
    
    # Save the feature list independently
    df_feat_log = pd.DataFrame({
        'Feature_Index': range(1, total_features + 1),
        'Feature_Name': features
    })
    df_feat_log.to_csv(output_features_csv, index=False)
    print(f"💾 Feature checklist saved to: {output_features_csv}")
    
    print("\n⚡ Step 2: Sampling 1% of rows from the 3GB file (Stream Processing)...")
    # Using chunksize allows us to process a 3GB file smoothly without RAM issues
    chunk_size = 50000
    is_first_chunk = True
    sampled_rows_count = 0
    
    for chunk in pd.read_csv(input_large_csv, chunksize=chunk_size, low_memory=False):
        # Take a random 1% sample of the current chunk
        sampled_chunk = chunk.sample(frac=0.01, random_state=42)
        sampled_rows_count += len(sampled_chunk)
        
        # Append rows sequentially to our target project folder CSV
        if is_first_chunk:
            sampled_chunk.to_csv(output_sample_csv, index=False, mode='w')
            is_first_chunk = False
        else:
            sampled_chunk.to_csv(output_sample_csv, index=False, mode='a', header=False)
            
    print(f"✅ Successfully created a 1% subset CSV ({sampled_rows_count} rows) at:")
    print(f"   👉 {output_sample_csv}")
    
    # --- Step 3: Render Matplotlib Feature PNG Checklist ---
    print("\n⚡ Step 3: Rendering visual feature chart...")
    fig, ax = plt.subplots(figsize=(7, total_features * 0.22))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df_feat_log.values, colLabels=df_feat_log.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    
    plt.title("Malaya Dataset (CICFlowMeter) Features", fontsize=14, pad=10)
    plt.savefig(output_features_png, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"🖼️ Feature visual saved to: {output_features_png}")

    # Step 4: Quick terminal printout
    print("\n📋 Here are the first 15 features of your Malaya file:")
    for idx, feature in enumerate(features[:15]):
        print(f" {idx+1}. {feature}")
    print(f"... and {total_features - 15} more columns (Saved in your CSV).")