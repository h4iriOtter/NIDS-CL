import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. SETUP TARGET FOLDERS AND OUTPUT PATH
folders_to_check = [
    r"D:\Dataset\Zoom",
    r"D:\Dataset\Bittorent",
    r"D:\Dataset\Teamviewer"
]

output_dir = r"D:\Projects\NIDS-CL\data"
os.makedirs(output_dir, exist_ok=True)

found_file = None

print("🔍 Searching for your new application CSVs...")

# 2. LOCATE A SAMPLE CSV FILE
for folder in folders_to_check:
    if os.path.exists(folder):
        for file_name in os.listdir(folder):
            if file_name.lower().endswith('.csv'):
                found_file = os.path.join(folder, file_name)
                break
    if found_file:
        break

# 3. EXTRACT AND SAVE FEATURES
if found_file:
    print(f"✅ Success! Found a file to inspect: {found_file}")
    print("⚡ Extracting exact column headers...\n")
    
    # Read only the first row (header) to save memory
    df = pd.read_csv(found_file, nrows=0)
    # Strip whitespace from column names just in case CICFlowMeter left any
    columns = [col.strip() for col in df.columns.tolist()]
    
    # Create a DataFrame for the features
    df_features = pd.DataFrame({
        'Feature_Index': range(1, len(columns) + 1),
        'Feature_Name': columns
    })
    
    # --- SAVE TO CSV ---
    csv_output_path = os.path.join(output_dir, 'app_features_list.csv')
    df_features.to_csv(csv_output_path, index=False)
    print(f"💾 Saved feature list to CSV: {csv_output_path}")
    
    # --- SAVE TO PNG IMAGE ---
    png_output_path = os.path.join(output_dir, 'app_features_list.png')
    
    # Dynamically scale height based on the number of features
    fig, ax = plt.subplots(figsize=(7, len(df_features) * 0.22))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df_features.values, colLabels=df_features.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    
    plt.title(f"Application Dataset Features\n(Source: {os.path.basename(found_file)})", fontsize=12, pad=10)
    plt.savefig(png_output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"🖼️ Saved visual checklist to PNG: {png_output_path}")
    
    # Quick terminal preview
    print("\n📋 Quick Preview (First 15 Columns):")
    for idx, row in df_features.head(15).iterrows():
        print(f" {row['Feature_Index']}. {row['Feature_Name']}")
    print(f"... and {len(columns) - 15} more columns (Check the saved CSV).")
    
else:
    print("❌ Could not find any .csv files. Please make sure the files are extracted inside those folders!")