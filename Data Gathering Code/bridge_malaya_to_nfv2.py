import os
import pandas as pd
import numpy as np

# 1. SETUP THE FOLDERS AND LABELS
app_folders = {
    "App_Zoom": r"D:\Dataset\Zoom",
    "App_BitTorrent": r"D:\Dataset\Bittorent",
    "App_Teamviewer": r"D:\Dataset\Teamviewer"
}

output_csv = r"D:\Projects\NIDS-CL\data\Task4_Malaya_Formatted.csv"

# 2. DEFINE THE EXACT 46 COLUMNS YOUR PREPROCESSOR EXPECTS
NF_COLS = [
    'IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO',
    'IN_BYTES', 'IN_PKTS', 'OUT_BYTES', 'OUT_PKTS', 'TCP_FLAGS', 'CLIENT_TCP_FLAGS',
    'SERVER_TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS', 'DURATION_IN', 'DURATION_OUT',
    'MIN_TTL', 'MAX_TTL', 'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT', 'MIN_IP_PKT_LEN',
    'MAX_IP_PKT_LEN', 'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES',
    'RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_IN_PKTS', 'RETRANSMITTED_OUT_BYTES',
    'RETRANSMITTED_OUT_PKTS', 'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT',
    'NUM_PKTS_UP_TO_128_BYTES', 'NUM_PKTS_128_TO_256_BYTES', 'NUM_PKTS_256_TO_512_BYTES',
    'NUM_PKTS_512_TO_1024_BYTES', 'NUM_PKTS_1024_TO_1514_BYTES', 'TCP_WIN_MAX_IN',
    'TCP_WIN_MAX_OUT', 'ICMP_TYPE', 'ICMP_IPV4_TYPE', 'DNS_QUERY_ID', 'DNS_QUERY_TYPE',
    'DNS_TTL_ANSWER', 'FTP_COMMAND_RET_CODE', 'Label', 'Attack', 'Dataset'
]

# 3. DIRECT MAPPING FROM CICFLOWMETER TO NETFLOW
mapping = {
    'src_ip': 'IPV4_SRC_ADDR',
    'src_port': 'L4_SRC_PORT',
    'dst_ip': 'IPV4_DST_ADDR',
    'dst_port': 'L4_DST_PORT',
    'protocol': 'PROTOCOL',
    'totlen_fwd_pkts': 'IN_BYTES',
    'tot_fwd_pkts': 'IN_PKTS',
    'totlen_bwd_pkts': 'OUT_BYTES',
    'tot_bwd_pkts': 'OUT_PKTS',
    'pkt_len_max': 'LONGEST_FLOW_PKT',
    'pkt_len_min': 'SHORTEST_FLOW_PKT',
    'init_fwd_win_byts': 'TCP_WIN_MAX_IN',
    'init_bwd_win_byts': 'TCP_WIN_MAX_OUT'
}

print("🚀 Starting Data Bridge: CICFlowMeter -> NF-v2 Format")
all_dataframes = []

for attack_name, folder_path in app_folders.items():
    if not os.path.exists(folder_path):
        print(f"⚠️ Skipping {attack_name}: Folder {folder_path} not found.")
        continue
        
    print(f"\n📂 Processing {attack_name} from {folder_path}...")
    
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            print(f"   -> Reading {file_name}")
            
            # Read CSV
            df_app = pd.read_csv(file_path, low_memory=False)
            
            # Clean column names (strip spaces)
            df_app.columns = [c.strip() for c in df_app.columns]
            
            # Create a blank DataFrame with exactly 46 columns filled with 0
            df_mapped = pd.DataFrame(0, index=np.arange(len(df_app)), columns=NF_COLS)
            
            # Map the overlapping features
            for cic_col, nf_col in mapping.items():
                if cic_col in df_app.columns:
                    df_mapped[nf_col] = df_app[cic_col]
            
            # Special Math Calculation: CIC flow_duration is in microseconds. NF is milliseconds.
            if 'flow_duration' in df_app.columns:
                df_mapped['FLOW_DURATION_MILLISECONDS'] = df_app['flow_duration'] / 1000.0
                
            # Set the Metadata columns so your Preprocessor handles it perfectly
            df_mapped['Label'] = 0                # 0 because it's benign traffic
            df_mapped['Attack'] = attack_name     # E.g., 'App_Zoom'
            df_mapped['Dataset'] = "Malaya_Task4" # This triggers the unique task folder in your code!
            
            all_dataframes.append(df_mapped)

if all_dataframes:
    print("\n⏳ Combining all files into a single master Task 4 dataset...")
    final_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Save the output
    final_df.to_csv(output_csv, index=False)
    print(f"✅ SUCCESS! Formatted dataset saved to: {output_csv}")
    print(f"📊 Total Rows: {len(final_df)}")
else:
    print("❌ No data was processed. Check your file paths.")