import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
import wandb
import time

# --- CUSTOM IMPORTS ---
from model import BigNet
from utils import NaturalSortTextLogger, calculate_smart_f1
import matplotlib.pyplot as plt

# --- AVALANCHE IMPORTS ---
# Benchmark: The container that holds your streams of data (Train, Test, Val)
from avalanche.benchmarks import benchmark_from_datasets
from avalanche.benchmarks.utils import AvalancheDataset
# Naive: The standard Supervised Training strategy (we add plugins to make it smart)
from avalanche.training.supervised import Naive
# Plugins: The modular components that inject logic (Replay, Logging, Scheduling)
from avalanche.training.plugins import ReplayPlugin, EvaluationPlugin, LRSchedulerPlugin, SupervisedPlugin
# Metrics: Standard CL metrics provided by Avalanche
from avalanche.evaluation.metrics import (
    forgetting_metrics, accuracy_metrics, loss_metrics 
)
from avalanche.logging import InteractiveLogger, WandBLogger
from avalanche.training.storage_policy import ReservoirSamplingBuffer

# ==========================================
# CONFIGURATION
# ==========================================
# [UPDATED] Added Malaya_Task4 to the Continual Learning stream!
TASK_ORDER = ['NF-UNSW-NB15-v2', 'NF-CSE-CIC-IDS2018-v2', 'NF-ToN-IoT-v2', 'Malaya_Task4']
TRAIN_BATCH_SIZE = 4096  # Speed up training (GPU friendly)
EVAL_BATCH_SIZE = 4096  # Higher batch size for faster evaluation
MEM_SIZE = 7500    # The size of the Replay Buffer (Memory of past tasks)
EPOCHS = 30         # 30 Epochs is sufficient to avoid overfitting
EVAL_FREQ = 2       # Evaluate every 2 epoch to generate high-resolution learning curves
NUM_CLASSES = 23    # [UPDATED] 23 Classes (Includes the 3 new Applications)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_FILE = 'results_NF+Malaya_Run1_4GB_2MB.txt' # [UPDATED] Log file name

# ==========================================
# PART 1: OPTIMIZED PLUGIN 
# ==========================================
class SmartContextManager(SupervisedPlugin):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.seen_classes_so_far = set()
        
        # Buffers for manual calculation
        self.local_preds = []
        self.local_true = []
        self.global_preds = []
        self.global_true = []

    def before_training_exp(self, strategy, **kwargs):
        print("\n[SmartContext] Analyzing dataset statistics...")
        try:
            y_data = torch.as_tensor(strategy.experience.dataset.targets).cpu()
        except:
            loader = DataLoader(strategy.experience.dataset, batch_size=20000, num_workers=4)
            y_data = torch.cat([batch[1].cpu() for batch in loader])
        
        y_np = y_data.numpy()
        self.seen_classes_so_far.update(np.unique(y_np).tolist())
        print(f"   [Metrics] Seen Classes: {sorted(list(self.seen_classes_so_far))}")
        
        # Calculate Weights (Cap=50.0)
        classes, counts = np.unique(y_np, return_counts=True)
        if len(classes) > 0:
            raw_weights = sum(counts) / (len(classes) * counts)
            full_weights = torch.ones(self.num_classes).to(strategy.device)
            for cls, w in zip(classes, raw_weights):
                if cls < self.num_classes:
                    full_weights[int(cls)] = min(float(w) ** 0.5, 50.0) # Sqrt + Cap
            strategy._criterion = nn.CrossEntropyLoss(weight=full_weights, label_smoothing=0.1)
            print(f"   [Weights] Updated & Smoothed for {len(classes)} classes.")

    def before_eval(self, strategy, **kwargs):
        self.global_preds = []
        self.global_true = []

    def before_eval_exp(self, strategy, **kwargs):
        self.local_preds = []
        self.local_true = []

    def after_eval_iteration(self, strategy, **kwargs):
        preds = torch.argmax(strategy.mb_output, dim=1).cpu().numpy()
        true_y = strategy.mb_y.cpu().numpy()
        
        self.local_preds.append(preds)
        self.local_true.append(true_y)
        self.global_preds.append(preds)
        self.global_true.append(true_y)

    def after_eval_exp(self, strategy, **kwargs):
        curr_id = strategy.experience.current_experience
        
        y_pred = np.concatenate(self.local_preds)
        y_true = np.concatenate(self.local_true)
        
        cm = sklearn_confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
        f1_local, prec, rec = calculate_smart_f1(cm, strict=True)
        
        wandb.log({
            f"F1_Score/Task_{curr_id}_Local": f1_local,
            f"Precision/Task_{curr_id}": prec,
            f"Recall/Task_{curr_id}": rec
        })
        msg = f"\t[Task {curr_id}] Local F1: {f1_local:.4f} | Recall: {rec:.4f}"
        print(msg)
        with open("output.log", 'a') as f: f.write(msg + "\n")

    def after_eval(self, strategy, **kwargs):
        if len(self.global_preds) > 0:
            y_pred = np.concatenate(self.global_preds)
            y_true = np.concatenate(self.global_true)
            
            cm = sklearn_confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
            f1_global, _, _ = calculate_smart_f1(cm, strict=False)
            
            wandb.log({"F1_Score/System_Global": f1_global})

            print("\n" + "="*40)
            print(f"   SYSTEM CONFUSION MATRIX ({self.num_classes}x{self.num_classes})")
            print("="*40)
            with np.printoptions(linewidth=200, edgeitems=23, formatter={'float': '{: 0.0f}'.format}):
                print(cm)
            print("="*40 + "\n")

            try:
                fig, ax = plt.subplots(figsize=(14, 14))
                cax = ax.matshow(cm, cmap='Blues')
                fig.colorbar(cax)
                
                ax.set_title(f"Confusion Matrix (System)")
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
                ax.set_xticks(np.arange(self.num_classes))
                ax.set_yticks(np.arange(self.num_classes))
                
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        count = int(cm[i, j])
                        if count > 0: 
                            ax.text(j, i, str(count), ha="center", va="center",
                                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
                
                wandb.log({"Confusion_Matrix_Image": wandb.Image(fig)})
                plt.close(fig)
            except Exception as e:
                print(f"[Warning] Failed to generate Heatmap: {e}")

            msg = f"\t[System] Global F1: {f1_global:.4f} (Stability)"
            print(msg)
            with open("output.log", 'a') as f: f.write(msg + "\n")

# ==========================================
# PART 3: DATA PIPELINE
# ==========================================
class TaskTensorDataset(TensorDataset):
    def __init__(self, x, y, task_id):
        super().__init__(x, y)
        self.task_id = task_id
        self.targets = y 
    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, y, self.task_id

def load_and_scale_data():
    repo_root = os.getcwd()
    train_datasets, test_datasets, val_datasets = [], [], []
    input_dim = 0 
    
    global_scaler = StandardScaler()

    print("\n[Data] Loading and Standardizing...")
    
    for i, task_name in enumerate(TASK_ORDER):
        # [UPDATED] Points to the new folder with the Malaya data
        data_dir = os.path.join(repo_root, 'benchmark_datawithmalaya', task_name)
        print(f"   Processing {task_name}...")
        
        try:
            train_np = np.load(os.path.join(data_dir, 'train.npy'))
            val_np   = np.load(os.path.join(data_dir, 'val.npy'))
            test_np  = np.load(os.path.join(data_dir, 'test.npy'))

            def clean_numpy(arr):
                if np.isnan(arr).any() or np.isinf(arr).any():
                    arr = arr[~np.isnan(arr).any(axis=1)] 
                    arr = arr[~np.isinf(arr).any(axis=1)]
                return arr

            train_np = clean_numpy(train_np)
            val_np   = clean_numpy(val_np)
            test_np  = clean_numpy(test_np)

            train_x, train_y = train_np[:, :-1], train_np[:, -1]
            val_x, val_y     = val_np[:, :-1], val_np[:, -1]
            test_x, test_y   = test_np[:, :-1], test_np[:, -1]
            
            if input_dim == 0: input_dim = train_x.shape[1]

            if i == 0:
                print("      [Scaler] Fitting Global Scaler on Task 1...")
                global_scaler.fit(train_x)
            
            train_x_scaled = global_scaler.transform(train_x)
            val_x_scaled   = global_scaler.transform(val_x)
            test_x_scaled  = global_scaler.transform(test_x)

            tx_train = torch.from_numpy(train_x_scaled).float()
            ty_train = torch.from_numpy(train_y).long()
            
            tx_val   = torch.from_numpy(val_x_scaled).float()
            ty_val   = torch.from_numpy(val_y).long()
            
            tx_test  = torch.from_numpy(test_x_scaled).float()
            ty_test  = torch.from_numpy(test_y).long()

            def create_ds(x, y, tid):
                dataset = TaskTensorDataset(x, y, task_id=tid)
                ds = AvalancheDataset(dataset)
                return ds

            train_datasets.append(create_ds(tx_train, ty_train, i))
            val_datasets.append(create_ds(tx_val, ty_val, i))
            test_datasets.append(create_ds(tx_test, ty_test, i))

        except Exception as e:
            print(f"   [Error] Failed to load {task_name}: {e}")
            import traceback
            traceback.print_exc()
            exit()

    return train_datasets, val_datasets, test_datasets, input_dim

# ==========================================
# PART 4: MAIN EXECUTION
# ==========================================
def main():
    print(f"Using device: {DEVICE}")

    train_ds, val_ds, test_ds, IN_DIM = load_and_scale_data()

    print("\n[Setup] Creating Benchmark...")
    benchmark = benchmark_from_datasets(train=train_ds, test=test_ds)

    print(f"\n[Setup] Initializing BigNet from model.py (Input Dim: {IN_DIM})...")
    model = BigNet(num_classes=NUM_CLASSES, input_dim=IN_DIM).to(DEVICE)

    print("\n[Setup] Configuring Loggers...")
    logger = [
        InteractiveLogger(),
        NaturalSortTextLogger(open(LOG_FILE, 'w')), 
        WandBLogger( 
            project_name="NIDS_Continual_Learning",
            run_name="NF+Malaya_Run1_4GB_2MB", # [UPDATED] Differentiates it on your WandB dashboard
            params={"config": {"strategy": "ReplayBalanced", "epochs": EPOCHS}}
        )
    ]

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True), 
        loggers=logger
    )

    print(f"\n[Training] Initializing Strategy (Freq={EVAL_FREQ})...")

    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0001) 
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    storage_policy = ReservoirSamplingBuffer(max_size=MEM_SIZE)
    smart_context_plugin = SmartContextManager(num_classes=NUM_CLASSES)

    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=TRAIN_BATCH_SIZE,
        train_epochs=EPOCHS,
        eval_mb_size=EVAL_BATCH_SIZE,
        device=DEVICE,
        evaluator=eval_plugin,
        eval_every=EVAL_FREQ,
        plugins=[
            LRSchedulerPlugin(scheduler=scheduler, metric="train_loss", step_granularity="epoch", reset_scheduler=True, reset_lr=False),
            ReplayPlugin(mem_size=MEM_SIZE, storage_policy=storage_policy)
        ]
    )

    strategy.plugins.append(smart_context_plugin)   

    print("\n[Execution] Starting Stream...")

    experiment_start_time = time.time()

    for experience in benchmark.train_stream:
        task_id = experience.current_experience
        print(f"\n>>> EXPERIENCE {task_id}: {TASK_ORDER[task_id]}")

        current_test_stream = benchmark.test_stream[:task_id+1]

        strategy.train(experience, eval_streams=[current_test_stream])
        
        print('   Finalizing Task...')
        strategy.eval(current_test_stream)

    experiment_end_time = time.time()
    total_duration_sec = experiment_end_time - experiment_start_time
    hours, rem = divmod(total_duration_sec, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    
    # ==========================================
    # HARDWARE & MEMORY REPORT
    # ==========================================
    print("\n==========================================")
    print("        HARDWARE & MEMORY REPORT")
    print("==========================================")
    
    print(f"[Hardware] Total Training & Eval Time: {time_str}")

    bytes_per_feature = 4
    bytes_per_label = 8  
    bytes_per_sample = (IN_DIM * bytes_per_feature) + bytes_per_label
    total_buffer_bytes = MEM_SIZE * bytes_per_sample
    buffer_mb = total_buffer_bytes / (1024 * 1024)

    print(f"[Storage] Replay Buffer Target: {MEM_SIZE} samples")
    print(f"[Storage] Exact Buffer Size: {total_buffer_bytes} Bytes ({buffer_mb:.4f} MB)")

    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        peak_vram_gb = peak_vram_mb / 1024
        print(f"[Compute] Batch Size Used: {TRAIN_BATCH_SIZE}")
        print(f"[Compute] Peak GPU VRAM Used: {peak_vram_mb:.2f} MB ({peak_vram_gb:.2f} GB)")
    else:
        print("[Compute] CPU Mode. GPU VRAM not applicable.")
    print("==========================================\n")

    # 3. EDGE INFERENCE LATENCY BENCHMARK
    print("\n[Latency] Running Edge Inference Benchmark with MIXED REAL DATA...")
    model.eval()
    
    real_packets_list = []
    # [UPDATED] Automatically splits the 1000 samples across the 4 datasets
    samples_per_task = 1000 // len(test_ds)
    remainder = 1000 % len(test_ds)
    
    for i, ds in enumerate(test_ds):
        batch_size = samples_per_task + (remainder if i == len(test_ds) - 1 else 0)
        
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        x_batch, _, _ = next(iter(loader))
        real_packets_list.append(x_batch)
        
    real_packets = torch.cat(real_packets_list, dim=0)
    real_packets = real_packets.to(DEVICE)
    
    with torch.no_grad():
        for i in range(100):
            _ = model(real_packets[i:i+1])
        if DEVICE == "cuda": torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        for i in range(1000):
            _ = model(real_packets[i:i+1])
        if DEVICE == "cuda": torch.cuda.synchronize()
        end_time = time.perf_counter()

    total_time_ms = (end_time - start_time) * 1000
    avg_latency_ms = total_time_ms / 1000
    
    print(f"[Latency] Processed {len(real_packets)} MIXED real network packets sequentially.")
    print(f"[Latency] Average Inference Time: {avg_latency_ms:.4f} milliseconds per packet")
    print("==========================================\n")

    print("[WandB] Syncing Hardware & Memory Report to dashboard...")
    wandb.log({
        "Hardware/Buffer_Size_MB": buffer_mb,
        "Hardware/Peak_VRAM_GB": peak_vram_mb / 1024 if torch.cuda.is_available() else 0,
        "Hardware/Inference_Latency_ms": avg_latency_ms,
        "Hardware/Total_Duration_Seconds": total_duration_sec,
        "Hardware/Total_Duration_Minutes": total_duration_sec / 60.0
    })

    print("\n[Complete] Run finished successfully.")
    wandb.finish()

if __name__ == "__main__":
    main()