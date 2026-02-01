import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import re
from sklearn.preprocessing import StandardScaler
import wandb

# --- CUSTOM IMPORTS ---
from model import BigNet
import utils
from utils import NaturalSortTextLogger, calculate_smart_f1, extract_stream_confmat

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
    forgetting_metrics, accuracy_metrics, loss_metrics,
    StreamConfusionMatrix, #class_accuracy_metrics,
)
from avalanche.logging import  InteractiveLogger, WandBLogger
from avalanche.training.storage_policy import ReservoirSamplingBuffer

# ==========================================
# CONFIGURATION
# ==========================================
TASK_ORDER = ['NF-UNSW-NB15-v2', 'NF-CSE-CIC-IDS2018-v2', 'NF-ToN-IoT-v2']
TRAIN_BATCH_SIZE = 4096  # Speed up training (GPU friendly)
EVAL_BATCH_SIZE = 4096  # 1 for Simulated Edge
MEM_SIZE = 10000    # The size of the Replay Buffer (Memory of past tasks)
EPOCHS = 20         # 20 Epochs is sufficient for testing convergence
EVAL_FREQ = -1       # Evaluate every 1 epoch to generate high-resolution learning curves
NUM_CLASSES = 20    # Fixed it to 20 eventhough class 13 is not used in this since we using dynamic seen
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_FILE = 'training_log_replay_balanced_optimized.txt'

# ==========================================
# PART 1: OPTIMIZED PLUGIN 
# (Combines Metrics + Weights using Utils)
# ==========================================
class SmartContextManager(SupervisedPlugin):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.seen_classes_so_far = set() 

    def before_training_exp(self, strategy, **kwargs):
        print("\n[SmartContext] Analyzing dataset statistics...")
        
        # 1. Fast Scan (CPU safe)
        try:
            y_data = torch.as_tensor(strategy.experience.dataset.targets).cpu()
        except:
            loader = DataLoader(
                strategy.experience.dataset, batch_size=20000, 
                num_workers=4, pin_memory=False
            )
            y_data = torch.cat([batch[1].cpu() for batch in loader])

        y_np = y_data.numpy()

        # 2. Update Seen Classes
        self.seen_classes_so_far.update(np.unique(y_np).tolist())
        print(f"   [Metrics] Seen Classes: {sorted(list(self.seen_classes_so_far))}")

        # 3. Dynamic Weighting
        classes, counts = np.unique(y_np, return_counts=True)
        if len(classes) == 0: return

        raw_weights = sum(counts) / (len(classes) * counts)
        full_weights = torch.ones(self.num_classes).to(strategy.device)

        for cls, w in zip(classes, raw_weights):
            if cls < self.num_classes:
                full_weights[int(cls)] = min(float(w), 1000.0)
        
        strategy._criterion = nn.CrossEntropyLoss(weight=full_weights)

        print(f"   [Weights] Updated Loss Weights for {len(classes)} classes.")

    def after_training_epoch(self, strategy, **kwargs):
        # Get metrics
        metrics = strategy.evaluator.get_last_metrics()
        # Note: Ensure you updated extract_stream_confmat in utils.py to handle mode='train'
        conf_mat = extract_stream_confmat(metrics, mode='train')
        
        if conf_mat is not None:
            # --- CALCULATE BOTH AT THE SAME TIME ---
            f1_local, _, _ = calculate_smart_f1(conf_mat, strict=True)
            f1_global, _, _ = calculate_smart_f1(conf_mat, strict=False)
            
            # Log to WandB with distinct names
            wandb.log({
                "F1_Score/Train_Local": f1_local,    # High score (Current Task)
                "F1_Score/Train_Global": f1_global,  # Low score (System Health)
                "Epoch": strategy.clock.train_exp_epochs
            })
            
            print(f"   [Epoch {strategy.clock.train_exp_epochs}] Train F1 -> Local: {f1_local:.4f} | Global: {f1_global:.4f}")

    def after_eval_exp(self, strategy, **kwargs):
        metrics = strategy.evaluator.get_last_metrics()
        conf_mat = extract_stream_confmat(metrics, mode='eval')
        
        if conf_mat is not None:
            # --- CALCULATE BOTH AT THE SAME TIME ---
            f1_local, prec_local, rec_local = calculate_smart_f1(conf_mat, strict=True)
            f1_global, prec_global, rec_global = calculate_smart_f1(conf_mat, strict=False)
            
            # Log to WandB
            wandb.log({
                "F1_Score/Test_Local": f1_local,
                "F1_Score/Test_Global": f1_global,
                "Precision/Test_Local": prec_local,
                "Recall/Test_Local": rec_local,
                "Precision/Test_Global": prec_global,
                "Recall/Test_Global": rec_global,
            })
            
            msg = f"\n\t[Smart Metrics] Local F1: {f1_local:.4f} (Task Performance) | Global F1: {f1_global:.4f} (System Stability)"
            print(msg)
            
            # Save to text log
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
    
    # Define Scaler OUTSIDE the loop
    global_scaler = StandardScaler()

    print("\n[Data] Loading and Standardizing...")
    
    for i, task_name in enumerate(TASK_ORDER):
        data_dir = os.path.join(repo_root, 'benchmark_data', task_name)
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

            # Fit ONLY on Task 1 (i==0), Transform others
            if i == 0:
                print("      [Scaler] Fitting Global Scaler on Task 1...")
                global_scaler.fit(train_x)
            
            # Apply the scaler
            train_x_scaled = global_scaler.transform(train_x)
            val_x_scaled   = global_scaler.transform(val_x)
            test_x_scaled  = global_scaler.transform(test_x)

            # Convert to tensors
            tx_train = torch.from_numpy(train_x_scaled).float()
            ty_train = torch.from_numpy(train_y).long()
            
            tx_val   = torch.from_numpy(val_x_scaled).float()
            ty_val   = torch.from_numpy(val_y).long()
            
            tx_test  = torch.from_numpy(test_x_scaled).float()
            ty_test  = torch.from_numpy(test_y).long()

            def create_ds(x, y, tid):
                # 1. Use your custom TaskTensorDataset (returns x, y, tid)
                dataset = TaskTensorDataset(x, y, task_id=tid)
                
                # 2. Wrap in AvalancheDataset without extra arguments
                # It automatically detects the 3rd return value as the task label
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
# PART 4: MAIN EXECUTION (WRAPPED)
# ==========================================
def main():
    print(f"Using device: {DEVICE}")

    # 1. Load Data
    train_ds, val_ds, test_ds, IN_DIM = load_and_scale_data()

    # 2. Setup Benchmark
    print("\n[Setup] Creating Benchmark...")
    benchmark = benchmark_from_datasets(train=train_ds, test=test_ds)

    # 3. Setup Model
    print(f"\n[Setup] Initializing BigNet from model.py (Input Dim: {IN_DIM})...")
    model = BigNet(num_classes=NUM_CLASSES, input_dim=IN_DIM).to(DEVICE)

    # 4. Setup Loggers (WandB init happens here, safe inside main)
    print("\n[Setup] Configuring Loggers...")
    logger = [
        InteractiveLogger(),
        NaturalSortTextLogger(open(LOG_FILE, 'w')), 
        WandBLogger( 
            project_name="NIDS_Continual_Learning",
            run_name="Balanced_Replay_20epochs", 
            params={"config": {"strategy": "ReplayBalanced", "epochs": EPOCHS}}
        )
    ]

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True), 
        # class_accuracy_metrics(experience=True), commented out because generate too many logs
        StreamConfusionMatrix(num_classes=NUM_CLASSES, save_image=False), 
        loggers=logger
    )

    # 5. Setup Strategy
    print(f"\n[Training] Initializing Strategy (Freq={EVAL_FREQ})...")

    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0001) 
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    storage_policy = ReservoirSamplingBuffer(max_size=MEM_SIZE)

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
            LRSchedulerPlugin(scheduler=scheduler, metric="train_loss", step_granularity="epoch", reset_scheduler=True, reset_lr=True),
            ReplayPlugin(mem_size=MEM_SIZE, storage_policy=storage_policy), 
            SmartContextManager(num_classes=NUM_CLASSES)
        ]
    )

    # 6. Execution Loop
    print("\n[Execution] Starting Stream...")
    for experience in benchmark.train_stream:
        task_id = experience.current_experience
        print(f"\n>>> EXPERIENCE {task_id}: {TASK_ORDER[task_id]}")

        strategy.train(experience)
        
        print('   Finalizing Task...')
        current_test_stream = benchmark.test_stream[:task_id+1]
        strategy.eval(current_test_stream)

    print("\n[Complete] Run finished successfully.")
    wandb.finish()

# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    main()