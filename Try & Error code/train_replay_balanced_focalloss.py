import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import wandb

# --- CUSTOM IMPORTS ---
from model import LiteNet
import utils 

# --- AVALANCHE IMPORTS ---
from avalanche.benchmarks import benchmark_from_datasets
# We use this helper to satisfy the type check
from avalanche.benchmarks.utils import as_classification_dataset
from avalanche.training.supervised import Naive
from avalanche.training.plugins import ReplayPlugin, EvaluationPlugin, LRSchedulerPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer 
from avalanche.evaluation.metrics import (
    forgetting_metrics, accuracy_metrics, loss_metrics, 
    StreamConfusionMatrix
)
from avalanche.logging import InteractiveLogger, TextLogger, WandBLogger

# ==========================================
# 0. CONFIGURATION
# ==========================================
TASK_ORDER = ['NF-BoT-IoT-v2', 'NF-ToN-IoT-v2']
BATCH_SIZE = 256
EPOCHS = 20        
NUM_CLASSES = 20   
MEM_SIZE = 5000    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ==========================================
# 1. HELPER CLASSES
# ==========================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        return focal_loss.sum()

# --- CUSTOM DATASET CLASS ---
class NIDSDataset(Dataset):
    """
    A robust custom dataset that:
    1. Returns (x, y, task_id) in __getitem__
    2. Exposes .targets for ClassBalancedBuffer
    """
    def __init__(self, x, y, task_id):
        self.x = x
        self.y = y
        self.t = task_id
        
        # CRITICAL: Expose targets explicitly for buffer
        self.targets = y 
        
    def __getitem__(self, index):
        # Return the triplet required by Avalanche
        return self.x[index], self.y[index], self.t
        
    def __len__(self):
        return len(self.x)

# ==========================================
# 2. DATA LOADING (THE FIX)
# ==========================================
def load_and_scale_data():
    repo_root = os.getcwd()
    train_datasets, test_datasets, val_datasets = [], [], []
    
    print("\n[Step 1] Loading and Standardizing Data...")
    
    for i, task_name in enumerate(TASK_ORDER):
        data_dir = os.path.join(repo_root, 'benchmark_data', task_name)
        print(f"   Processing Task {i}: {task_name}...")

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

            # Split X and Y
            train_x, train_y = train_np[:, :-1], train_np[:, -1]
            val_x, val_y     = val_np[:, :-1], val_np[:, -1]
            test_x, test_y   = test_np[:, :-1], test_np[:, -1]

            # Scale
            scaler = StandardScaler().fit(train_x)
            train_x = scaler.transform(train_x)
            val_x   = scaler.transform(val_x)
            test_x  = scaler.transform(test_x)
            
            # Convert to Tensors
            tx_train, ty_train = torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long()
            tx_val, ty_val     = torch.from_numpy(val_x).float(), torch.from_numpy(val_y).long()
            tx_test, ty_test   = torch.from_numpy(test_x).float(), torch.from_numpy(test_y).long()

            # --- THE LOGIC FIX ---
            # 1. Instantiate our Custom Dataset (Handles .targets and Task ID)
            raw_train = NIDSDataset(tx_train, ty_train, task_id=i)
            raw_val   = NIDSDataset(tx_val,   ty_val,   task_id=i)
            raw_test  = NIDSDataset(tx_test,  ty_test,  task_id=i)

            # 2. Wrap it to satisfy "datasets must be AvalancheDatasets"
            # We pass NO arguments because the custom dataset already handles logic.
            # This converts it to the type expected by the benchmark generator.
            train_ds = as_classification_dataset(raw_train)
            val_ds   = as_classification_dataset(raw_val)
            test_ds  = as_classification_dataset(raw_test)

            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
            test_datasets.append(test_ds)

        except Exception as e:
            print(f"   [ERROR] Failed to load {task_name}: {e}")
            import traceback
            traceback.print_exc()
            exit()

    return train_datasets, val_datasets, test_datasets

# Load Data
train_ds, val_ds, test_ds = load_and_scale_data()

# Create Benchmark
print("\n[Step 2] Creating Benchmark...")
# Now these lists contain proper AvalancheDatasets
benchmark = benchmark_from_datasets(train=train_ds, test=test_ds)
val_benchmark = benchmark_from_datasets(train=train_ds, test=val_ds)

# ==========================================
# 3. STRATEGY SETUP
# ==========================================
print("\n[Step 3] Initializing Model & Strategy...")

model = LiteNet(num_classes=NUM_CLASSES).to(DEVICE)

# LOGGING
logger = [
    InteractiveLogger(), 
    TextLogger(open('training_log_final.txt', 'w')),
    WandBLogger(
        project_name="NIDS_Continual_Learning",
        run_name="Replay_Final_Wrapper_Fix",
        params={"config": {"strategy": "Replay", "mem_size": MEM_SIZE, "loss": "Focal"}}
    )
]

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    StreamConfusionMatrix(num_classes=NUM_CLASSES, save_image=False),
    loggers=logger
)

# OPTIMIZER (High LR for speed)
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

# LOSS (Focal for Imbalance)
criterion = FocalLoss(gamma=2.0)

# SCHEDULER
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
lr_plugin = LRSchedulerPlugin(scheduler=scheduler, metric="val_loss", step_granularity="epoch", reset_scheduler=True, reset_lr=True)

# REPLAY PLUGIN (Balanced)
# This will work because the underlying NIDSDataset has .targets
storage_policy = ClassBalancedBuffer(max_size=MEM_SIZE, adaptive_size=True)
replay_plugin = ReplayPlugin(mem_size=MEM_SIZE, storage_policy=storage_policy)

# STRATEGY
strategy = Naive(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_mb_size=BATCH_SIZE,
    train_epochs=EPOCHS,
    eval_mb_size=BATCH_SIZE,
    device=DEVICE,
    evaluator=eval_plugin,
    plugins=[lr_plugin, replay_plugin]
)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
print("\n[Step 4] Starting Training...")

for experience in benchmark.train_stream:
    task_id = experience.current_experience
    print(f"\n>>> EXPERIENCE {task_id}: {TASK_ORDER[task_id]}")
    
    # Train
    strategy.train(experience, eval_streams=[val_benchmark.test_stream])
    
    # Evaluate
    print('   Evaluating on Test Stream...')
    res = strategy.eval(benchmark.test_stream)
    
    # Log Custom Macro F1
    if hasattr(utils, 'extract_stream_confmat'):
        conf_mat_tensor = utils.extract_stream_confmat(res)
        if conf_mat_tensor is not None:
            _, _, macro_f1 = utils.class_acc_and_macro_f1_from_confmat(conf_mat_tensor)
            print(f"   [METRIC] Macro F1: {macro_f1:.4f}")
            try:
                wandb.log({f"Macro_F1/Task_{task_id}": macro_f1}, commit=True)
            except:
                pass

print("\n=== Experiment Complete ===")
wandb.finish()