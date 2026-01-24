import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset
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
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.training.supervised import EWC
from avalanche.evaluation.metrics import (
    forgetting_metrics, accuracy_metrics, loss_metrics, 
    bwt_metrics, StreamConfusionMatrix, class_accuracy_metrics
)
from avalanche.logging import InteractiveLogger, TextLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin

# ==========================================
# CONFIGURATION
# ==========================================
TASK_ORDER = ['NF-BoT-IoT-v2', 'NF-ToN-IoT-v2']
# other two data 'NF-CSE-CIC-IDS2018-v2', 'NF-UNSW-NB15-v2'
BATCH_SIZE = 256
EPOCHS = 20         # Set to 20 for real training
NUM_CLASSES = 20    # Matches your check_labels.py
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ==========================================
# STEP 1: PREPARE YOUR DATA
# ==========================================
class TaskTensorDataset(TensorDataset):
    def __init__(self, x, y, task_id):
        super().__init__(x, y)
        self.task_id = task_id

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        # Force return of triplet: (Input, Label, Task_ID)
        return x, y, self.task_id

def load_and_scale_data():
    """
    Loads data and forces strict (x, y, t) format using a custom wrapper.
    """
    repo_root = os.getcwd()
    train_datasets, test_datasets, val_datasets = [], [], []
    
    print("\n[Step 1] Loading and Standardizing Data...")
    
    for i, task_name in enumerate(TASK_ORDER):
        data_dir = os.path.join(repo_root, 'benchmark_data', task_name)
        print(f"   Processing Task {i}: {task_name}...")

        try:
            # 1. Load Numpy Arrays
            train_np = np.load(os.path.join(data_dir, 'train.npy'))
            val_np   = np.load(os.path.join(data_dir, 'val.npy'))
            test_np  = np.load(os.path.join(data_dir, 'test.npy'))

            # 2. CLEANING FUNCTION
            def clean_numpy(arr, name):
                if np.isnan(arr).any() or np.isinf(arr).any():
                    arr = arr[~np.isnan(arr).any(axis=1)] 
                    arr = arr[~np.isinf(arr).any(axis=1)]
                return arr

            train_np = clean_numpy(train_np, "train")
            val_np   = clean_numpy(val_np, "val")
            test_np  = clean_numpy(test_np, "test")

            # 3. Separate Features and Labels
            train_x, train_y = train_np[:, :-1], train_np[:, -1]
            val_x, val_y     = val_np[:, :-1], val_np[:, -1]
            test_x, test_y   = test_np[:, :-1], test_np[:, -1]

            # 4. Standardize Features
            scaler = StandardScaler().fit(train_x)
            train_x = scaler.transform(train_x)
            val_x   = scaler.transform(val_x)
            test_x  = scaler.transform(test_x)
            
            # 5. CONVERT TO TENSORS
            tx_train, ty_train = torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long()
            tx_val, ty_val     = torch.from_numpy(val_x).float(), torch.from_numpy(val_y).long()
            tx_test, ty_test   = torch.from_numpy(test_x).float(), torch.from_numpy(test_y).long()

            # 6. WRAP WITH CUSTOM CLASS (The "Bulletproof" Fix)
            # This bypasses Avalanche's init arguments and manually injects the task ID.
            # We wrap it in AvalancheDataset at the end just to satisfy type checks.
            
            train_ds = AvalancheDataset(TaskTensorDataset(tx_train, ty_train, task_id=i))
            val_ds   = AvalancheDataset(TaskTensorDataset(tx_val, ty_val, task_id=i))
            test_ds  = AvalancheDataset(TaskTensorDataset(tx_test, ty_test, task_id=i))

            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
            test_datasets.append(test_ds)

        except Exception as e:
            print(f"   [ERROR] Failed to load {task_name}: {e}")
            exit()

    return train_datasets, val_datasets, test_datasets

train_ds, val_ds, test_ds = load_and_scale_data()

# ==========================================
# STEP 2: CREATE A BENCHMARK (SCENARIO)
# ==========================================
print("\n[Step 2] Creating Avalanche Benchmark...")
benchmark = benchmark_from_datasets(train=train_ds, test=test_ds)
val_benchmark = benchmark_from_datasets(train=train_ds, test=val_ds)

print(f"   Number of experiences: {len(benchmark.train_stream)}")

# ==========================================
# STEP 3: DEFINE YOUR MODEL
# ==========================================
print("\n[Step 3] Defining Model...")
model = LiteNet(num_classes=NUM_CLASSES)
model = model.to(DEVICE)

# ==========================================
# STEP 4: SET UP EVALUATION METRICS & LOGGING
# ==========================================
print("\n[Step 4] Setting up Metrics & WandB...")

logger = [
    InteractiveLogger(), 
    TextLogger(open('training_log.txt', 'w')),
    WandBLogger(
        project_name="NIDS_Continual_Learning",
        run_name="EWC_Test_Run_20 epoch",
        params={"config": {"strategy": "EWC", "lambda": 400, "epochs": EPOCHS}}
    )
]

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    bwt_metrics(experience=True, stream=True),
    class_accuracy_metrics(experience=True),
    StreamConfusionMatrix(num_classes=NUM_CLASSES, save_image=False),
    loggers=logger
)

# ==========================================
# STEP 5: CHOOSE A TRAINING STRATEGY (EWC)
# ==========================================
print("\n[Step 5] Initializing EWC Strategy...")

optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()

scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
lr_plugin = LRSchedulerPlugin(scheduler=scheduler, metric="val_loss", step_granularity="epoch", reset_scheduler=True, reset_lr=True)

strategy = EWC(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    ewc_lambda=400.0,
    mode="online",
    decay_factor=0.9,
    train_mb_size=BATCH_SIZE,
    train_epochs=EPOCHS,
    eval_mb_size=BATCH_SIZE,
    device=DEVICE,
    evaluator=eval_plugin,
    plugins=[lr_plugin]
)

# ==========================================
# STEP 6: TRAIN THE MODEL
# ==========================================
print("\n[Step 6] Starting Training Loop...")

for experience in benchmark.train_stream:
    task_id = experience.current_experience
    print(f"\n>>> EXPERIENCE {task_id}: {TASK_ORDER[task_id]}")
    
    try:
        # We manually collect the Y (label) from every sample in the batch
        # Dataset returns (x, y, t), so we take [1] for y.
        unique_classes = set()
        for _, y, _ in experience.dataset:
            # y might be a Tensor or an Int, handle both
            val = y.item() if hasattr(y, 'item') else y
            unique_classes.add(val)
            
        print(f"   Classes found: {sorted(list(unique_classes))}")
    except Exception as e:
        print(f"   (Could not print classes: {e})")

    # Train
    strategy.train(experience, eval_streams=[val_benchmark.test_stream])
    
    # Evaluate
    print('   Evaluating on Test Stream...')
    res = strategy.eval(benchmark.test_stream)
    
    # Custom Metric: Macro F1
    conf_mat_tensor = utils.extract_stream_confmat(res)
    if conf_mat_tensor is not None:
        _, _, macro_f1 = utils.class_acc_and_macro_f1_from_confmat(conf_mat_tensor)
        print(f"   [METRIC] Macro F1: {macro_f1:.4f}")
        wandb.log(
            {f"Macro_F1/Task_{task_id}": macro_f1, "Macro_F1/Stream": macro_f1},
            commit=False
        )

# ==========================================
# STEP 7: EVALUATE RESULTS
# ==========================================
print("\n[Step 7] Final Evaluation...")

final_results = strategy.eval(benchmark.test_stream)
print("\n=== FINAL RESULTS SUMMARY ===")
acc_key = 'Top1_Acc_Stream/eval_phase/test_stream'
print(f"Final Average Accuracy: {final_results.get(acc_key, 'Key not found')}")
wandb.finish()