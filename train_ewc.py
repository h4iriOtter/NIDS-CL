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
from model import SimpleCNN1D
import utils

# --- AVALANCHE IMPORTS ---
from avalanche.benchmarks.generators import benchmark_from_datasets
from avalanche.evaluation.metrics import (
    forgetting_metrics, accuracy_metrics, loss_metrics, 
    bwt_metrics, StreamConfusionMatrix, class_accuracy_metrics
)
from avalanche.logging import InteractiveLogger, TextLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from avalanche.training.strategies import EWC

# ==========================================
# CONFIGURATION
# ==========================================
TASK_ORDER = ['NF-BoT-IoT-v2', 'NF-ToN-IoT-v2', 'NF-CSE-CIC-IDS2018-v2', 'NF-UNSW-NB15-v2']
BATCH_SIZE = 256
EPOCHS = 1         # Set to 20 for real training (or 1 for debugging)
NUM_CLASSES = 20    # Matches your check_labels.py
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ==========================================
# STEP 1: PREPARE YOUR DATA
# ==========================================
def load_and_scale_data():
    """
    Loads data from 4 separate folders and scales them individually
    to prevent data leakage.
    """
    repo_root = os.getcwd()
    train_datasets, test_datasets, val_datasets = [], [], []
    
    print("\n[Step 1] Loading and Standardizing Data...")
    
    for task_name in TASK_ORDER:
        data_dir = os.path.join(repo_root, 'benchmark_data', task_name)
        print(f"   Processing {task_name}...")

        try:
            # Load Numpy Arrays
            train_np = np.load(os.path.join(data_dir, 'train.npy'))
            val_np   = np.load(os.path.join(data_dir, 'val.npy'))
            test_np  = np.load(os.path.join(data_dir, 'test.npy'))

            # Separate Features (X) and Labels (Y)
            train_x, train_y = train_np[:, :-1], train_np[:, -1].astype(np.int64)
            val_x, val_y     = val_np[:, :-1], val_np[:, -1].astype(np.int64)
            test_x, test_y   = test_np[:, :-1], test_np[:, -1].astype(np.int64)

            # Standardize Features (Fit on Train ONLY)
            scaler = StandardScaler().fit(train_x)
            train_x = scaler.transform(train_x)
            val_x   = scaler.transform(val_x)
            test_x  = scaler.transform(test_x)
            
            # Create TensorDatasets
            train_datasets.append(TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y)))
            val_datasets.append(TensorDataset(torch.from_numpy(val_x).float(), torch.from_numpy(val_y)))
            test_datasets.append(TensorDataset(torch.from_numpy(test_x).float(), torch.from_numpy(test_y)))

        except Exception as e:
            print(f"   [ERROR] Failed to load {task_name}: {e}")
            exit()

    return train_datasets, val_datasets, test_datasets

train_ds, val_ds, test_ds = load_and_scale_data()

# ==========================================
# STEP 2: CREATE A BENCHMARK (SCENARIO)
# ==========================================
print("\n[Step 2] Creating Avalanche Benchmark...")
# We use benchmark_from_datasets because we have distinct folders
benchmark = benchmark_from_datasets(train=train_ds, test=test_ds)
val_benchmark = benchmark_from_datasets(train=train_ds, test=val_ds)

print(f"   Number of experiences: {len(benchmark.train_stream)}")

# ==========================================
# STEP 3: DEFINE YOUR MODEL
# ==========================================
print("\n[Step 3] Defining Model...")
model = SimpleCNN1D(num_classes=NUM_CLASSES)
model = model.to(DEVICE)

# ==========================================
# STEP 4: SET UP EVALUATION METRICS & LOGGING
# ==========================================
print("\n[Step 4] Setting up Metrics & WandB...")

# Loggers
logger = [
    InteractiveLogger(), 
    TextLogger(open('training_log.txt', 'w')),
    WandBLogger(
        project_name="NIDS_Continual_Learning",
        run_name="EWC_Structured_Run",
        params={"config": {"strategy": "EWC", "lambda": 5000, "epochs": EPOCHS}}
    )
]

# Evaluation Plugin
eval_plugin = EvaluationPlugin(
    # Accuracy
    accuracy_metrics(epoch=True, experience=True, stream=True),
    # Loss
    loss_metrics(epoch=True, experience=True, stream=True),
    # Forgetting & Transfer
    forgetting_metrics(experience=True, stream=True),
    bwt_metrics(experience=True, stream=True),
    # Per-class stats
    class_accuracy_metrics(experience=True),
    StreamConfusionMatrix(num_classes=NUM_CLASSES, save_image=False),
    # Attach loggers
    loggers=logger
)

# ==========================================
# STEP 5: CHOOSE A TRAINING STRATEGY (EWC)
# ==========================================
print("\n[Step 5] Initializing EWC Strategy...")

optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()

# Scheduler
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
lr_plugin = LRSchedulerPlugin(scheduler=scheduler, metric="val_loss", step_granularity="epoch", reset_scheduler=True, reset_lr=True)

strategy = EWC(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    ewc_lambda=5000.0,       # Strength of EWC (Higher = Less Forgetting)
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
    print(f"   Classes: {experience.classes_in_this_experience}")
    
    # Train
    strategy.train(experience, eval_streams=[val_benchmark.test_stream])
    
    # Evaluate on current test stream
    print('   Evaluating on Test Stream...')
    res = strategy.eval(benchmark.test_stream)
    
    # Custom Metric: Macro F1 (Logged to WandB)
    conf_mat_tensor = utils.extract_stream_confmat(res)
    if conf_mat_tensor is not None:
        _, _, macro_f1 = utils.class_acc_and_macro_f1_from_confmat(conf_mat_tensor)
        print(f"   [METRIC] Macro F1: {macro_f1:.4f}")
        wandb.log({f"Macro_F1/Task_{task_id}": macro_f1, "Macro_F1/Stream": macro_f1})

# ==========================================
# STEP 7: EVALUATE RESULTS
# ==========================================
print("\n[Step 7] Final Evaluation...")

# Final full evaluation
final_results = strategy.eval(benchmark.test_stream)

print("\n=== FINAL RESULTS SUMMARY ===")
print(f"Final Average Accuracy: {final_results.get('Top1_Acc_Stream/eval_phase/test_stream/Task003', 'N/A')}")
print(f"Average Forgetting:     {final_results.get('StreamForgetting/eval_phase/test_stream', 'N/A')}")

wandb.finish()