import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import wandb # Import wandb library

# --- CUSTOM IMPORTS ---
from model import SimpleCNN1D
import utils  # This automatically applies your patch to fix WandB logging

# --- AVALANCHE IMPORTS ---
from avalanche.benchmarks.generators import benchmark_from_datasets
from avalanche.evaluation.metrics import (
    forgetting_metrics, accuracy_metrics, loss_metrics, 
    bwt_metrics, StreamConfusionMatrix
)
from avalanche.logging import InteractiveLogger, TextLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from avalanche.training.strategies import EWC

# ==========================================
# 1. CONFIGURATION
# ==========================================
TASK_ORDER = ['NF-BoT-IoT-v2', 'NF-ToN-IoT-v2', 'NF-CSE-CIC-IDS2018-v2', 'NF-UNSW-NB15-v2']
BATCH_SIZE = 256
EPOCHS = 1          # Set to 1 for testing, change to 20 later
NUM_CLASSES = 20    # Ensure this matches your check_labels.py
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# ==========================================
# 2. DATA LOADING & SCALING
# ==========================================
def load_and_scale_data():
    repo_root = os.getcwd()
    train_datasets, test_datasets, val_datasets = [], [], []
    
    print("\n--- LOADING AND SCALING DATA ---")
    
    for task_name in TASK_ORDER:
        data_dir = os.path.join(repo_root, 'benchmark_data', task_name)
        print(f"Processing {task_name}...")

        try:
            # 1. Load Numpy
            train_np = np.load(os.path.join(data_dir, 'train.npy'))
            val_np   = np.load(os.path.join(data_dir, 'val.npy'))
            test_np  = np.load(os.path.join(data_dir, 'test.npy'))

            # 2. Split X and Y
            train_x_raw = train_np[:, :-1]
            train_y = train_np[:, -1].astype(np.int64)
            
            val_x_raw = val_np[:, :-1]
            val_y = val_np[:, -1].astype(np.int64)
            
            test_x_raw = test_np[:, :-1]
            test_y = test_np[:, -1].astype(np.int64)

            # 3. Fit Scaler on TRAIN only
            scaler = StandardScaler().fit(train_x_raw)
            
            # 4. Transform all sets
            train_x = scaler.transform(train_x_raw)
            val_x   = scaler.transform(val_x_raw)
            test_x  = scaler.transform(test_x_raw)
            
            # 5. Wrap in TensorDatasets
            train_ds = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y))
            val_ds   = TensorDataset(torch.from_numpy(val_x).float(), torch.from_numpy(val_y))
            test_ds  = TensorDataset(torch.from_numpy(test_x).float(), torch.from_numpy(test_y))
            
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
            test_datasets.append(test_ds)
            print(f"   -> Done. Train size: {len(train_ds)}")

        except Exception as e:
            print(f"   [ERROR] Failed to load {task_name}: {e}")
            exit()

    return train_datasets, val_datasets, test_datasets

train_ds, val_ds, test_ds = load_and_scale_data()

benchmark = benchmark_from_datasets(train=train_ds, test=test_ds)
val_benchmark = benchmark_from_datasets(train=train_ds, test=val_ds)

# ==========================================
# 3. MODEL & LOGGING (WANDB ENABLED)
# ==========================================
model = SimpleCNN1D(num_classes=NUM_CLASSES)
model = model.to(DEVICE)

# --- LOGGERS ---
# 1. Interactive: Prints to console
# 2. TextLogger: Saves to .txt file
# 3. WandBLogger: Sends to your dashboard
logger = [InteractiveLogger(), TextLogger(open('EWC_log.txt', 'w'))]

# Initialize WandB
# project_name: The name of the project on your dashboard
# run_name: A specific name for this experiment
wandb_logger = WandBLogger(
    project_name="NIDS_Continual_Learning", 
    run_name="SimpleCNN_EWC_Experiment",
    params={"config": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "model": "SimpleCNN1D"}}
)
logger.append(wandb_logger)

# Metrics Plugin
eval_plugin = EvaluationPlugin(
    accuracy_metrics(experience=True, stream=True),
    loss_metrics(epoch=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    bwt_metrics(experience=True, stream=True),
    StreamConfusionMatrix(num_classes=NUM_CLASSES, save_image=False),
    loggers=logger
)

# Optimizer
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
lr_plugin = LRSchedulerPlugin(scheduler=scheduler, metric="val_loss", step_granularity="epoch", reset_scheduler=True, reset_lr=True)

# EWC Strategy
print("\n--- INITIALIZING EWC STRATEGY ---")
cl_strategy = EWC(
    model=model,
    optimizer=optimizer,
    criterion=nn.CrossEntropyLoss(),
    ewc_lambda=5000.0,
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
# 4. TRAINING LOOP
# ==========================================
print('\nStarting experiment...')

if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

for experience in benchmark.train_stream:
    task_id = experience.current_experience
    print(f"\n>>> PROCESSING TASK {task_id}: {TASK_ORDER[task_id]}")
    
    # Train
    cl_strategy.train(experience, eval_streams=[val_benchmark.test_stream])
    
    # Save Model
    torch.save(model.state_dict(), f"saved_models/ewc_task_{task_id}.pth")

    # Evaluate
    print('   Evaluating...')
    res = cl_strategy.eval(benchmark.test_stream)
    
    # Extract F1 and Log to WandB manually if needed
    conf_mat_tensor = utils.extract_stream_confmat(res)
    if conf_mat_tensor is not None:
        _, _, macro_f1 = utils.class_acc_and_macro_f1_from_confmat(conf_mat_tensor)
        print(f"\n   [METRIC] Macro F1: {macro_f1:.4f}")
        
        # Log F1 Score specifically to WandB
        wandb.log({f"Macro_F1/Task_{task_id}": macro_f1})
        wandb.log({"Macro_F1/Stream": macro_f1})
    else:
        print("   [WARNING] No Confusion Matrix found.")

print('\n--- EXPERIMENT FINISHED ---')
wandb.finish()