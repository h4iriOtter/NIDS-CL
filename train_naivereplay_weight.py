import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import wandb

# --- CUSTOM IMPORTS ---
from model import LiteNet
import utils

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
    bwt_metrics, StreamConfusionMatrix, class_accuracy_metrics
)
from avalanche.logging import InteractiveLogger, TextLogger, WandBLogger


# ==========================================
# CONFIGURATION
# ==========================================
TASK_ORDER = ['NF-BoT-IoT-v2', 'NF-ToN-IoT-v2']
BATCH_SIZE = 4096   # Large batch size speeds up training on GPUs
MEM_SIZE = 10000    # The size of the Replay Buffer (Memory of past tasks)
EPOCHS = 20         # 20 Epochs is sufficient for testing convergence
EVAL_FREQ = 1       # Evaluate every 1 epoch to generate high-resolution learning curves
NUM_CLASSES = 20   
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_FILE = 'training_log_replay_weighted.txt'

print(f"Using device: {DEVICE}")

# ==========================================
# PLUGIN 1: DYNAMIC CLASS WEIGHTING
# ==========================================
# KNOWLEDGE: NIDS datasets are heavily imbalanced. "Normal" traffic dominates 
# while "Attacks" are rare. Without this, the model will just guess "Normal" 
# and get 99% accuracy but 0% F1 score.
class DynamicClassWeighting(SupervisedPlugin):
    """
    Computes and updates class weights before each experience to handle 
    data imbalance effectively.
    """
    # This hook runs automatically BEFORE the training loop starts for a new task
    def before_training_exp(self, strategy, **kwargs):
        print("\n[Weights] Computing class distribution...")
        
        # 1. Extract all target labels from the current dataset
        try: 
            targets = np.array(strategy.experience.dataset.targets)
        except:
            # Fallback: If direct access fails, iterate through the loader
            loader = DataLoader(strategy.experience.dataset, batch_size=5000)
            targets = []
            for batch in loader: 
                targets.extend(batch[1].numpy())
            targets = np.array(targets)
        
        # 2. Identify which classes exist in this specific task
        classes_present = np.unique(targets)
        
        # 3. Calculate weights: Rare classes get HIGH weight, Common classes get LOW weight
        cw = compute_class_weight('balanced', classes=classes_present, y=targets)
        
        # 4. Create a weight vector for the Loss Function
        full_weights = torch.ones(NUM_CLASSES).to(strategy.device)
        
        for cls, w in zip(classes_present, cw):
            if cls < NUM_CLASSES: full_weights[int(cls)] = float(w)
            
        # 5. Inject these weights into the Loss Function
        strategy._criterion = nn.CrossEntropyLoss(weight=full_weights)
        print(f"   Updated criterion with weights for {len(classes_present)} classes.")

# ==========================================
# PLUGIN 2: CUSTOM METRICS LOGGING
# ==========================================
# KNOWLEDGE: Avalanche provides basic metrics, but for a analysis, we need 
# explicit Macro-F1, Precision, and Recall aligned perfectly with our Steps.
class CustomMetricsLogger(SupervisedPlugin):
    """
    Extracts confusion matrix data to calculate and log Precision, Recall, 
    and F1 scores to WandB and local logs.
    """
    # This hook runs automatically AFTER the evaluation phase ends
    def after_eval_exp(self, strategy, **kwargs):
        # 1. Access the raw metrics stored by Avalanche
        metrics = strategy.evaluator.get_last_metrics()
        
        # 2. Extract the Confusion Matrix (The raw count of TP, FP, FN, TN)
        conf_mat_tensor = utils.extract_stream_confmat(metrics)
        
        if conf_mat_tensor is not None:
            # 3. Calculate modl metrics using our custom util
            _, _, macro_f1, macro_prec, macro_rec = utils.class_acc_and_macro_f1_from_confmat(conf_mat_tensor)
            
            # 4. Get the current training step (to align X-axis on graphs)
            current_step = strategy.clock.train_iterations
            
            # 5. Log to WandB
            # 'commit=True' forces the data to upload NOW. This ensures
            # the F1 score aligns perfectly with the Accuracy score on the graph.
            wandb.log(
                {
                    f"Macro_F1/Task_{strategy.experience.current_experience}": macro_f1, 
                    "Macro_F1/Stream": macro_f1,
                    "Macro_Precision/Stream": macro_prec,
                    "Macro_Recall/Stream": macro_rec
                }, 
                step=current_step, 
                commit=True
            )

            # 6. Log to Text File (Safety backup)
            curr_epoch = strategy.clock.train_exp_epochs
            header = f"\n\t[Metrics] Epoch {curr_epoch} Summary:"
            msg_f1 = f"\tMacro_F1        = {macro_f1:.4f}"
            msg_pr = f"\tMacro_Precision = {macro_prec:.4f}"
            msg_rc = f"\tMacro_Recall    = {macro_rec:.4f}\n"
            
            log_payload = f"{header}\n{msg_f1}\n{msg_pr}\n{msg_rc}"
            
            print(log_payload)
            with open(LOG_FILE, 'a') as f:
                f.write(log_payload)

# ==========================================
# STEP 1: DATA PIPELINE
# ==========================================
# KNOWLEDGE: Avalanche expects data in triplets (x, y, task_id). 
# Standard PyTorch only gives (x, y). This wrapper adds the missing 'task_id'.
class TaskTensorDataset(TensorDataset):
    def __init__(self, x, y, task_id):
        super().__init__(x, y)
        self.task_id = task_id
        self.targets = y  # Expose targets for the ReplayPlugin to use

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, y, self.task_id

def load_and_scale_data():
    repo_root = os.getcwd()
    train_datasets, test_datasets, val_datasets = [], [], []
    
    print("\n[Data] Loading and Standardizing...")
    
    for i, task_name in enumerate(TASK_ORDER):
        data_dir = os.path.join(repo_root, 'benchmark_data', task_name)
        print(f"   Processing {task_name}...")

        try:
            # Load raw Numpy arrays
            train_np = np.load(os.path.join(data_dir, 'train.npy'))
            val_np   = np.load(os.path.join(data_dir, 'val.npy'))
            test_np  = np.load(os.path.join(data_dir, 'test.npy'))

            # Helper to remove Infinity/NaN values which crash training
            def clean_numpy(arr):
                if np.isnan(arr).any() or np.isinf(arr).any():
                    arr = arr[~np.isnan(arr).any(axis=1)] 
                    arr = arr[~np.isinf(arr).any(axis=1)]
                return arr

            train_np = clean_numpy(train_np)
            val_np   = clean_numpy(val_np)
            test_np  = clean_numpy(test_np)

            # Split Features (X) and Labels (Y)
            train_x, train_y = train_np[:, :-1], train_np[:, -1]
            val_x, val_y     = val_np[:, :-1], val_np[:, -1]
            test_x, test_y   = test_np[:, :-1], test_np[:, -1]

            # Scaling: Normalizes data to Mean=0, Std=1. 
            # Crucial for Neural Networks to converge.
            scaler = StandardScaler().fit(train_x)
            tx_train = torch.from_numpy(scaler.transform(train_x)).float()
            ty_train = torch.from_numpy(train_y).long()
            tx_val   = torch.from_numpy(scaler.transform(val_x)).float()
            ty_val   = torch.from_numpy(val_y).long()
            tx_test  = torch.from_numpy(scaler.transform(test_x)).float()
            ty_test  = torch.from_numpy(test_y).long()

            # Create Avalanche-compatible datasets
            train_ds = AvalancheDataset(TaskTensorDataset(tx_train, ty_train, task_id=i))
            train_ds.targets = ty_train # Explicitly set targets for efficient access
            # Task labels help Avalanche know "This data belongs to Task 0"
            train_ds.targets_task_labels = torch.full((len(ty_train),), i, dtype=torch.long)
            
            val_ds = AvalancheDataset(TaskTensorDataset(tx_val, ty_val, task_id=i))
            val_ds.targets = ty_val
            val_ds.targets_task_labels = torch.full((len(ty_val),), i, dtype=torch.long)

            test_ds = AvalancheDataset(TaskTensorDataset(tx_test, ty_test, task_id=i))
            test_ds.targets = ty_test
            test_ds.targets_task_labels = torch.full((len(ty_test),), i, dtype=torch.long)

            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
            test_datasets.append(test_ds)

        except Exception as e:
            print(f"   [Error] Failed to load {task_name}: {e}")
            exit()

    return train_datasets, val_datasets, test_datasets

train_ds, val_ds, test_ds = load_and_scale_data()

# ==========================================
# STEP 2 & 3: SETUP
# ==========================================
print("\n[Setup] Creating Benchmark...")
# Converts our list of datasets into a Continuous Stream
benchmark = benchmark_from_datasets(train=train_ds, test=test_ds)
val_benchmark = benchmark_from_datasets(train=train_ds, test=val_ds)

print("\n[Setup] Initializing LiteNet...")
model = LiteNet(num_classes=NUM_CLASSES).to(DEVICE)

# ==========================================
# STEP 4: LOGGERS
# ==========================================
print("\n[Setup] Configuring Loggers...")
logger = [
    InteractiveLogger(),           # Prints progress bars to terminal
    TextLogger(open(LOG_FILE, 'w')), # Saves logs to .txt file
    WandBLogger(                   # Uploads logs to Weights & Biases cloud
        project_name="NIDS_Continual_Learning",
        run_name="Replay_Weighted_2Tasks",
        params={"config": {"strategy": "Replay", "epochs": EPOCHS, "batch_size": BATCH_SIZE}}
    )
]

# EvaluationPlugin: Defines WHAT we want to measure
eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True), # How much did we forget old tasks?
    class_accuracy_metrics(experience=True),
    StreamConfusionMatrix(num_classes=NUM_CLASSES, save_image=False), # Needed for F1 calc
    loggers=logger
)

# ==========================================
# STEP 5: STRATEGY
# ==========================================
print(f"\n[Training] Initializing Strategy (Freq={EVAL_FREQ})...")

optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss() # Placeholder (Overwritten by Plugin 1)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

# Initialize Plugins
lr_plugin = LRSchedulerPlugin(scheduler=scheduler, metric="val_loss", step_granularity="epoch", reset_scheduler=True, reset_lr=True)
replay_plugin = ReplayPlugin(mem_size=MEM_SIZE) 
weight_plugin = DynamicClassWeighting()         # Our imbalance handler
metrics_logger = CustomMetricsLogger()          # Our logging handler

# NAIVE Strategy: The base class for supervised continual learning
strategy = Naive(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_mb_size=BATCH_SIZE,
    train_epochs=EPOCHS,
    eval_mb_size=BATCH_SIZE,
    device=DEVICE,
    evaluator=eval_plugin,
    eval_every=EVAL_FREQ, # Controls how often 'metrics_logger' runs (Every 1 epoch)
    plugins=[lr_plugin, replay_plugin, weight_plugin, metrics_logger]
)

# ==========================================
# STEP 6: EXECUTION
# ==========================================
print("\n[Execution] Starting Stream...")

# Iterate through the tasks (Experiences) in the benchmark
for experience in benchmark.train_stream:
    task_id = experience.current_experience
    print(f"\n>>> EXPERIENCE {task_id}: {TASK_ORDER[task_id]}")
    
    # TRAIN:
    # 1. 'weight_plugin' runs -> calculates weights
    # 2. Training loop runs for EPOCHS
    # 3. Every 'EVAL_FREQ' epochs, 'eval_streams' are tested
    # 4. 'metrics_logger' runs -> logs F1/Prec/Recall
    strategy.train(experience, eval_streams=[val_benchmark.test_stream])
    
    # FINAL EVAL:
    # Double check performance on the Test set after training is done
    print('   Finalizing Task...')
    strategy.eval(benchmark.test_stream)

# ==========================================
# STEP 7: FINALIZE
# ==========================================
print("\n[Complete] Run finished successfully.")
wandb.finish() # Closes the connection to the cloud