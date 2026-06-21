import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import time
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, f1_score
import wandb 

# --- CUSTOM IMPORT ---
from model import LiteNet

# --- CONFIGURATION ---
TASK_NAME = 'NF-CSE-CIC-IDS2018-v2'
BATCH_SIZE = 256
EPOCHS = 20        
LR = 0.001
NUM_CLASSES = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Initialize WandB
wandb.init(
    project="NIDS_Continual_Learning",
    name=f"{TASK_NAME}_SingleTask",
    config={
        "task": TASK_NAME,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "model": "LiteNet",
        "num_classes": NUM_CLASSES,
        "loss": "Weighted CrossEntropy"
    }
)

print(f"Running Single Task Experiment on: {DEVICE}")

# --- DATA LOADING ---
def load_task_data(task_name):
    repo_root = os.getcwd()
    data_dir = os.path.join(repo_root, 'benchmark_data', task_name)
    
    print(f"\n[Loading] {task_name} from {data_dir}...")
    
    try:
        train_np = np.load(os.path.join(data_dir, 'train.npy'))
        val_np   = np.load(os.path.join(data_dir, 'val.npy'))
        test_np  = np.load(os.path.join(data_dir, 'test.npy'))
    except FileNotFoundError:
        print(f"ERROR: Data files not found at {data_dir}")
        exit()

    def clean(arr):
        if np.isnan(arr).any() or np.isinf(arr).any():
            arr = arr[~np.isnan(arr).any(axis=1)]
            arr = arr[~np.isinf(arr).any(axis=1)]
        return arr

    train_np = clean(train_np)
    val_np   = clean(val_np)
    test_np  = clean(test_np)

    X_train, y_train = train_np[:, :-1], train_np[:, -1]
    X_val,   y_val   = val_np[:, :-1],   val_np[:, -1]
    X_test,  y_test  = test_np[:, :-1],  test_np[:, -1]

    print("[Preprocessing] Scaling features...")
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# --- EXECUTION ---
(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_task_data(TASK_NAME)

# --- 2. WEIGHT FIX (FOR 20 CLASSES) ---
print(f"[Setup] Computing Class Weights for {NUM_CLASSES} classes...")

# A. Calculate weights for the classes that actually exist in this file
present_classes = np.unique(y_train)
calculated_weights = compute_class_weight(class_weight='balanced', classes=present_classes, y=y_train)

# B. Create a full weight tensor of 1.0s (default) with size 20
full_weights = torch.ones(NUM_CLASSES, dtype=torch.float)

# C. Map the calculated weights to the correct indices
# If class 13 exists, its weight goes to index 13. All others stay 1.0.
for cls_idx, weight in zip(present_classes, calculated_weights):
    if cls_idx < NUM_CLASSES:
        full_weights[int(cls_idx)] = float(weight)
    else:
        print(f"   [WARNING] Label {cls_idx} found in data but exceeds NUM_CLASSES ({NUM_CLASSES})!")

weights_tensor = full_weights.to(DEVICE)

print(f"   Weights Tensor Size: {weights_tensor.shape}")
wandb.log({"class_weights": weights_tensor.tolist()})

# 3. Create DataLoaders
train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
val_ds   = TensorDataset(torch.FloatTensor(X_val),   torch.LongTensor(y_val))
test_ds  = TensorDataset(torch.FloatTensor(X_test),  torch.LongTensor(y_test))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# 4. Initialize Model & Loss
print(f"[Setup] Initializing LiteNet with {NUM_CLASSES} output classes...")
model = LiteNet(num_classes=NUM_CLASSES).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

# 5. Training Loop
print(f"\n[Training] Starting {EPOCHS} epochs...")
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    avg_loss = train_loss / len(train_loader)
    
    print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
    wandb.log({
        "train_loss": avg_loss,
        "train_acc": acc,
        "epoch": epoch + 1
    })

print(f"Training finished in {time.time() - start_time:.2f}s")

# 7. Final Evaluation
print("\n[Evaluation] Calculating Metrics on Test Set...")
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# 8. Final Report
acc = accuracy_score(all_targets, all_preds)
macro_f1 = f1_score(all_targets, all_preds, average='macro')
report = classification_report(all_targets, all_preds, output_dict=True)

print("-" * 60)
print(f"Overall Accuracy: {acc:.4f}")
print(f"Macro F1-Score:   {macro_f1:.4f}")
print("-" * 60)

wandb.log({
    "test_accuracy": acc,
    "test_macro_f1": macro_f1
})

# Log detailed breakdown per class as a Table
wandb.log({"classification_report": wandb.Table(
    columns=["Class", "Precision", "Recall", "F1-Score", "Support"],
    data=[
        [k, v['precision'], v['recall'], v['f1-score'], v['support']]
        for k, v in report.items() if k.isdigit()
    ]
)})

wandb.finish()