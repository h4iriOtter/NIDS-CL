import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import warnings
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_class_weight
import wandb 

# --- AVALANCHE IMPORTS ---
from avalanche.benchmarks import benchmark_from_datasets
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.training.supervised import Naive
from avalanche.training.plugins import EWCPlugin, SupervisedPlugin
from avalanche.logging import BaseLogger, InteractiveLogger
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    loss_metrics,
    forgetting_metrics,
    bwt_metrics,
    forward_transfer_metrics,
    confusion_matrix_metrics
)

# --- SKLEARN IMPORTS ---
from sklearn.metrics import (
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score, 
    balanced_accuracy_score,
    matthews_corrcoef,
    average_precision_score
)
from sklearn.exceptions import UndefinedMetricWarning

# Suppress specific sklearn warnings to clean up logs
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message="No positive class found in y_true")

# --- CUSTOM MODEL ---
from model import LiteNet

# --- CONFIGURATION ---
BATCH_SIZE = 256
EPOCHS = 20         
LR = 0.001
EWC_LAMBDA = 5000.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 20

TASKS = [
    'NF-BoT-IoT-v2',
    'NF-ToN-IoT-v2'
]

# ==============================================================================
# 1. METRICS
# ==============================================================================

class SklearnClassMetric(PluginMetric):
    def __init__(self, metric_name, average='macro'):
        super().__init__()
        self.metric_name = metric_name
        self.average = average
        self.y_true = []
        self.y_pred = []

    def reset(self):
        self.y_true = []
        self.y_pred = []

    def result(self):
        if len(self.y_true) == 0: return 0.0
        y_true = np.concatenate(self.y_true)
        y_pred = np.concatenate(self.y_pred)
        
        if self.metric_name == 'Balanced_Accuracy':
            return balanced_accuracy_score(y_true, y_pred)
        
        if self.metric_name == 'MCC':
            return matthews_corrcoef(y_true, y_pred)

        if self.metric_name == 'F1':
            return f1_score(y_true, y_pred, average=self.average, zero_division=0)
        elif self.metric_name == 'Precision':
            return precision_score(y_true, y_pred, average=self.average, zero_division=0)
        elif self.metric_name == 'Recall':
            return recall_score(y_true, y_pred, average=self.average, zero_division=0)
        
        return 0.0

    def after_eval_iteration(self, strategy):
        self.y_true.append(strategy.mb_y.cpu().numpy())
        self.y_pred.append(torch.argmax(strategy.mb_output, dim=1).cpu().numpy())

    def after_eval_exp(self, strategy):
        val = self.result()
        if self.metric_name in ['Balanced_Accuracy', 'MCC']:
            name = f"Sklearn_{self.metric_name}/Exp_{strategy.experience.current_experience}"
        else:
            name = f"Sklearn_{self.metric_name}_{self.average}/Exp_{strategy.experience.current_experience}"
        
        return [MetricValue(self, name, val, strategy.clock.train_iterations)]
    
    def __str__(self):
        return f"{self.metric_name}_{self.average}"

class SklearnProbaMetric(PluginMetric):
    def __init__(self, metric_name, num_classes, average='macro'):
        super().__init__()
        self.metric_name = metric_name
        self.num_classes = num_classes
        self.average = average
        self.y_true = []
        self.y_probs = []

    def reset(self):
        self.y_true = []
        self.y_probs = []

    def result(self):
        if len(self.y_true) == 0: return 0.0
        y_true = np.concatenate(self.y_true)
        y_probs = np.concatenate(self.y_probs)
        
        # Binarize to ensure shape compatibility
        y_true_bin = label_binarize(y_true, classes=np.arange(self.num_classes))
        
        try:
            if self.metric_name == 'ROC_AUC':
                return roc_auc_score(y_true_bin, y_probs, average=self.average, multi_class='ovr')
            elif self.metric_name == 'Average_Precision':
                return average_precision_score(y_true_bin, y_probs, average=self.average)
        except ValueError:
            return 0.0
        return 0.0

    def after_eval_iteration(self, strategy):
        self.y_true.append(strategy.mb_y.cpu().numpy())
        probs = torch.softmax(strategy.mb_output, dim=1)
        self.y_probs.append(probs.cpu().numpy())

    def after_eval_exp(self, strategy):
        val = self.result()
        name = f"Sklearn_{self.metric_name}_{self.average}/Exp_{strategy.experience.current_experience}"
        return [MetricValue(self, name, val, strategy.clock.train_iterations)]
    
    def __str__(self):
        return f"{self.metric_name}_{self.average}"

# ==============================================================================
# 2. LOGGING & DATA
# ==============================================================================
class SafeWandBLogger(BaseLogger):
    def __init__(self, project_name, run_name, config):
        super().__init__()
        wandb.init(project=project_name, name=run_name, config=config)

    def log_single_metric(self, name, value, x_plot):
        if isinstance(value, (int, float, np.number)):
            wandb.log({name: value, "step": x_plot})
    
    def close(self):
        wandb.finish()

def create_robust_dataset(x_data, y_data, task_id):
    x_tensor = torch.FloatTensor(x_data)
    y_tensor = torch.LongTensor(y_data)
    tl_tensor = torch.full((len(y_data),), task_id, dtype=torch.long)
    base_dataset = TensorDataset(x_tensor, y_tensor)
    
    try:
        return AvalancheDataset(base_dataset, task_labels=tl_tensor)
    except TypeError:
        ds = AvalancheDataset(base_dataset)
        ds.targets_task_labels = tl_tensor
        return ds

def load_and_create_benchmark():
    repo_root = os.getcwd()
    print("Loading data...")
    raw_train_x, raw_train_y = [], []
    raw_test_x, raw_test_y = [], []
    task_boundaries = [] 

    for task in TASKS:
        path = os.path.join(repo_root, 'benchmark_data', task)
        try:
            t_np = np.load(os.path.join(path, 'train.npy'))
            test_np = np.load(os.path.join(path, 'test.npy'))
        except FileNotFoundError:
            print(f"Error: {path} not found.")
            exit()
        
        t_np = t_np[~np.isnan(t_np).any(axis=1)]
        test_np = test_np[~np.isnan(test_np).any(axis=1)]

        raw_train_x.append(t_np[:, :-1])
        raw_train_y.append(t_np[:, -1])
        raw_test_x.append(test_np[:, :-1])
        raw_test_y.append(test_np[:, -1])
        task_boundaries.append(len(t_np))

    X_train_all = np.vstack(raw_train_x)
    scaler = StandardScaler()
    X_train_all = scaler.fit_transform(X_train_all)

    train_datasets = []
    test_datasets = []
    current_idx = 0
    for i, count in enumerate(task_boundaries):
        x_chunk = X_train_all[current_idx : current_idx + count]
        d_train = create_robust_dataset(x_chunk, raw_train_y[i], task_id=i)
        train_datasets.append(d_train)
        
        x_test_local = scaler.transform(raw_test_x[i])
        d_test = create_robust_dataset(x_test_local, raw_test_y[i], task_id=i)
        test_datasets.append(d_test)
        current_idx += count

    return benchmark_from_datasets(train=train_datasets, test=test_datasets)

# ==============================================================================
# 3. WEIGHTED LOSS PLUGIN
# ==============================================================================
class AvalancheLossWrapper:
    def __init__(self, strategy, loss_fn):
        self.strategy = strategy
        self.loss_fn = loss_fn
    
    def __call__(self):
        return self.loss_fn(self.strategy.mb_output, self.strategy.mb_y)

class WeightedLossPlugin(SupervisedPlugin):
    def before_training_exp(self, strategy, **kwargs):
        print(f"\n[Plugin] Calculating Class Weights...")
        try:
            train_y = np.array(strategy.experience.dataset.targets)
        except:
            dl = DataLoader(strategy.experience.dataset, batch_size=5000)
            all_y = []
            for batch in dl: all_y.extend(batch[1].numpy())
            train_y = np.array(all_y)

        classes = np.unique(train_y)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_y)
        full_weights = torch.ones(NUM_CLASSES, dtype=torch.float)
        for cls_idx, w in zip(classes, weights):
            if cls_idx < NUM_CLASSES: full_weights[int(cls_idx)] = float(w)
        
        base_loss = nn.CrossEntropyLoss(weight=full_weights.to(strategy.device))
        strategy.criterion = AvalancheLossWrapper(strategy, base_loss)

# ==============================================================================
# 4. MAIN RUN
# ==============================================================================
if __name__ == "__main__":
    scenario = load_and_create_benchmark()

    # --- DEFINE METRICS ---
    metrics_list = [
        accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=False, epoch=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
        forward_transfer_metrics(experience=True, stream=True),
        confusion_matrix_metrics(num_classes=NUM_CLASSES, save_image=False, normalize='true'),

        # Sklearn Scientific Metrics
        SklearnClassMetric('Balanced_Accuracy'),      
        SklearnClassMetric('MCC'),                    
        SklearnClassMetric('F1', average='macro'),
        SklearnClassMetric('Precision', average='macro'),
        SklearnClassMetric('Recall', average='macro'),
        
        # Probabilistic Metrics
        SklearnProbaMetric('ROC_AUC', num_classes=NUM_CLASSES, average='macro'),
        SklearnProbaMetric('Average_Precision', num_classes=NUM_CLASSES, average='macro')
    ]

    safe_wandb_logger = SafeWandBLogger(
        project_name="NIDS_Continual_Learning",
        run_name="Naive+EWCPlugin+Weighted",
        config={"strategy": "EWC", "lambda": EWC_LAMBDA}
    )
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        *metrics_list,
        loggers=[safe_wandb_logger, interactive_logger]
    )

    model = LiteNet(num_classes=NUM_CLASSES)

    # --- STRATEGY (FIXED) ---
    strategy = Naive(
        model=model,
        optimizer=AdamW(model.parameters(), lr=LR),
        criterion=nn.CrossEntropyLoss(), 
        train_mb_size=BATCH_SIZE,
        train_epochs=EPOCHS,
        eval_mb_size=BATCH_SIZE,
        eval_every=1,  # <--- CRITICAL FIX: Required for forward_transfer_metrics
        device=DEVICE,
        plugins=[
            EWCPlugin(ewc_lambda=EWC_LAMBDA), 
            WeightedLossPlugin()              
        ],
        evaluator=eval_plugin
    )

    print(f"Starting Experiment on {DEVICE}...")

    try:
        for experience in scenario.train_stream:
            print(f"\n--- Task {experience.current_experience} ---")
            strategy.train(experience)
            print("Evaluating...")
            strategy.eval(scenario.test_stream)
    finally:
        wandb.finish()