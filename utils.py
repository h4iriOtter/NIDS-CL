import re
import torch
import numpy as np
from avalanche.logging.wandb_logger import WandBLogger
from avalanche.evaluation.metric_results import AlternativeValues, Image, TensorImage
from avalanche.logging import TextLogger
from torch import Tensor
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# ==========================================
# 1. WANDB MONKEY PATCH (Runs on Import)
# ==========================================
def log_single_metric_no_viz(self, name, value, x_plot):
    # Check WandB's internal step counter
    current_wandb_step = self.wandb.run.step
    
    # If Avalanche's counter (x_plot) is behind WandB's counter, 
    # force it to catch up.
    if x_plot < current_wandb_step:
        self.step = current_wandb_step
    else:
        self.step = x_plot

    if name.startswith("WeightCheckpoint"):
        return

    if isinstance(value, AlternativeValues):
        value = value.best_supported_value(Image, Tensor, TensorImage, Figure, float, int)

    if not isinstance(value, (Image, TensorImage, Tensor, Figure, float, int)):
        return

    if isinstance(value, Tensor):
        if "ConfusionMatrix" in name:
            # --- HEATMAP VISUALIZATION CODE ---
            val_np = value.cpu().numpy()
            
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111)
            
            # Use a blue colormap for better readability
            cax = ax.matshow(val_np, cmap='Blues')
            fig.colorbar(cax)
            
            ax.set_title(name.split('/')[-1], pad=20)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            
            # Ticks
            num_classes = val_np.shape[0]
            ax.set_xticks(np.arange(num_classes))
            ax.set_yticks(np.arange(num_classes))
            
            # Add numbers inside the heatmap squares
            thresh = val_np.max() / 2.
            for i in range(val_np.shape[0]):
                for j in range(val_np.shape[1]):
                    count = int(val_np[i, j])
                    if count > 0:
                        ax.text(j, i, str(count),
                                ha="center", va="center",
                                color="white" if val_np[i, j] > thresh else "black",
                                fontsize=8)

            self.wandb.log({name: self.wandb.Image(fig)}, step=self.step)
            plt.close(fig)
            # ----------------------------------
            
        else:
            value = np.histogram(value.view(-1).cpu().numpy())
            self.wandb.log({name: self.wandb.Histogram(np_histogram=value)}, step=self.step)

    elif isinstance(value, (float, int)):
        self.wandb.log({name: value}, step=self.step)
    
    pass

# Apply patch immediately
WandBLogger.log_single_metric = log_single_metric_no_viz

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def extract_stream_confmat(metrics_dict, mode='eval'):
    """
    Extracts the Confusion Matrix based on the mode (train or eval).
    """
    # 1. Decide which key to look for based on mode
    if mode == 'train':
        keyword = "train_phase/train_stream"
    else:
        keyword = "eval_phase/test_stream"
    
    # 2. Search the metrics dictionary
    for k, v in metrics_dict.items():
        if "ConfusionMatrix_Stream" in k and keyword in k:
            return v.detach().cpu().clone()
            
    return None

def calculate_smart_f1(conf_matrix, strict=True):
    """
    Args:
        strict (bool): 
            True = Local F1 (Ignores missing classes).
            False = Global F1 (Penalizes missing classes with 0.0).
    """
    conf_matrix = conf_matrix.float().cpu().numpy()
    
    tp = np.diag(conf_matrix)
    pred_sum = np.sum(conf_matrix, axis=0)
    true_sum = np.sum(conf_matrix, axis=1) # Support
    
    fp = pred_sum - tp
    fn = true_sum - tp

    if strict:
        # --- LOCAL (STRICT) MODE ---
        # Only evaluate classes that exist in this batch or were predicted
        valid_mask = (true_sum > 0) | (pred_sum > 0)
        if np.sum(valid_mask) == 0: return 0.0, 0.0, 0.0
        
        # Filter the arrays
        tp = tp[valid_mask]
        fp = fp[valid_mask]
        fn = fn[valid_mask]
    else:
        # --- GLOBAL MODE ---
        # Evaluate ALL classes. Rows with 0 support will result in 0.0 F1.
        pass

    # Safe division
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    
    f1 = 2 * (precision * recall) / (precision + recall)
    f1 = np.nan_to_num(f1) 

    return np.mean(f1), np.mean(precision), np.mean(recall)

# ==========================================
# 3. LOGGERS (Moved from Main)
# ==========================================
class NaturalSortTextLogger(TextLogger):
    def log_metrics(self, metric_values):
        def natural_keys(metric):
            text = metric.name
            return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', text)]
        sorted_metrics = sorted(metric_values, key=natural_keys)
        super().log_metrics(sorted_metrics)