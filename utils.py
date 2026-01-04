import wandb
import numpy as np
from avalanche.logging.wandb_logger import WandBLogger
from avalanche.evaluation.metric_results import AlternativeValues, Image, TensorImage
import torch
from torch import Tensor
from matplotlib.figure import Figure

def log_single_metric_no_viz(self, name, value, x_plot):
    self.step = x_plot

    if name.startswith("WeightCheckpoint"):
        if self.log_artifacts:
            self._log_checkpoint(name, value, x_plot)
        return

    if isinstance(value, AlternativeValues):
        value = value.best_supported_value(
            Image,
            Tensor,
            TensorImage,
            Figure,
            float,
            int
        )

    if not isinstance(value, (Image, TensorImage, Tensor, Figure, float, int)):
        return

    if isinstance(value, Image):
        self.wandb.log({name: self.wandb.Image(value)}, step=self.step)

    elif isinstance(value, Tensor):
        value = np.histogram(value.view(-1).cpu().numpy())
        self.wandb.log({name: self.wandb.Histogram(np_histogram=value)}, step=self.step)

    elif isinstance(value, (float, int, Figure)):
        self.wandb.log({name: value}, step=self.step)

    elif isinstance(value, TensorImage):
        self.wandb.log({name: self.wandb.Image(np.array(value))}, step=self.step)

# Apply monkey patch
WandBLogger.log_single_metric = log_single_metric_no_viz



def extract_stream_confmat(metrics_dict):
    # key example: "ConfusionMatrix_Stream/eval_phase/test_stream"
    for k, v in metrics_dict.items():
        if "ConfusionMatrix_Stream/eval_phase/test_stream" in k:
            # v is a torch.Tensor
            return v.detach().cpu().clone()
    return None

def class_acc_and_macro_f1_from_confmat(confmat: torch.Tensor):
    """
    confmat: (C, C) tensor where rows = true class, cols = predicted class.
    Returns:
      - per_class_acc: dict {class_id: accuracy_i}
      - per_class_f1:  dict {class_id: f1_i}
      - macro_f1: float
    """
    cm = confmat.numpy().astype(np.int64)
    tp = np.diag(cm)
    row_sum = cm.sum(axis=1)  # support per class (true count)
    col_sum = cm.sum(axis=0)  # predicted count per class

    # Per-class accuracy: TP / row_sum
    per_class_acc = {}
    for i in range(len(tp)):
        denom = row_sum[i]
        per_class_acc[i] = (tp[i] / denom) if denom > 0 else 0.0

    # Precision / Recall / F1 per class from CM
    per_class_f1 = {}
    f1_list = []
    for i in range(len(tp)):
        fp = col_sum[i] - tp[i]
        fn = row_sum[i] - tp[i]
        # precision and recall with 0-safe handling
        prec = tp[i] / (tp[i] + fp) if (tp[i] + fp) > 0 else 0.0
        rec  = tp[i] / (tp[i] + fn) if (tp[i] + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_f1[i] = f1
        f1_list.append(f1)

    macro_f1 = float(np.mean(f1_list)) if len(f1_list) > 0 else 0.0
    return per_class_acc, per_class_f1, macro_f1