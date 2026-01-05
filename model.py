import torch
import torch.nn as nn

# ==========================================
# 1. ORIGINAL MODEL (The Baseline)
# ==========================================
class SimpleCNN1D(nn.Module):
    """
    Original 1D CNN. 
    Good for simple tasks, but might be too weak for 20-class NIDS.
    """
    def __init__(self, num_classes=10, input_channels=1):
        super(SimpleCNN1D, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),

            # Block 3
            nn.Conv1d(64, 64, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            
            # Global Pooling
            nn.AdaptiveMaxPool1d(1),
            nn.Dropout(p=0.25),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ==========================================
# 2. BOOSTED MODEL (The Fix)
# ==========================================
class SimpleCNN1DBoosted(nn.Module):
    """
    Upgraded 1D CNN with:
    - More Features (256 vs 64)
    - BatchNorm (Stability)
    - MLP Head (Better Classification)
    """
    def __init__(self, num_classes=10, input_channels=1):
        # FIX: Changed 'SimpleCNN1D' to 'SimpleCNN1DBoosted'
        super(SimpleCNN1DBoosted, self).__init__() 

        self.features = nn.Sequential(
            # --- Block 1: Expand ---
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            # --- Block 2: Deepen ---
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),

            # --- Block 3: Refine ---
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
            
            # --- Global Pooling ---
            nn.AdaptiveMaxPool1d(1)
        )

        # --- Stronger Classifier Head ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4), 
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = self.classifier(x)
        return x