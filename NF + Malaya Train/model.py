import torch.nn as nn
import torch

class LiteNet(nn.Module):
    def __init__(self, num_classes):
        super(LiteNet, self).__init__()

        
        conv_in_channels = 1

        self.branch1x1 = nn.Sequential(
            nn.Conv1d(conv_in_channels, 16, kernel_size=1),
            nn.ReLU()
        )

        self.branch3x3 = nn.Sequential(
            nn.Conv1d(conv_in_channels, 24, kernel_size=1),
            nn.Conv1d(24, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.branch5x5 = nn.Sequential(
            nn.Conv1d(conv_in_channels, 8, kernel_size=1),
            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.ReLU()
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(conv_in_channels, 16, kernel_size=1),
            nn.ReLU()
        )

        # Robust to input length; ensures pooled length = 4 → 64*4 = 256
        self.global_pool = nn.AdaptiveAvgPool1d(4)
        self.fc1 = nn.Linear(256, 128)
        self.activation5 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.activation6 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch, num_features) → (batch, 1, seq_len)
        if x.dim() == 3:
            x = x.squeeze(1)
        x = x.float().unsqueeze(1)

        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        conv_out = torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], 1)
        pool_out = self.global_pool(conv_out).flatten(start_dim=1)

        fc1_out = self.activation5(self.fc1(pool_out))
        fc2_out = self.activation6(self.fc2(fc1_out))
        x = self.fc3(fc2_out)
        #print(x.shape)

        return x

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        
        # We split the target output channels across the 4 branches
        # e.g., if out_channels=32, each branch gets 8 filters.
        branch_channels = out_channels // 4

        # Branch 1: 1x1 conv
        self.branch1x1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )

        # Branch 2: 1x1 -> 3x3 conv
        self.branch3x3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )

        # Branch 3: 1x1 -> 5x5 conv
        self.branch5x5 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )

        # Branch 4: MaxPool -> 1x1 conv
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )

    def forward(self, x):
        b1 = self.branch1x1(x)
        b2 = self.branch3x3(x)
        b3 = self.branch5x5(x)
        b4 = self.branch_pool(x)
        return torch.cat([b1, b2, b3, b4], 1)


class BigNet(nn.Module):
    def __init__(self, num_classes, input_dim=42): # input_dim might be ~42 or ~70 depending on dataset
        super(BigNet, self).__init__()

        # --- FEATURE EXTRACTOR ---
        # We stack two Inception Blocks.
        # Block 1: Takes 1 channel (reshaped input), outputs 64 channels
        self.block1 = InceptionBlock(in_channels=1, out_channels=64)
        
        # Block 2: Takes 64 channels, outputs 128 channels
        self.block2 = InceptionBlock(in_channels=64, out_channels=128)

        # Block 3: Takes 128 channels, outputs 256 channels (Optional, makes it very deep)
        self.block3 = InceptionBlock(in_channels=128, out_channels=256)

        # Global Pooling: Compresses time dimension to fixed size 4
        # Output shape here will be: 256 channels * 4 = 1024 features
        self.global_pool = nn.AdaptiveAvgPool1d(4)

        # --- CLASSIFIER (MLP) ---
        flat_features = 256 * 4  # 1024
        
        self.classifier = nn.Sequential(
            nn.Linear(flat_features, 512),
            nn.BatchNorm1d(512),   # Stabilizes training
            nn.ReLU(),
            nn.Dropout(0.3),       # <--- CRITICAL: Prevents overfitting to Benign class
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),       # <--- CRITICAL
            
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # 1. Reshape Input: (Batch, Features) -> (Batch, 1, Features)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # 2. Extract Features
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # 3. Pool and Flatten
        x = self.global_pool(x)
        x = x.flatten(start_dim=1) # Flatten (Batch, 256, 4) -> (Batch, 1024)
        
        # 4. Classify
        x = self.classifier(x)
        
        return x