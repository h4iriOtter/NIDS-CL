import torch
import torch.nn as nn

class SimpleCNN1D(nn.Module):
    """
    1D Convolutional Neural Network
    Adapted from Avalanche's SimpleCNN (originally for 2D images).
    Optimized for Network Traffic (Sequential/Tabular Data).
    """
    def __init__(self, num_classes=10, input_channels=1):
        super(SimpleCNN1D, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            # Input: (Batch, 1, Features)
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
            
            # Global Pooling: Forces output to fixed size regardless of input length
            # Output becomes (Batch, 64, 1) -> Flattened to (Batch, 64)
            nn.AdaptiveMaxPool1d(1),
            nn.Dropout(p=0.25),
        )

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Ensure input is 3D: (Batch, Channels, Length)
        # If input is (Batch, Features), we add the Channel dimension.
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten features
        x = self.classifier(x)
        return x

# --- USAGE EXAMPLE ---
# model = SimpleCNN1D(num_classes=25, input_channels=1)
# print(model)