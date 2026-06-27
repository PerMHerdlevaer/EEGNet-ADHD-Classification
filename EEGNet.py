import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class EEGWindowDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, n_channels, window_size)
        # y: (N,)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)  
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EEGNetBinary(nn.Module):
    """
    Faithful EEGNet (Lawhern et al.):
    Temporal Conv -> Depthwise Spatial Conv -> Separable Conv -> Sigmoid logit
    Input: (B, C, T)
    Output: (B, 1) logits
    """
    def __init__(self, n_channels=32, window_size=2500, 
                F1=8, D=2, kernel_length=125, dropout_p=0.5):
        super().__init__()
        F2 = F1 * D

        # --- Block 1: Temporal conv ---
        self.conv_temporal = nn.Conv2d(
            1, F1, kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2), bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # --- Depthwise spatial conv ---
        self.conv_spatial = nn.Conv2d(
            F1, F1 * D, kernel_size=(n_channels, 1),
            groups=F1, bias=False
        )


        self.bn2 = nn.BatchNorm2d(F1 * D)


        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(dropout_p)

        # --- Block 2: Separable conv ---
        self.sep_depthwise = nn.Conv2d(
            F1 * D, F1 * D, kernel_size=(1, 16),
            padding=(0, 8), groups=F1 * D, bias=False
        )
        self.sep_pointwise = nn.Conv2d(
            F1 * D, F2, kernel_size=(1, 1), bias=False
        )


        self.bn3 = nn.BatchNorm2d(F2)

        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(dropout_p)

        # compute flatten dim dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, window_size)
            feat_dim = self._forward_features(dummy).shape[1]

        self.classifier = nn.Linear(feat_dim, 1)  # 1 logit for BCE

    def _forward_features(self, x):
        # x: (B, C, T) -> (B, 1, C, T)
        x = x.unsqueeze(1)

        x = self.conv_temporal(x)
        x = self.bn1(x)
        x = F.elu(x)

        x = self.conv_spatial(x)
        x = self.bn2(x)
        x = F.elu(x)

        x = self.pool1(x)
        x = self.drop1(x)

        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)
        x = self.bn3(x)
        x = F.elu(x)

        x = self.pool2(x)
        x = self.drop2(x)

        return x.flatten(start_dim=1)

    def forward(self, x):
        x = self._forward_features(x)
        return self.classifier(x)
