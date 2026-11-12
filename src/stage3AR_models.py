"""
Stage 3A-R Neural Network Architectures and Loss Functions
Includes:
- Shallow baselines: ANN-128, ANN-256, ANN-512
- Deep MLPs: Model D1, Model D2, Model D3
- Residual MLP: ResMLP
- Multi-Output Regional Stress Network: MultiOutputANN
- Loss functions: MSE, HuberLoss, PhysicallyWeightedMSELoss
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple

class ShallowANN(nn.Module):
    """Single hidden-layer baseline network."""
    def __init__(self, in_features: int, hidden_neurons: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, 1)
        )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DeepMLP_D1(nn.Module):
    """3 Hidden Layers: Input -> 256 -> 256 -> 128 -> 1"""
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DeepMLP_D2(nn.Module):
    """3 Hidden Layers High Capacity: Input -> 512 -> 256 -> 128 -> 1"""
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DeepMLP_D3(nn.Module):
    """4 Hidden Layers: Input -> 512 -> 256 -> 128 -> 64 -> 1"""
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ResBlock(nn.Module):
    """Residual fully-connected block with skip connection."""
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc2.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out + residual)
        return out

class ResMLP(nn.Module):
    """Residual MLP: Input -> 256 -> ResBlock(256) -> ResBlock(256) -> 128 -> 1"""
    def __init__(self, in_features: int):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU()
        )
        self.block1 = ResBlock(256)
        self.block2 = ResBlock(256)
        self.out_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        nn.init.kaiming_normal_(self.in_proj[0].weight, nonlinearity='relu')
        nn.init.zeros_(self.in_proj[0].bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.block1(h)
        h = self.block2(h)
        return self.out_head(h)

class MultiOutputANN(nn.Module):
    """Shared trunk network with two heads: Central-hole Kt and Defect-hole Kt."""
    def __init__(self, in_features: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.head_center = nn.Linear(128, 1)
        self.head_defect = nn.Linear(128, 1)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
        nn.init.xavier_normal_(self.head_center.weight)
        nn.init.zeros_(self.head_center.bias)
        nn.init.xavier_normal_(self.head_defect.weight)
        nn.init.zeros_(self.head_defect.bias)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        out_c = self.head_center(h)
        out_d = self.head_defect(h)
        return out_c, out_d

class PhysicallyWeightedMSELoss(nn.Module):
    """Weighted MSE assigning higher penalty to high-gradient physical regimes."""
    def __init__(self):
        super().__init__()
        
    def forward(self, preds: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        sq_err = (preds - targets) ** 2
        weighted_err = sq_err * weights
        return torch.mean(weighted_err)

