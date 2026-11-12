"""
Stage 3D Deep Learning Architectures:
- Model D4: CNN Spatial-Field Inverse Model (Direct Regression with Stress Tensor Ablations)
- Model D5: Spatial-Field U-Net Heatmap & Spatial Uncertainty Model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class DefectInverseCNN(nn.Module):
    """
    Model D4: Convolutional Neural Network for Direct Spatial Defect Localization.
    Supports in_channels = 1 (Von Mises only) or 4 (Full in-plane stress tensor).
    Accepts scalar metadata (aspect_ratio, and optional defect_d_over_Dc).
    """
    def __init__(self, in_channels=1, meta_dim=2, out_dim=2):
        super().__init__()
        self.in_channels = in_channels
        self.meta_dim = meta_dim
        self.out_dim = out_dim
        
        self.enc1 = ConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(128, 256)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.head = nn.Sequential(
            nn.Linear(256 + meta_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, out_dim),
            nn.Sigmoid()  # normalized coordinates are in [0, 1]
        )
        
    def forward(self, x_field, x_meta):
        h = self.pool1(self.enc1(x_field))
        h = self.pool2(self.enc2(h))
        h = self.pool3(self.enc3(h))
        h = self.enc4(h)
        
        feat = self.gap(h).flatten(1)
        if self.meta_dim > 0:
            feat = torch.cat([feat, x_meta], dim=1)
            
        out = self.head(feat)
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class DefectHeatmapUNet(nn.Module):
    """
    Model D5: U-Net Architecture Predicting a 2D Spatial Heatmap of Defect Location.
    Enables soft-argmax coordinate extraction and empirical spatial uncertainty estimation.
    """
    def __init__(self, in_channels=1):
        super().__init__()
        # Encoder
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(64, 32)
        
        self.outc = nn.Conv2d(32, 1, kernel_size=1)
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        u1 = self.up1(x4)
        u1 = torch.cat([u1, x3], dim=1)
        u1 = self.conv_up1(u1)
        
        u2 = self.up2(u1)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.conv_up2(u2)
        
        u3 = self.up3(u2)
        u3 = torch.cat([u3, x1], dim=1)
        u3 = self.conv_up3(u3)
        
        logits = self.outc(u3)
        return logits

def soft_argmax_2d(heatmap_logits, beta=50.0):
    """
    Extracts 2D coordinate expectation (xi, eta) and spatial uncertainty covariance
    from predicted heatmap logits using a temperature-scaled spatial softmax.
    
    Returns:
    --------
    coords_pred : Tensor of shape (B, 2) in normalized [0, 1] range [xi, eta]
    unc_radius  : Tensor of shape (B,) representing 1-sigma uncertainty radius in [0, 1]
    cov_matrix  : Tensor of shape (B, 2, 2) representing spatial covariance
    """
    B, C, H, W = heatmap_logits.shape
    assert C == 1
    
    # Flatten spatial dimensions and apply temperature softmax
    flat_logits = heatmap_logits.view(B, -1) * beta
    prob = F.softmax(flat_logits, dim=1).view(B, H, W)
    
    # Coordinate grids in [0, 1]
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0.0, 1.0, H, device=heatmap_logits.device),
        torch.linspace(0.0, 1.0, W, device=heatmap_logits.device),
        indexing='ij'
    )
    
    # Compute 1st moments: expected coordinates
    exp_x = torch.sum(prob * grid_x, dim=(1, 2))
    exp_y = torch.sum(prob * grid_y, dim=(1, 2))
    coords_pred = torch.stack([exp_x, exp_y], dim=1)
    
    # Compute 2nd central moments: spatial covariance
    diff_x = grid_x.unsqueeze(0) - exp_x.view(B, 1, 1)
    diff_y = grid_y.unsqueeze(0) - exp_y.view(B, 1, 1)
    
    var_x = torch.sum(prob * (diff_x ** 2), dim=(1, 2))
    var_y = torch.sum(prob * (diff_y ** 2), dim=(1, 2))
    cov_xy = torch.sum(prob * diff_x * diff_y, dim=(1, 2))
    
    # Uncertainty radius = sqrt(var_x + var_y)
    unc_radius = torch.sqrt(torch.clamp(var_x + var_y, min=1e-8))
    
    # Covariance tensor (B, 2, 2)
    cov_matrix = torch.zeros((B, 2, 2), device=heatmap_logits.device)
    cov_matrix[:, 0, 0] = var_x
    cov_matrix[:, 1, 1] = var_y
    cov_matrix[:, 0, 1] = cov_xy
    cov_matrix[:, 1, 0] = cov_xy
    
    return coords_pred, unc_radius, cov_matrix

