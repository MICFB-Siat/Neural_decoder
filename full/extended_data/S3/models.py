import torch, math
from torch import nn
CIFTI_DIM=4096
MODES_DIM=2000
HIDDEN_DIM=1024
class GaussHead(nn.Module):
    def __init__(self, d_in: int, lat_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.mu_head = nn.Linear(hidden, lat_dim)
        self.lv_head = nn.Linear(hidden, lat_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu_head(h), self.lv_head(h).clamp(-8, 4)

class ClsHead(nn.Module):
    def __init__(self, lat_dim: int, hidden: int = 256,
                 n_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(lat_dim),
            nn.Linear(lat_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, mu):
        return self.net(mu.mean(dim=1))

class GlobalPoE(nn.Module):


    def __init__(self):
        super().__init__()
        self.cifti_mean = nn.Linear(CIFTI_DIM, HIDDEN_DIM)
        self.modes_mean = nn.Linear(MODES_DIM, HIDDEN_DIM)
        self.cifti_log_precision = nn.Parameter(torch.full((HIDDEN_DIM,), -math.log(2.0)))
        self.modes_log_precision = nn.Parameter(torch.full((HIDDEN_DIM,), -math.log(2.0)))

    def forward(self, cifti: torch.Tensor, modes: torch.Tensor):
        cifti_mean = self.cifti_mean(cifti)
        modes_mean = self.modes_mean(modes)
        cifti_precision = torch.exp(3.0 * torch.tanh(self.cifti_log_precision / 3.0))
        modes_precision = torch.exp(3.0 * torch.tanh(self.modes_log_precision / 3.0))
        total_precision = cifti_precision + modes_precision
        fused_mean = (
            cifti_mean * cifti_precision + modes_mean * modes_precision
        ) / total_precision
        fused_variance = total_precision.reciprocal().expand_as(fused_mean)
        return fused_mean[:, None], {
            "fused_mean": fused_mean,
            "fused_variance": fused_variance,
            "cifti_precision": cifti_precision.expand_as(fused_mean),
            "modes_precision": modes_precision.expand_as(fused_mean),
        }

class AnchoredPoE(GlobalPoE):


    def __init__(self):
        super().__init__()
        with torch.no_grad():
            self.cifti_log_precision.zero_()
            self.modes_log_precision.fill_(-2.0)
