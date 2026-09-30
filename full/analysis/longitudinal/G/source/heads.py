from torch import nn
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
