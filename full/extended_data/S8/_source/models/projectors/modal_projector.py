import torch
import torch.nn as nn


class ModalProjector(nn.Module):














    def __init__(
        self,
        d_model: int = 1024,
        d_llm: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(d_model, d_llm),
            nn.LayerNorm(d_llm),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:






        return self.proj(x)
