import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class PeriodicEncoder(nn.Module):
    """Fourier numerical feature encoder for sharp tabular decision boundaries."""

    def __init__(self, in_features: int = 165, n_frequencies: int = 3, sigma: float = 0.2):
        super().__init__()
        self.coeffs = nn.Parameter(torch.randn(in_features, n_frequencies) * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = 2 * torch.pi * x.unsqueeze(-1) * self.coeffs.unsqueeze(0)
        return torch.cat([x, torch.sin(v).flatten(1), torch.cos(v).flatten(1)], dim=-1)


class SwiGLUBlock(nn.Module):
    """Gated residual block (SwiGLU) for tabular branch."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.35):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim)
        self.w_gate = nn.Linear(dim, hidden_dim)
        self.w_val = nn.Linear(dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.w_out = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + self.w_out(self.drop(F.silu(self.w_gate(h)) * self.w_val(h)))


class GATResNet(nn.Module):
    """
    Stabilized GAT-ResNet with Periodic-SwiGLU Residual Fusion:
    - 3 GAT layers with self-residual connections and BatchNorm1d
    - Residual connection between layer 1 and layer 2 outputs (x1 + x2) -> 100-dim Z_gat
    - Periodic-SwiGLU skip branch on 165 normalized transaction features
    - embed() returns 100-dim structural node embeddings Z_gat for downstream fusion (BF-25)
    """

    def __init__(
        self,
        in_channels: int = 165,
        hidden_channels: int = 100,
        heads: int = 4,
        out_channels: int = 2,
        dropout_prob: float = 0.35,
        use_skip: bool = True,
    ):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.use_skip = use_skip
        hid_dim = hidden_channels * heads  # 400

        # Graph branch: 3-layer GAT-ResNet
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.1, residual=True)
        self.bn1 = nn.BatchNorm1d(hid_dim)

        self.conv2 = GATConv(hid_dim, hidden_channels, heads=heads, dropout=0.1, residual=True)
        self.bn2 = nn.BatchNorm1d(hid_dim)

        # Bottleneck 400 -> 100 dims (Z_gat)
        self.embed_layer = nn.Linear(hid_dim, hidden_channels)
        self.bn_emb = nn.BatchNorm1d(hidden_channels)

        # 3rd GAT layer for 64-dim neighborhood context
        self.conv3 = GATConv(hidden_channels, 64, heads=1, concat=False, residual=True)
        self.bn3 = nn.BatchNorm1d(64)

        # Tabular skip branch: Periodic numerical embeddings + SwiGLU
        if self.use_skip:
            n_freq = 3
            enc_dim = in_channels * (1 + 2 * n_freq)  # 1155
            self.periodic = PeriodicEncoder(in_channels, n_frequencies=n_freq, sigma=0.2)
            self.tab_proj = nn.Sequential(
                nn.Linear(enc_dim, 128),
                nn.BatchNorm1d(128),
                nn.SiLU(),
                nn.Dropout(dropout_prob),
            )
            self.tab_block = SwiGLUBlock(128, 256, dropout=dropout_prob)
            fusion_dim = 64 + hidden_channels + 128  # 292
        else:
            fusion_dim = 64 + hidden_channels

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(fusion_dim),
            nn.Dropout(dropout_prob),
            nn.Linear(fusion_dim, out_channels),
        )

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Returns 100-dim graph node embeddings Z_gat."""
        x1 = F.elu(self.bn1(self.conv1(x, edge_index)))
        x2 = F.elu(self.bn2(self.conv2(x1, edge_index)))
        return F.elu(self.bn_emb(self.embed_layer(x1 + x2)))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_probs: bool = False,
    ) -> torch.Tensor:
        out, _ = self._forward(x, edge_index)
        return F.softmax(out, dim=1) if return_probs else out

    def forward_with_attention(self, x: torch.Tensor, edge_index: torch.Tensor):
        """Return logits, Z_gat, and (edges, coefficients) for each GAT layer."""
        if self.training:
            raise ValueError("Call eval() before extracting attention")
        attention = []
        logits, embedding = self._forward(x, edge_index, attention)
        return logits, embedding, attention

    def _forward(self, x, edge_index, attention=None):
        def apply_conv(conv, features):
            if attention is None:
                return conv(features, edge_index)
            output, weights = conv(features, edge_index, return_attention_weights=True)
            attention.append(weights)
            return output

        x1 = F.dropout(F.elu(self.bn1(apply_conv(self.conv1, x))), p=self.dropout_prob, training=self.training)
        x2 = F.dropout(F.elu(self.bn2(apply_conv(self.conv2, x1))), p=self.dropout_prob, training=self.training)

        z_gat = F.elu(self.bn_emb(self.embed_layer(x1 + x2)))
        z_g3 = F.elu(
            self.bn3(apply_conv(self.conv3, F.dropout(z_gat, p=self.dropout_prob, training=self.training)))
        )

        if self.use_skip:
            z_tab = self.tab_block(self.tab_proj(self.periodic(x)))
            fused = torch.cat([z_g3, z_gat, z_tab], dim=-1)
        else:
            fused = torch.cat([z_g3, z_gat], dim=-1)

        return self.classifier(fused), z_gat
