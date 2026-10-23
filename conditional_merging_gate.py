import torch.nn as nn
import torch

projection_dim_reduced = 64

def _get_hidden_dims(embed_dim):
    """
    For ViT-B, uses the established 512/256 sizing.
    For smaller backbones (DeiT-S, DeiT-T), uses h_big = embed_dim to avoid introducing an implicit bottleneck before the explicit one.
    """
    if embed_dim == 768:
        return 512, 256
    elif embed_dim == 384:
        return 384, 192
    elif embed_dim == 192:
        return 192, 96
    else:
        raise ValueError(
            f"Unsupported embed_dim={embed_dim}. "
            f"Add an explicit entry to _get_hidden_dims."
        )
    
class ConditionalMergingGate(nn.Module):
    def __init__(self, embed_dim, bottleneck_dim, K):
        super().__init__()
        self.K = K
        
        h_big, _ = _get_hidden_dims(embed_dim)

        self.gate = nn.Sequential(
            nn.Linear(bottleneck_dim, h_big),
            nn.GELU(),
            nn.LayerNorm(h_big),
            nn.Linear(h_big, projection_dim_reduced * self.K),
        )

        self.projection_back = nn.Linear(projection_dim_reduced, embed_dim)

    def forward(self, z, patches):
        # z: [B, bottleneck_dim]
        # patches: [B, N, embed_dim]
        B, N, D = patches.shape
        gate_values = self.gate(z).reshape(B, self.K, projection_dim_reduced)  # [B, K, projection_dim_reduced]
        gate_values = self.projection_back(gate_values)  # [B, K, embed_dim]
        logits = torch.einsum('bkd,bnd->bnk', gate_values, patches) / (D ** 0.5)  # [B, N, K] # Scale by sqrt(embed_dim) for stability and distribute the patches across the K clusters
        assignments = torch.softmax(logits, dim=-1)  # [B, N, K]
        return assignments, gate_values
    
    def merge(self, patches, assignments, return_size=True):
        # patches: [B, N, D]
        # assignments: [B, N, K]
        weights = assignments.transpose(-1, -2)  # [B, K, N]
        size = weights.sum(dim=-1, keepdim=True)  # [B, K, 1] 
        merged = torch.einsum('bkn,bnd->bkd', weights, patches) / (size + 1e-8)  # [B, K, embed_dim] # Avoid division by zero
        if return_size:
            return merged, size
        return merged