import torch
import torch.nn as nn

def _get_hidden_dims(embed_dim):

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

class CLSIB(nn.Module):
    """
    The encoder runs on the sender and produces (mu, logvar) -> z which only crosses the channel.
    The decoder runs on the receiver and reconstructs an in-distribution CLS embedding for the post-blocks.
    """

    def __init__(self, embed_dim, bottleneck_dim, use_kl_div=True, device="cuda", free_bits_per_dim=0.0):
        super().__init__()
        self.device = device
        self.use_kl_div = use_kl_div
        self.bottleneck_dim = bottleneck_dim
        self.free_bits_per_dim = free_bits_per_dim

        h_big, h_small = _get_hidden_dims(embed_dim)
        
        if bottleneck_dim < h_small:
            self.encoder = nn.Sequential(
                nn.Linear(embed_dim, h_big),
                nn.GELU(), 
                nn.LayerNorm(h_big),
                nn.Linear(h_big, h_small),
                nn.GELU(), 
                nn.LayerNorm(h_small),
            )
            enc_out = h_small
        else:
            self.encoder = nn.Sequential(
                nn.Linear(embed_dim, h_big),
                nn.GELU(), 
                nn.LayerNorm(h_big),
            )
            enc_out = h_big
        
        self.enc_mu = nn.Linear(enc_out, bottleneck_dim)
        self.enc_log_var = nn.Linear(enc_out, bottleneck_dim)
        
        if bottleneck_dim < h_small:
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck_dim, h_small),
                nn.GELU(), 
                nn.LayerNorm(h_small),
                nn.Linear(h_small, h_big),
                nn.GELU(), 
                nn.LayerNorm(h_big),
                nn.Linear(h_big, embed_dim),
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck_dim, h_big),
                nn.GELU(), 
                nn.LayerNorm(h_big),
                nn.Linear(h_big, embed_dim),
            )

    def encode(self, cls_token):
        """
        Sender-side: cls_token -> (z, kl, mu, logvar).
        Samples z when training and use_kl_div=True; otherwise z = mu for deployment.
        """
        h = self.encoder(cls_token)
        mu = self.enc_mu(h)
        logvar = self.enc_log_var(h)

        if self.use_kl_div and self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std

        else:
            z = mu

        if self.use_kl_div:
            # KL[ q(z|x) || N(0, I) ] per sample, then averaged.
            #kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            
            kl_per_dim_raw = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=0)
            kl_per_dim_clamped = torch.clamp(kl_per_dim_raw, min=self.free_bits_per_dim) 
            kl_for_loss = kl_per_dim_clamped.sum()  
            kl_actual = kl_per_dim_raw.sum()
        else:
            kl_for_loss = torch.zeros((), device=cls_token.device)
            kl_actual = torch.zeros((), device=cls_token.device)

        return z, kl_for_loss, mu, logvar, kl_actual

    def decode(self, z):
        """Receiver-side: z -> reconstructed CLS embedding."""
        return self.decoder(z)