import os
import torch
import torch.nn as nn
import timm

import cls_ib
import conditional_merging_gate

def _get_hidden_dims(embed_dim):
    """Hidden dimensions for IB encoder/decoder and merging gate.
    
    For ViT-B, uses the established 512/256 sizing that matches our trained
    checkpoints. For smaller backbones (DeiT-S, DeiT-T), uses h_big = embed_dim
    to avoid introducing an implicit bottleneck before the explicit one.
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



class ConditionedViT(nn.Module):
    def __init__(self, split_block_nr=9, use_kl_div=True, device="cuda", pretrained_path="", model_name="", freeze_backbone=True, free_bits_KL=0.0, top_k=None):
        super().__init__()
        self.device = device
        self.split_block_nr = split_block_nr

        if pretrained_path and os.path.isfile(pretrained_path):
            vit = timm.create_model(model_name, pretrained=False)
            state_dict = torch.load(pretrained_path, map_location="cuda")
            vit.load_state_dict(state_dict)
        else:
            vit = timm.create_model(model_name, pretrained=True)
            torch.save(vit.state_dict(), f"{model_name}.pth")

        # extract ViT components. The "pre" blocks run on the sender, the "post" blocks run on the receiver.
        self.patch_embed = vit.patch_embed
        self.cls_token   = vit.cls_token
        self.pos_embed   = vit.pos_embed
        self.pos_drop    = vit.pos_drop
        self.blocks_pre  = nn.ModuleList(vit.blocks[:split_block_nr])
        self.blocks_post = nn.ModuleList(vit.blocks[split_block_nr:])
        self.norm_post   = vit.norm   
        self.head        = vit.head
        self.embed_dim = vit.cls_token.shape[-1]#embed_dim
        self.bottleneck_dim = _get_hidden_dims(self.embed_dim)[1]

        assert top_k is not None, "top_k must be specified for ConditionedViT"
        self.gate = conditional_merging_gate.ConditionalMergingGate(embed_dim=self.embed_dim, bottleneck_dim=self.bottleneck_dim, K=top_k)
        self.cls_ib = cls_ib.CLSIB(embed_dim=self.embed_dim, bottleneck_dim=self.bottleneck_dim, use_kl_div=use_kl_div, device=device, free_bits_per_dim=free_bits_KL)

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze every original ViT parameter. The IB and gate stay trainable."""
        for p in self.patch_embed.parameters():
            p.requires_grad = False
        for p in self.blocks_pre.parameters():
            p.requires_grad = False
        for p in self.blocks_post.parameters():
            p.requires_grad = False
        for p in self.norm_post.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = False
        self.cls_token.requires_grad = False
        self.pos_embed.requires_grad = False

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device="cuda", **kwargs):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
            config = checkpoint.get("config", {})
        else:
            state_dict = checkpoint
            config = {}

        top_k = kwargs.pop("top_k", None)
        if top_k is None:
            top_k = config.get("gate_top_k")
        elif isinstance(top_k, (list, tuple)):
            if "gate_top_k" in config:
                top_k = config["gate_top_k"]
            elif len(top_k) == 1:
                top_k = top_k[0]
            elif top_k == []:
                top_k = config.get("gate_top_k")
            else:
                raise ValueError(
                    "ConditionedViT.from_checkpoint expects a single integer top_k "
                    "for the gate architecture. Pass the trained gate_top_k, not a sweep list."
                )

        if top_k is None:
            raise ValueError(
                "Could not infer gate top_k from the checkpoint. Pass top_k explicitly "
                "or save gate_top_k in the checkpoint config."
            )

        free_bits_KL = kwargs.pop("free_bits_KL", config.get("free_bits_KL", 0.0))
        model = cls(
            device=device,
            top_k=top_k,
            free_bits_KL=free_bits_KL,
            pretrained_backbone=False,
            **kwargs,
        )
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        return model

    def _embed(self, x):
        """Patch + position embedding + CLS prepend. Runs on the sender."""
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        return x

    def forward(self, x, top_k=None):
        # sender
        x = self._embed(x)
        for block in self.blocks_pre:
            x = block(x)

        cls_token = x[:, 0, :]
        patches   = x[:, 1:, :]

        z, kl, mu, logvar, kl_actual = self.cls_ib.encode(cls_token)

        if top_k is not None and top_k < patches.shape[1]:
            assignment, gate_values = self.gate(z=z, patches=patches)
            merged_patches, size = self.gate.merge(patches, assignment)
            transmitted_patches = merged_patches

        else:
            transmitted_patches = patches
            size = torch.tensor(patches.shape[1], device=patches.device).expand(patches.shape[0], 1)  # [B, 1] - For diagnostics, the "size" of the merged patch is just the original number of patches when not merging.
        
        # Channel: z and transmitted_patches cross the wire here. The rest is the receiver side.
        # receiver
        reconstructed_cls = self.cls_ib.decode(z)
        seq = torch.cat((reconstructed_cls.unsqueeze(1), transmitted_patches), dim=1)
        for block in self.blocks_post:
            seq = block(seq)
        seq = self.norm_post(seq)
        logits = self.head(seq[:, 0, :])

        return {
            "logits": logits,
            "kl": kl,
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "kl_actual": kl_actual,
            "reconstructed_cls": reconstructed_cls,
            "original_cls": cls_token,
            "size": size,
        }