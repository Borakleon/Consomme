import os
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import conditioned_vit
import imagenet_dataloader

import matplotlib.pyplot as plt

class ConditionedViTTrainer:
    def __init__(self, vit, train_loader, val_loader, device="cuda", lr=1e-4, weight_decay=0.0, kl_weight=1e-3, recon_weight=1.0, kl_anneal_epochs=5, recon_anneal_epochs=5, gate_top_k=64, grad_clip=1.0, log_dir="./runs", run_name="conditioned_vit",
                 lr_scheduler=False, total_epochs=100, min_lr_ratio=0.01, warmup_epochs=5):
        self.vit = vit.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.kl_weight = kl_weight
        self.recon_weight = recon_weight
        self.kl_anneal_epochs = kl_anneal_epochs
        self.recon_anneal_epochs = recon_anneal_epochs
        self.gate_top_k = gate_top_k
        self.grad_clip = grad_clip

        # Only optimize parameters that require grad (IB, gate, skip projection, and any unfrozen ViT params
        trainable = [p for p in self.vit.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

        self.criterion_cls = nn.CrossEntropyLoss()

        lr_scheduler = lr_scheduler and (total_epochs > 1)  # Only use scheduler if requested and if training for more than 1 epoch
        self.total_epochs = total_epochs
        if lr_scheduler:
            self.warmup_epochs = warmup_epochs
            self.min_lr_ratio = min_lr_ratio
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_lambda)
        else:
            self.lr_scheduler = None
            self.warmup_epochs = 0
            self.min_lr_ratio = 0.01

        self.log_dir = log_dir
        self.run_name = run_name
        os.makedirs(log_dir, exist_ok=True)

        self.history = {
            "train_loss": [], 
            "train_cls": [], 
            "train_kl": [], 
            "train_recon": [],
            "val_acc": [], 
            "val_acc_topk_off": [],
            "kl_weight_eff": [], 
            "recon_weight_eff": [],
            "mu_norm": [], 
            "sigma_mean": [],
        }
        if self.lr_scheduler is not None:
            self.history["lr"] = []

    def _kl_weight(self, epoch):
        return self.kl_weight * min(1.0, epoch / max(1, self.kl_anneal_epochs))

    def _recon_weight(self, epoch):
        return self.recon_weight * min(1.0, epoch / max(1, self.recon_anneal_epochs))

    def lr_lambda(self, epoch):
        if epoch < self.warmup_epochs: 
            return (epoch + 1) / self.warmup_epochs # Linear warmup from 0 to 1 over the first few epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cos_progress = 0.5 * (1 + math.cos(math.pi * progress))  # Cosine annealing from 1 to 0
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cos_progress

    def train_epoch(self, epoch):
        self.vit.train()

        kl_w = self._kl_weight(epoch)
        recon_w = self._recon_weight(epoch)

        running = {"loss": 0.0, 
                   "cls": 0.0,
                    "kl": 0.0, 
                    "recon": 0.0,
                   "mu_norm": 0.0, 
                   "sigma_mean": 0.0
                   }
        
        n_batches = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [train]", leave=False)

        for inputs, labels in pbar:
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            out = self.vit(inputs, top_k=self.gate_top_k)

            cls_loss = self.criterion_cls(out["logits"], labels)
            kl_loss = out["kl"]
            recon_loss = F.mse_loss(out["reconstructed_cls"],
                                    out["original_cls"].detach())

            loss = cls_loss + kl_w * kl_loss + recon_w * recon_loss

            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.vit.parameters() if p.requires_grad],
                    self.grad_clip,
                )
            self.optimizer.step()

            with torch.no_grad():
                running["loss"]  += loss.item()
                running["cls"]   += cls_loss.item()
                running["kl"]    += kl_loss.item() if torch.is_tensor(kl_loss) else float(kl_loss)
                running["kl_actual"] = out.get("kl_actual", 0.0)  # For diagnostics, not used in loss.
                running["recon"] += recon_loss.item()

                running["mu_norm"]   += out["mu"].pow(2).sum(dim=1).mean().sqrt().item()
                running["sigma_mean"] += torch.exp(0.5 * out["logvar"]).mean().item()
            n_batches += 1

            pbar.set_postfix({
                "loss":  f"{running['loss']/n_batches:.3f}",
                "cls":   f"{running['cls']/n_batches:.3f}",
                "kl":    f"{running.get('kl', 0.0)/n_batches:.3f}",
                "kl_act": f"{running.get('kl_actual', 0.0)/n_batches:.3f}",
                "recon": f"{running.get('recon', 0.0)/n_batches:.3f}",
            })

        avg = {k: v / max(1, n_batches) for k, v in running.items()}
        self.history["train_loss"].append(avg["loss"])
        self.history["train_cls"].append(avg["cls"])
        self.history["train_kl"].append(avg["kl"])
        self.history["train_recon"].append(avg["recon"])
        self.history["kl_weight_eff"].append(kl_w)
        self.history["recon_weight_eff"].append(recon_w)
        self.history["mu_norm"].append(avg["mu_norm"])
        self.history["sigma_mean"].append(avg["sigma_mean"])
        return avg

    @torch.no_grad()
    def evaluate(self, top_k=None, label="val"):
        self.vit.eval()
        correct, total = 0, 0
        pbar = tqdm(self.val_loader, desc=f"[{label} top_k={top_k}]", leave=False)
        for inputs, labels in pbar:
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            out = self.vit(inputs, top_k=top_k)
            preds = out["logits"].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({"acc": f"{100 * correct / total:.2f}%"})
        return correct / max(1, total)

    @torch.no_grad()
    def evaluate_rate_distortion(self, top_k_values, label="rd"):
        """Sweep top_k for a rate-distortion curve at a fixed KL weight."""
        results = {}
        for k in top_k_values:
            acc = self.evaluate(top_k=k, label=f"{label}_k{k}")
            results[k] = acc
        return results

    def save_checkpoint(self, epoch, path=None):
        path = path or os.path.join(self.log_dir, f"{self.run_name}_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": self.vit.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "history": self.history,
            "scheduler_state": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            "config": {
                "free_bits_KL": self.vit.cls_ib.free_bits_per_dim,
                "kl_weight": self.kl_weight,
                "recon_weight": self.recon_weight,
                "gate_top_k": self.gate_top_k,
            },
        }, path)
        return path

    def load_checkpoint(self, path, resume_training=False):
        ckpt = torch.load(path, map_location=self.device)
        self.vit.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.history = ckpt.get("history", self.history)
        last_trained_epoch = ckpt.get("epoch", 0)
        if self.lr_scheduler is not None and "scheduler_state" in ckpt:
            self.lr_scheduler.load_state_dict(ckpt["scheduler_state"])

        if resume_training:
            # Remove all history entries after the loaded epoch, since they may be from a different training run if we're loading a pretrained model rather than resuming training.
            # Data is overwritten when results are saved at the end of each epoch.
            for key in self.history:
                self.history[key] = self.history[key][:last_trained_epoch + 1]
            return last_trained_epoch + 1 # Start from the next epoch after the loaded one.
        return 0

    def save_history(self):
        with open(os.path.join(self.log_dir, f"{self.run_name}_history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

    def fit(self, num_epochs, eval_every=1, save_every=5, start_epoch=0, rd_sweep_during_training=False):
        
        if start_epoch > 0:
            best_acc = max(self.history["val_acc"][:start_epoch]) if self.history["val_acc"] else 0.0
        else:
            best_acc = 0.0
        for epoch in range(start_epoch, num_epochs):
            train_metrics = self.train_epoch(epoch)
            if (epoch + 1) % eval_every == 0:
                if rd_sweep_during_training:
                    val_acc = self.evaluate_rate_distortion(top_k_values=[8, 16, 32, 64, 96, 128, 196], label="val_sweep")
                else:
                    val_acc = self.evaluate(top_k=self.gate_top_k, label="val")
                self.history["val_acc"].append(val_acc)
                tqdm.write(
                    f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f} "
                    f"cls={train_metrics['cls']:.4f} kl={train_metrics['kl']:.4f} "
                    f"recon={train_metrics['recon']:.4f} | val_acc={val_acc} "
                    f"| ||mu||={train_metrics['mu_norm']:.3f} "
                    f"sigma={train_metrics['sigma_mean']:.3f}"
                )
                if rd_sweep_during_training:
                    if val_acc[self.gate_top_k] > best_acc:
                        self.save_checkpoint(epoch, path=os.path.join(self.log_dir, f"{self.run_name}_best.pt"))
                        best_acc = val_acc[self.gate_top_k]
                else:
                    if val_acc > best_acc:
                        best_acc = val_acc
                        self.save_checkpoint(epoch, path=os.path.join(self.log_dir, f"{self.run_name}_best.pt"))

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                    self.history["lr"].append(self.lr_scheduler.get_last_lr()[0]) # Log actual learning rate here, to reduce lr_scheduler existing checks

            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch)
            self.save_history()

        return best_acc

    def get_sigma(self, epoch, total_epochs):
        """Annealing sigma for pertubed top-k selection to approach hard top-k selection towards the end of training."""
        return max(0.01, 0.1 - 0.09 * (epoch / max(1, total_epochs - 1))) # Linearly anneal from 0.1 to 0.01 over the course of training, with a floor at 0.01 to maintain some noise for stability.
    
    def generate_train_plots(self, path):

        plt.figure(figsize=(20, 12))
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.savefig(path + "_total_loss.png")
        plt.close()

        plt.figure(figsize=(20, 12))
        plt.plot(self.history["train_cls"], label="Train Classification Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Classification Loss")
        plt.legend()
        plt.savefig(path + "_cls_loss.png")
        plt.close()

        plt.figure(figsize=(20, 12))
        plt.plot(self.history["train_kl"], label="Train KL Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training KL Loss")
        plt.legend()
        plt.savefig(path + "_kl_loss.png")
        plt.close()

        plt.figure(figsize=(20, 12))
        plt.plot(self.history["train_recon"], label="Train Reconstruction Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Reconstruction Loss")
        plt.legend()
        plt.savefig(path + "_recon_loss.png")
        plt.close()

        plt.figure(figsize=(20, 12))
        plt.plot(self.history["mu_norm"], label="Latent Mean Norm")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Latent Mean Norm")
        plt.legend()
        plt.savefig(path + "_mu_norm.png")
        plt.close()

        plt.figure(figsize=(20, 12))
        plt.plot(self.history["sigma_mean"], label="Latent Sigma Mean")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Latent Sigma Mean")
        plt.legend()
        plt.savefig(path + "_sigma_mean.png")
        plt.close()

        if self.lr_scheduler is not None:
            plt.figure(figsize=(20, 12))
            plt.plot(self.history["lr"], label="Learning Rate")
            plt.xlabel("Epoch")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Schedule")
            plt.legend()
            plt.savefig(path + "_lr_schedule.png")
            plt.close()
        

if __name__ == "__main__":
    import argparse

    argparse = argparse.ArgumentParser(description="Train a Conditioned ViT model on ImageNet.")
    argparse.add_argument("--resume", action="store_true", help="Resume training from a checkpoint.")
    argparse.add_argument("--model_path", type=str, default=None, help="Path to the model checkpoint to resume training from.")
    
    argparse.add_argument("--data_dir", type=str, required=True, default=None, help="Path to the ImageNet dataset.")
    argparse.add_argument("--model_name", type=str, default="deit_small_patch16_224", help="Name of the ViT model to use.")
    argparse.add_argument("--pretrained_path", type=str, default=None, help="Path to the pretrained ViT model.")

    argparse.add_argument("--log_dir", type=str, default="./runs", help="Directory to save logs and checkpoints.")
    argparse.add_argument("--split_block_nr", type=int, required=True, help="Number of ViT blocks to run on the sender side.")
    argparse.add_argument("--gate_top_k", type=int, required=True, help="Number of top-k patches to select in the merging gate.")
    argparse.add_argument("--device", type=str, default="cuda", help="Device to use for training (e.g., 'cuda' or 'cpu').")

    argparse.add_argument("--batch_size_train", type=int, default=512, help="Training batch size.")
    argparse.add_argument("--batch_size_val", type=int, default=512, help="Validation batch size.")
    argparse.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers.")
    argparse.add_argument("--num_epochs", type=int, default=40, help="Number of training epochs.")
    argparse.add_argument("--min_lr_ratio", type=float, default=0.01, help="Minimum learning rate ratio.")
    argparse.add_argument("--warmup_epochs", type=int, default=5, help="Number of epochs to warm up the learning rate.")

    argparse.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    argparse.add_argument("--lr_scheduler", action="store_false", help="Use a learning rate scheduler.")
    argparse.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay for optimizer.")
    argparse.add_argument("--kl_weight", type=float, default=0.5, help="Weight for the KL divergence loss.")
    argparse.add_argument("--use_kl_div", action="store_false", help="Whether to use KL divergence in the loss.")
    argparse.add_argument("--recon_weight", type=float, default=0.1, help="Weight for the reconstruction loss.")
    argparse.add_argument("--kl_anneal_epochs", type=int, default=30, help="Number of epochs to anneal the KL weight.")
    argparse.add_argument("--recon_anneal_epochs", type=int, default=5, help="Number of epochs to anneal the reconstruction weight.")
    argparse.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value.")
    argparse.add_argument("--free_bits_KL", type=float, default=0.5, help="Free bits for KL divergence to prevent posterior collapse.")
    argparse.add_argument("--rd_sweep_during_training", action="store_true", help="Perform rate-distortion sweep during training.")
    argparse.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")


    args = argparse.parse_args()

    torch.backends.cudnn.benchmark = False    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True



    train_loader = imagenet_dataloader.get_dataloader(
        batch_size=args.batch_size_train, num_workers=args.num_workers,
        data_dir=args.data_dir, split="train", pin_memory=True, model_name=args.model_name)
    val_loader = imagenet_dataloader.get_dataloader(
        batch_size=args.batch_size_val, num_workers=args.num_workers,
        data_dir=args.data_dir, split="val", pin_memory=True, model_name=args.model_name)

    print(f"Training with split_block_nr={args.split_block_nr} and gate_top_k={args.gate_top_k} on device {args.device}.")

    vit = conditioned_vit.ConditionedViT(
        split_block_nr=args.split_block_nr,
        use_kl_div=args.use_kl_div,
        device=args.device,
        pretrained_path=args.pretrained_path,
        model_name=args.model_name,
        free_bits_KL=args.free_bits_KL,
        top_k=args.gate_top_k,
    )

    run_name = (
        f"consomme_{args.model_name}_{args.split_block_nr}_top_k{args.gate_top_k}"
    )

    new_log_path = os.path.join(args.log_dir, run_name)
    if not args.resume:
        os.makedirs(new_log_path, exist_ok=False)
    else:
        model_path = args.model_path
        assert os.path.exists(model_path), f"Model path {model_path} does not exist for resuming training."

    trainer = ConditionedViTTrainer(
        vit=vit, train_loader=train_loader, val_loader=val_loader, device=args.device,
        lr=args.lr, weight_decay=args.weight_decay,
        kl_weight=args.kl_weight, recon_weight=args.recon_weight,
        kl_anneal_epochs=args.kl_anneal_epochs,
        recon_anneal_epochs=args.recon_anneal_epochs,
        gate_top_k=args.gate_top_k, grad_clip=args.grad_clip,
        log_dir=new_log_path, run_name=run_name,
        lr_scheduler=args.lr_scheduler,
        min_lr_ratio=args.min_lr_ratio, warmup_epochs=args.warmup_epochs,
        total_epochs=args.num_epochs
    )

    epoch = trainer.load_checkpoint(model_path, resume_training=args.resume) if args.resume else 0
    with open(os.path.join(new_log_path, f"{run_name}_config.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    best_acc = trainer.fit(num_epochs=args.num_epochs, start_epoch=epoch, rd_sweep_during_training=args.rd_sweep_during_training)
    print(f"Best val accuracy: {best_acc:.4f}")

    trainer.generate_train_plots(os.path.join(new_log_path, run_name))