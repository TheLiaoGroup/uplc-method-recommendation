import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import os



class WeightedRegressionLoss(torch.nn.Module):
    def __init__(self, bin_edges, bin_weights, mode="huber", delta=0.05):
        """
        bin_edges: shape [num_bins+1]，包含首尾（min/max）
        bin_weights: shape [num_bins]
        mode: "mae" | "mse" | "huber"
        delta: huber 的阈值
               如果你的 y 是 min-max 到 [0,1]，delta 可先试 0.03~0.08
               如果 y 是 z-score，delta 可先试 0.5~1.5
        """
        super().__init__()
        self.register_buffer("bin_edges", bin_edges)
        self.register_buffer("bin_weights", bin_weights)
        self.mode = mode
        self.delta = float(delta)

    def _sample_weights(self, target):
        # target: [B]
        # bucketize 需要内部边界（不含两端）
        inner = self.bin_edges[1:-1]
        bin_ids = torch.bucketize(target, inner, right=False)  # 0..num_bins-1
        return self.bin_weights[bin_ids]

    def forward(self, pred, target):
        import torch

        pred = pred.view(-1)
        target = target.view(-1)
        w = self._sample_weights(target)

        diff = pred - target

        if self.mode == "mae":
            loss = torch.abs(diff)
        elif self.mode == "mse":
            loss = diff * diff
        elif self.mode == "huber":
            abs_diff = torch.abs(diff)
            d = self.delta
            quad = torch.minimum(abs_diff, torch.tensor(d, device=abs_diff.device))
            lin = abs_diff - quad
            loss = 0.5 * quad * quad + d * lin
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return (w * loss).mean()



def forward_and_loss(model, batch, device, criterion=None):
    """
    统一 GNN / BERT 的 forward + loss 计算
    """

    # -------- BERT / ChemBERTa：DataLoader 默认 collate 为 dict --------
    if isinstance(batch, dict):
        labels = batch.get("labels")
        if labels is None:
            raise ValueError("BERT batch missing labels")

        # 不修改原 batch，复制一个送入设备
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = labels.to(device).view(-1).float()

        outputs = model(**inputs)
        preds = outputs.logits if hasattr(outputs, "logits") else outputs
        preds = preds.squeeze(-1)

        if criterion is None:
            raise ValueError("BERT training requires criterion when model output has no loss")

        loss = criterion(preds, labels)
        return loss, preds, labels

    # -------- BERT / ChemBERTa：手动 tuple 形式 (inputs_dict, labels) --------
    if isinstance(batch, (tuple, list)):
        inputs, labels = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device).view(-1).float()

        outputs = model(**inputs)
        preds = outputs.logits if hasattr(outputs, "logits") else outputs
        preds = preds.squeeze(-1)

        if criterion is None:
            raise ValueError("BERT training requires criterion when model output has no loss")

        loss = criterion(preds, labels)
        return loss, preds, labels

    # -------- GNN (PyG) 情况 --------
    # batch = torch_geometric.data.Data
    batch = batch.to(device)
    preds = model(batch)
    labels = batch.y.view(-1, 1)

    if criterion is None:
        raise ValueError("GNN training requires criterion")

    loss = criterion(preds, labels)
    return loss, preds.squeeze(-1), labels.squeeze(-1)

def train_model(
    model, train_loader, val_loader,
    num_epochs=100, learning_rate=0.001,
    criterion='MSELoss', beta=0.5,
    scheduler=None, weight_decay=1e-4,
    save_path="best_model.pth", device=None,
):
    if criterion == "MSELoss":
        criterion = nn.MSELoss()
    elif criterion == "SmoothL1Loss":
        criterion = nn.SmoothL1Loss(beta=beta)
    elif criterion == "WeightedHuber":
        # 从 loader 里拿 dataset
        dataset = train_loader.dataset

        bin_edges = dataset.bin_edges.to(device)
        bin_weights = dataset.bin_weights.to(device)

        criterion = WeightedRegressionLoss(
            bin_edges=bin_edges,
            bin_weights=bin_weights,
            mode="huber",
            delta=1)


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay)

    # ---------- scheduler ----------
    if scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)
    elif scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    elif scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    elif scheduler == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif scheduler == "OneCycleLR":
        total_steps = len(train_loader) * num_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps)
    else:
        scheduler = None

    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    print(f"Starting training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(num_epochs):

        # ===== train =====
        model.train()
        total_train_loss = 0
        train_preds, train_trues = [], []

        for batch in train_loader:
            optimizer.zero_grad()

            loss, preds, targets = forward_and_loss(
                model, batch, device, criterion
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

            total_train_loss += loss.item()
            train_preds.append(preds.detach().cpu().numpy())
            train_trues.append(targets.detach().cpu().numpy())

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        train_preds = np.concatenate(train_preds)
        train_trues = np.concatenate(train_trues)
        train_r2 = r2_score(train_trues, train_preds)

        # ===== validation =====
        model.eval()
        total_val_loss = 0
        val_preds, val_trues = [], []

        with torch.no_grad():
            for batch in val_loader:
                loss, preds, targets = forward_and_loss(
                    model, batch, device, criterion
                )

                total_val_loss += loss.item()
                val_preds.append(preds.cpu().numpy())
                val_trues.append(targets.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)
        val_r2 = r2_score(val_trues, val_preds)

        # scheduler step
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif scheduler is not None and not isinstance(
            scheduler, torch.optim.lr_scheduler.OneCycleLR
        ):
            scheduler.step()

        # save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "train_r2": train_r2,
                    "val_r2": val_r2,
                },
                save_path
            )

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {avg_train_loss:.6f}, R2: {train_r2:.4f}")
            print(f"  Val   Loss: {avg_val_loss:.6f}, R2: {val_r2:.4f}")
            print(f"  Best  Val  : {best_val_loss:.6f}")

    return train_losses, val_losses

