import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os


def evaluate_model(model, test_loader, device, model_path=None):
    """
    通用评估函数：
    - 兼容 PyG GNN (batch.y, model(batch))
    - 兼容 BERT / Transformer (dict batch, model(**batch))
    """

    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {model_path}")

    model = model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []
    all_smiles = []

    with torch.no_grad():
        for batch in test_loader:

            # ---------- 情况 A：tuple / list ----------
            if isinstance(batch, (list, tuple)):
                batch_inputs, batch_labels = batch
            else:
                batch_inputs = batch
                batch_labels = None

            # 如果是 dict 且包含 labels，则补上标签（兼容 BERT DataLoader 默认行为）
            if batch_labels is None and isinstance(batch_inputs, dict):
                batch_labels = batch_inputs.get("labels")

            # ========== BERT ==========
            if isinstance(batch_inputs, dict):
                batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                outputs = model(**batch_inputs)

                if hasattr(outputs, "logits"):
                    preds = outputs.logits
                else:
                    preds = outputs

                all_predictions.extend(preds.cpu().numpy().flatten())

                if batch_labels is not None:
                    batch_labels = batch_labels.to(device)
                    all_targets.extend(batch_labels.cpu().numpy().flatten())
                
                if hasattr(batch_inputs, "smiles"):
                    all_smiles.extend(batch_inputs.smiles)

            # ========== GNN ==========
            else:
                batch_inputs = batch_inputs.to(device)
                preds = model(batch_inputs)
                targets = batch_inputs.y.view(-1, 1)

                all_predictions.extend(preds.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())

                if hasattr(batch_inputs, "smiles"):
                    all_smiles.extend(batch_inputs.smiles)

    # =========================
    # 数值指标（scaled）
    # =========================
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)

    # =========================
    # 反归一化（如果存在）
    # =========================
    mse_orig = rmse_orig = mae_orig = r2_orig = None
    preds_orig = targets_orig = None

    try:
        dataset = getattr(test_loader, "dataset", None)
        scaler = getattr(dataset, "rt_scaler", None)

        if scaler is not None:
            preds_arr = np.array(all_predictions).reshape(-1, 1)
            targets_arr = np.array(all_targets).reshape(-1, 1)

            preds_orig = scaler.inverse_transform(preds_arr).reshape(-1)
            targets_orig = scaler.inverse_transform(targets_arr).reshape(-1)

            mse_orig = mean_squared_error(targets_orig, preds_orig)
            rmse_orig = np.sqrt(mse_orig)
            mae_orig = mean_absolute_error(targets_orig, preds_orig)
            r2_orig = r2_score(targets_orig, preds_orig)

    except Exception as e:
        print(f"Warning: inverse transform failed: {e}")

    # =========================
    # 打印结果
    # =========================
    print("\n=== Test Results (scaled) ===")
    print(f"MSE : {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE : {mae:.6f}")
    print(f"R²  : {r2:.6f}")

    if preds_orig is not None:
        print("\n=== Test Results (original scale) ===")
        print(f"MSE : {mse_orig:.6f}")
        print(f"RMSE: {rmse_orig:.6f}")
        print(f"MAE : {mae_orig:.6f}")
        print(f"R²  : {r2_orig:.6f}")

    print("\n=== End Evaluation ===")

    return {
        "predictions": all_predictions,
        "targets": all_targets,
        "smiles": all_smiles,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "predictions_orig": preds_orig.tolist() if preds_orig is not None else None,
        "targets_orig": targets_orig.tolist() if targets_orig is not None else None,
        "mse_orig": mse_orig,
        "rmse_orig": rmse_orig,
        "mae_orig": mae_orig,
        "r2_orig": r2_orig,
    }

