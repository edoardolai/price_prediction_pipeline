import torch
from torchmetrics.regression import (
    R2Score,
    MeanAbsoluteError,
    MeanSquaredError,
    MeanAbsolutePercentageError,
)
from typing import Tuple
import math
import os


def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    checkpoint_path: str,
) -> Tuple[float, float, float]:
    """
    Evaluates the model on the test dataset and returns the loss, MAE, and R² score.
    """
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    print("Best model loaded.")
    model.eval()
    test_loss = 0
    r2_score = R2Score().to(device)
    mae = MeanAbsoluteError().to(device)
    mse = MeanSquaredError().to(device)
    mape = MeanAbsolutePercentageError().to(device)

    with torch.no_grad():
        for batch_X_num, batch_X_cat, dist_ids, prop_ids, batch_y in test_loader:
            batch_X_num, batch_X_cat, dist_ids, prop_ids, batch_y = (
                batch_X_num.to(device),
                batch_X_cat.to(device),
                dist_ids.to(device),
                prop_ids.to(device),
                batch_y.to(device).unsqueeze(1),
            )

            outputs = model(batch_X_num, batch_X_cat, dist_ids, prop_ids)
            loss = criterion(outputs, batch_y)

            test_loss += loss.item()
            r2_score.update(outputs, batch_y)
            mae.update(outputs, batch_y)
            mse.update(outputs, batch_y)
            mape.update(outputs, batch_y)

    avg_loss = test_loss / len(test_loader)
    avg_r2 = r2_score.compute().item()
    avg_mae = mae.compute().item()
    avg_mape = mape.compute().item()
    avg_smape = mape.compute().item() ** 2
    avg_rmse = math.sqrt(mse.compute().item())
    print(
        f"Final Test -> Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}, R²: {avg_r2:.4f}, RMSE: {avg_rmse:.4f}, MAPE: {avg_mape:.4f}, sMAPE: {avg_rmse:.4f}"
    )
    return avg_loss, avg_mae, avg_r2, avg_mape, avg_smape, avg_rmse
