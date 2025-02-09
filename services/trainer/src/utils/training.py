import torch
import torch.optim as optim
from torchmetrics.regression import (
    R2Score,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
)
import torch.optim.lr_scheduler as lr_scheduler
import os
import math


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: lr_scheduler._LRScheduler,
    checkpoint_path: str,
    num_epochs: int,
    device: torch.device,
    patience=20,
) -> None:
    """
    Trains the model on the training dataset for the specified number of epochs.
    Includes functionality to load existing checkpoints and save new ones.
    """
    # Initialize metrics
    r2_score = R2Score().to(device)
    mae = MeanAbsoluteError().to(device)
    mse = MeanSquaredError().to(device)
    mape = MeanAbsolutePercentageError().to(device)
    best_train_loss = float("inf")
    no_improve_count = 0  # Counter for early stopping

    # Check if a checkpoint exists
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from '{checkpoint_path}'...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_train_loss = checkpoint["best_train_loss"]
        mae.load_state_dict(checkpoint["mae_state_dict"])
        mape.load_state_dict(checkpoint["mape_state_dict"])
        mse.load_state_dict(checkpoint["mse_state_dict"])
        r2_score.load_state_dict(checkpoint["r2_score_state_dict"])
        no_improve_count = checkpoint.get("no_improve_count", 0)
        print("Checkpoint loaded successfully.")
    else:
        print("No checkpoint found. Starting training from scratch.")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        r2_score.reset()
        mae.reset()

        for batch_X_num, batch_X_cat, dist_ids, prop_ids, batch_y in train_loader:

            batch_X_num, batch_X_cat, dist_ids, prop_ids, batch_y = (
                batch_X_num.to(device),
                batch_X_cat.to(device),
                dist_ids.to(device),
                prop_ids.to(device),
                batch_y.to(device).unsqueeze(1),
            )

            optimizer.zero_grad()
            outputs = model(batch_X_num, batch_X_cat, dist_ids, prop_ids)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            r2_score.update(outputs, batch_y)
            mae.update(outputs, batch_y)
            mse.update(outputs, batch_y)
            mape.update(outputs, batch_y)

        avg_loss = epoch_loss / len(train_loader)
        avg_r2 = r2_score.compute().item()
        avg_mae = mae.compute().item()
        avg_mape = mape.compute().item()
        avg_smape = mape.compute().item() ** 2
        avg_rmse = math.sqrt(mse.compute().item())
        scheduler.step(avg_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Loss: {avg_loss:.4f}, "
            f"MAE: {avg_mae:.4f}, "
            f"RMSE: {avg_rmse:.4f}, "
            f"MAPE: {avg_mape:.4f} ,"
            f"sMAPE: {avg_smape:.4f} ,"
            f"R2: {avg_r2:.4f}"
        )
        # Check for improvement
        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            no_improve_count = 0
            # Save model checkpoint
            checkpoint = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_train_loss": best_train_loss,
                "mae_state_dict": mae.state_dict(),
                "mse_state_dict": mse.state_dict(),
                "mape_state_dict": mape.state_dict(),
                "r2_score_state_dict": r2_score.state_dict(),
                "no_improve_count": no_improve_count,
            }
            torch.save(checkpoint, checkpoint_path)
            print(
                f"New best model saved at epoch {epoch+1} with Loss: {best_train_loss:.4f}"
            )
        else:
            no_improve_count += 1
            print(f"No improvement for {no_improve_count} epoch(s).")
            if no_improve_count >= patience:
                print("Early stopping triggered.")
                break
