import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
import joblib
from torchmetrics import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    R2Score,
)


current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from utils.model import NeuralNetwork
from utils.dataset import CustomDataset
from utils.training import train_model
from utils.evaluation import evaluate_model


class PropertyModelTrainer:
    def __init__(
        self,
        input_dir: str,
        artifacts_dir: str,
        config: dict,
        force_retrain: bool = False,
    ):
        self.input_dir = Path(input_dir).resolve()
        self.artifacts_dir = Path(artifacts_dir).resolve()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metadata_file = self.artifacts_dir / "metadata.json"
        self.force_retrain = force_retrain

        print(f"Input directory (abs): {self.input_dir}")
        print(f"Artifacts directory (abs): {self.artifacts_dir}")
        print(
            f"Checkpoint path (abs): {(self.artifacts_dir / 'checkpoints' / 'best_model_checkpoint.pth').resolve()}"
        )

        # Create directories
        (self.artifacts_dir / "encoders").mkdir(parents=True, exist_ok=True)
        (self.artifacts_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def _should_train(self, current_data_size: int) -> bool:
        """Determine if training is needed"""
        # First run check
        checkpoint_path = (
            self.artifacts_dir / "checkpoints" / "best_model_checkpoint.pth"
        )

        if checkpoint_path.exists():
            print("Found existing checkpoint")
        else:
            print("No checkpoint found - will train from scratch")
            return True

        # Force retrain check
        if self.force_retrain:
            return True

        # Data change check
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                metadata = json.load(f)
                # Train if data size changed by more than 5%
                if (
                    abs(metadata["data_size"] - current_data_size)
                    / metadata["data_size"]
                    > 0.05
                ):
                    return True
        return False

    def _setup_training(self, df: pd.DataFrame):
        # Split data
        X, y = self._split_data(df, target="price")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Impute missing values
        imputer = KNNImputer(n_neighbors=5)
        impute_cols = [
            "state_of_building",
            "surface_of_the_plot",
            "nb_bedrooms",
            "living_area",
        ]
        X_train[impute_cols] = imputer.fit_transform(X_train[impute_cols])
        X_test[impute_cols] = imputer.transform(X_test[impute_cols])

        # Save imputer
        joblib.dump(imputer, self.artifacts_dir / "encoders" / "knn_imputer.joblib")

        # Scale numerical features
        num_cols = ["surface_of_the_plot", "living_area", "nb_bedrooms"]
        scaler = RobustScaler()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

        # Save scaler
        joblib.dump(scaler, self.artifacts_dir / "encoders" / "numerical_scaler.joblib")

        # Create datasets
        cat_cols = [
            "district_id",
            "property_sub_type_id",
            "state_of_building",
            "equipped_kitchen",
            "garden",
            "swimming_pool",
            "terrace",
            "furnished",
        ]

        # Save model configuration
        model_config = {
            "numeric_cols": num_cols,
            "cat_cols": cat_cols,
            "numeric_input_dim": len(num_cols),
            "cat_input_dim": len(cat_cols),
            "num_districts": len(df["district_id"].unique()),
            "district_emb_dim": self.config["district_emb_dim"],
            "num_properties": len(df["property_sub_type_id"].unique()),
            "property_emb_dim": self.config["property_emb_dim"],
            "dropout_rate": self.config["dropout_rate"],
        }

        joblib.dump(
            model_config, self.artifacts_dir / "encoders" / "model_config.joblib"
        )

        train_dataset = CustomDataset(X_train, y_train, num_cols, cat_cols)
        test_dataset = CustomDataset(X_test, y_test, num_cols, cat_cols)

        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config["batch_size"], shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.config["batch_size"], shuffle=False
        )

        # Initialize model
        model = NeuralNetwork(
            numeric_input_dim=model_config["numeric_input_dim"],
            cat_input_dim=model_config["cat_input_dim"],
            num_districts=model_config["num_districts"],
            district_emb_dim=model_config["district_emb_dim"],
            num_properties=model_config["num_properties"],
            property_emb_dim=model_config["property_emb_dim"],
            dropout_rate=model_config["dropout_rate"],
        ).to(self.device)

        return {
            "model": model,
            "train_loader": train_loader,
            "test_loader": test_loader,
            "train_size": len(train_dataset),
            "test_size": len(test_dataset),
            "model_config": model_config,
        }

    def _train_and_evaluate(self, training_setup: dict):
        model = training_setup["model"]
        train_loader = training_setup["train_loader"]
        test_loader = training_setup["test_loader"]
        model_config = training_setup["model_config"]

        # Setup training
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        criterion = torch.nn.SmoothL1Loss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=10, factor=0.5
        )

        checkpoint_path = (
            self.artifacts_dir / "checkpoints" / "best_model_checkpoint.pth"
        )

        # Initialize metrics for initial checkpoint
        r2_score = R2Score().to(self.device)
        mae = MeanAbsoluteError().to(self.device)
        mse = MeanSquaredError().to(self.device)
        mape = MeanAbsolutePercentageError().to(self.device)

        torch.save(
            {
                "model_state": model.state_dict(),
                "model_config": model_config,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_train_loss": float("inf"),
                "mae_state_dict": mae.state_dict(),
                "mse_state_dict": mse.state_dict(),
                "mape_state_dict": mape.state_dict(),
                "r2_score_state_dict": r2_score.state_dict(),
                "no_improve_count": 0,
            },
            checkpoint_path,
        )

        # Train
        train_model(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_path=str(checkpoint_path),
            num_epochs=self.config["num_epochs"],
            device=self.device,
            patience=self.config["early_stopping_patience"],
        )

        # Evaluate
        metrics = evaluate_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=self.device,
            checkpoint_path=str(checkpoint_path),
        )

        return {
            "loss": metrics[0],
            "mae": metrics[1],
            "r2": metrics[2],
            "mape": metrics[3],
            "smape": metrics[4],
            "rmse": metrics[5],
            "train_size": training_setup["train_size"],
            "test_size": training_setup["test_size"],
        }

    def _split_data(self, df: pd.DataFrame, target: str):
        return df.drop(target, axis=1), df[target]

    def train(self):
        # Load preprocessed data
        data_path = self.input_dir / "properties_clean.csv"
        df = pd.read_csv(data_path)

        if not self._should_train(len(df)):
            print("No training needed - using existing model")
            return {
                "model_path": str(
                    self.artifacts_dir / "checkpoints" / "best_model_checkpoint.pth"
                ),
                "metrics": None,
                "training_performed": False,
            }

        # Setup and run training
        training_setup = self._setup_training(df)
        metrics = self._train_and_evaluate(training_setup)

        return {
            "model_path": str(
                self.artifacts_dir / "checkpoints" / "best_model_checkpoint.pth"
            ),
            "metrics": metrics,
        }


def get_config():
    """Get configuration either from environment variable or default config file"""
    config_path = os.getenv("CONFIG_PATH", "airflow/dags/config/etl_config.json")
    try:
        with open(config_path) as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback config for standalone testing
        return {
            "model": {
                "batch_size": 256,
                "num_epochs": 500,
                "learning_rate": 0.004,
                "weight_decay": 0.001,
                "early_stopping_patience": 20,
                "district_emb_dim": 8,
                "property_emb_dim": 4,
                "dropout_rate": 0.15,
            }
        }


if __name__ == "__main__":
    # This is only used when running the container standalone for testing
    config = get_config()
    input_dir = os.getenv("INPUT_DIR", "/app/data/processed")
    artifacts_dir = os.getenv("ARTIFACTS_DIR", "/app/data/artifacts")

    trainer = PropertyModelTrainer(
        input_dir=input_dir,
        artifacts_dir=artifacts_dir,
        config=config["model"],
        force_retrain=False,
    )
    trainer.train()
