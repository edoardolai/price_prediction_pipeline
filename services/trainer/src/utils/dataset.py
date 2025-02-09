import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import List, Tuple


class CustomDataset(Dataset):
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        numeric_cols: List[str],
        cat_cols: List[str],
    ) -> None:
        """
        Custom dataset for handling features and labels.

        Parameters:
        - X: A DataFrame containing the features.
        - y: A Series containing the target values.
        - numeric_cols: List of column names for numeric features.
        - cat_cols: List of column names for categorical features.
        """
        self.X_numeric = torch.tensor(X[numeric_cols].values, dtype=torch.float32)
        self.X_categorical = torch.tensor(X[cat_cols].values, dtype=torch.float32)
        self.X_district = torch.tensor(X["district_id"].values, dtype=torch.long)
        self.X_property = torch.tensor(
            X["property_sub_type_id"].values, dtype=torch.long
        )
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
        return len(self.y)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample from the dataset.

        Parameters:
        - idx: The index of the sample to retrieve.

        Returns:
        - A tuple containing the numeric features, district IDs, property type IDs, and the target value.
        """
        return (
            self.X_numeric[idx],
            self.X_categorical[idx],
            self.X_district[idx],
            self.X_property[idx],
            self.y[idx],
        )
