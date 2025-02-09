import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from typing import Tuple
from pathlib import Path


def load_data(houses_path: str, apartments_path: str) -> pd.DataFrame:
    """
    Loads and merges the housing and apartment data from the specified paths.
    """
    houses = pd.read_csv(houses_path)
    apartments = pd.read_csv(apartments_path)
    df = pd.concat([houses, apartments], axis=0)
    return df


def preprocess_data(df: pd.DataFrame, artifacts_dir: Path) -> pd.DataFrame:
    """
    Perform preprocessing tasks such as dropping columns, removing outliers,
    encoding categorical variables, and scaling numerical features.
    """
    # cleaning
    df.drop(
        [
            "terrace_surface",
            "garden_surface",
            "id",
            "locality",
            "nb_facades",
        ],
        axis=1,
        inplace=True,
    )
    # Drop rows with too many missing values
    threshold = 0.05 * len(df)
    columns_low_missing_data = df.columns[df.isna().sum() < threshold]
    df.dropna(subset=columns_low_missing_data, inplace=True)

    # Drop duplicates
    df.drop_duplicates(
        subset=["zip_code", "latitude", "longitude"], keep="first", inplace=True
    )

    # One-hot encode property_type (This differentiates house vs apartment)
    df = pd.concat([df, pd.get_dummies(df["property_type"])], axis=1)
    df.drop("property_type", axis=1, inplace=True)
    # Province classification based on zip_code patterns
    provinces = [
        "Bruxelles_province",
        "Brabant_Wallon_province",
        "Brabant_Flamand_province",
        "Anvers_province",
        "Limbourg_province",
        "Liège_province",
        "Namur_province",
        "Hainaut_province",
        "Luxembourg_province",
        "Flandre_Occidentale_province",
        "Flandre_Orientale_province",
    ]

    brussels_capital = "^10|^11|^12"
    walloon_brabant = "^13|^14"
    flemish_brabant = "^15|^16|^17|^18|^19|^30|^31|^32|^33|^34"
    antwerp = "^20|^21|^22|^23|^24|^25|^26|^27|^28|^29"
    limburg = "^35|^36|^37|^38|^39"
    liège = "^40|^41|^42|^43|^44|^45|^46|^47|^48|^49"
    namur = "^50|^51|^55|^53|^55|^55|^56|^57|^58|^59"
    hainaut = "^60|^61|^62|63|^64|^65|^70|^71|^77|^73|^74|^75|^76|^77|^78|^79"
    luxembourg = "^66|^67|^68|^69"
    west_flanders = "^80|^81|^82|^83|^84|^85|^86|^87|^88|^89"
    east_flanders = "^90|^91|^92|^93|^94|^95|^96|^97|^98|^99"

    conditions = [
        df["zip_code"].astype(str).str.contains(brussels_capital),
        df["zip_code"].astype(str).str.contains(walloon_brabant),
        df["zip_code"].astype(str).str.contains(flemish_brabant),
        df["zip_code"].astype(str).str.contains(antwerp),
        df["zip_code"].astype(str).str.contains(limburg),
        df["zip_code"].astype(str).str.contains(liège),
        df["zip_code"].astype(str).str.contains(namur),
        df["zip_code"].astype(str).str.contains(hainaut),
        df["zip_code"].astype(str).str.contains(luxembourg),
        df["zip_code"].astype(str).str.contains(west_flanders),
        df["zip_code"].astype(str).str.contains(east_flanders),
    ]

    df["province"] = np.select(conditions, provinces, default="Other")

    # Drop rows with missing essential features
    df["surface_of_the_plot"] = df.apply(
        lambda row: (
            0
            if row["APARTMENT"] == 1 and pd.isna(row["surface_of_the_plot"])
            else row["surface_of_the_plot"]
        ),
        axis=1,
    )

    df.dropna(
        subset=[
            "latitude",
            "longitude",
            "state_of_building",
            "living_area",
            "surface_of_the_plot",
        ],
        inplace=True,
    )
    df = remove_outliers(df.copy(), ["price", "living_area", "surface_of_the_plot"])

    # Encode state_of_building
    states = [
        "To renovate",
        "To be done up",
        "To restore",
        "Good",
        "Just renovated",
        "As new",
    ]
    # df.to_csv("./data/processed/properties.csv")

    ordinal_encoder = OrdinalEncoder(categories=[states])
    state = df["state_of_building"]
    state_not_null = state[state.notnull()].values.reshape(-1, 1)
    state_encoded = ordinal_encoder.fit_transform(state_not_null)
    df.loc[state.notnull(), "state_of_building"] = np.squeeze(state_encoded)

    district_encoder = LabelEncoder()
    property_sub_type_encoder = LabelEncoder()

    district_encoder.fit(df["district"])
    property_sub_type_encoder.fit(df["property_sub_type"])
    district_encoder_mapping = {
        district: int(idx)
        for district, idx in zip(
            district_encoder.classes_,
            district_encoder.transform(district_encoder.classes_),
        )
    }
    property_sub_type_encode_mapping = {
        prop_type: int(idx)
        for prop_type, idx in zip(
            property_sub_type_encoder.classes_,
            property_sub_type_encoder.transform(property_sub_type_encoder.classes_),
        )
    }
    df["district_id"] = district_encoder.fit_transform(df["district"].astype(str))
    df["property_sub_type_id"] = property_sub_type_encoder.fit_transform(
        df["property_sub_type"].astype(str)
    )

    encoders_dir = artifacts_dir / "encoders"
    encoders_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(district_encoder_mapping, encoders_dir / "district_encoder.joblib")
    joblib.dump(
        property_sub_type_encode_mapping, encoders_dir / "property_sub_type.joblib"
    )

    df.drop(
        [
            "property_sub_type",
            "province",
            "latitude",
            "longitude",
            "zip_code",
            "APARTMENT",
            "HOUSE",
            "fireplace",
        ],
        axis=1,
        inplace=True,
    )
    return df


def remove_outliers(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Removes outliers from the specified columns using the Interquartile Range (IQR) method.
    """
    for col in columns:
        if col in data.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1 * IQR
            upper_bound = Q3 + 1 * IQR
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    return data


def split_data(
    df: pd.DataFrame, target: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the data into training and testing sets, and returns the feature set and target.
    """
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y
