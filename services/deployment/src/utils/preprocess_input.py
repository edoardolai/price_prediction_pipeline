import pandas as pd


def preprocess_input(
    input_data, property_map, district_map, numeric_cols, cat_cols, scaler, imputer
):
    """
    Preprocesses a single property input for prediction.

    Args:
        property_data (dict): Dictionary containing property features.

    Returns:
        pd.DataFrame: Preprocessed data ready for prediction.
    """
    try:
        translated_district = district_map[input_data["district"]]
        translated_property_sub_type = property_map[input_data["property_sub_type"]]

        translated_input = {
            "nb_bedrooms": input_data["nb_bedrooms"],
            "living_area": input_data["living_area"],
            "surface_of_the_plot": input_data["surface_of_the_plot"],
            "state_of_building": input_data["state_of_building"],
            "equipped_kitchen": input_data["equipped_kitchen"],
            "district_id": translated_district,
            "property_sub_type_id": translated_property_sub_type,
            "garden": input_data["garden"],
            "swimming_pool": input_data["swimming_pool"],
            "terrace": input_data["terrace"],
            "furnished": input_data["furnished"],
        }

        input_df = pd.DataFrame([translated_input])
        impute_cols = [
            "state_of_building",
            "surface_of_the_plot",
            "nb_bedrooms",
            "living_area",
        ]
        input_df[impute_cols] = imputer.transform(input_df[impute_cols])
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        for col in cat_cols:
            input_df[col] = input_df[col].astype(int)

        return input_df

    except KeyError as e:
        print(f"Error: {e}")
        return None
