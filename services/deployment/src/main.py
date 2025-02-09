import torch
import joblib
import numpy as np
import streamlit as st
import pandas as pd
import altair as alt

from pathlib import Path
import os
import sys

current_dir = Path(__file__).parent
project_root = current_dir.parents[2]
data_dir = project_root / "data"

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", data_dir / "artifacts"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", data_dir / "processed"))
FRONTEND_DATA = current_dir.parent / "data"

sys.path.append(str(project_root))
# Model imports
from services.trainer.src.utils.model import NeuralNetwork
from services.deployment.src.utils.predict import predict_price
from services.deployment.src.utils.preprocess_input import preprocess_input

# Page configuration
st.set_page_config(
    layout="wide",
    page_title="Property Price Prediction",
    page_icon="🏠",
    initial_sidebar_state="expanded",
)


# Model loading
@st.cache_resource
def load_model(artifacts_dir: Path):
    """Load trained model and config"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = joblib.load(artifacts_dir / "encoders" / "model_config.joblib")

    model = NeuralNetwork(
        numeric_input_dim=model_config["numeric_input_dim"],
        cat_input_dim=model_config["cat_input_dim"],
        num_districts=model_config["num_districts"],
        district_emb_dim=model_config["district_emb_dim"],
        num_properties=model_config["num_properties"],
        property_emb_dim=model_config["property_emb_dim"],
        dropout_rate=model_config["dropout_rate"],
    ).to(device)
    checkpoint = torch.load(
        artifacts_dir / "checkpoints" / "best_model_checkpoint.pth",
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, model_config


# Data loading
@st.cache_data
def load_resources():
    """Load encoders and data"""
    zip_df = pd.read_csv(FRONTEND_DATA / "zip_codes_map.csv")
    zip_df["district_name"] = zip_df.apply(
        lambda row: (
            row["Arrondissement name (French)"].strip()
            if pd.notnull(row["Arrondissement name (French)"])
            and str(row["Arrondissement name (French)"]).strip() != ""
            else str(row["Arrondissement name (Dutch)"]).strip()
        ),
        axis=1,
    )
    zip_to_district = dict(zip(zip_df["Post code"], zip_df["district_name"]))

    # Load encoders
    district_map = joblib.load(ARTIFACTS_DIR / "encoders" / "district_encoder.joblib")
    property_map = joblib.load(ARTIFACTS_DIR / "encoders" / "property_sub_type.joblib")
    scaler = joblib.load(ARTIFACTS_DIR / "encoders" / "numerical_scaler.joblib")
    imputer = joblib.load(ARTIFACTS_DIR / "encoders" / "knn_imputer.joblib")

    # Load processed data
    properties = pd.read_csv(PROCESSED_DIR / "properties_clean.csv")

    return zip_to_district, district_map, property_map, scaler, imputer, properties


# Initialize session state
if "chart_generated" not in st.session_state:
    st.session_state["chart_generated"] = False
if "predicted_price" not in st.session_state:
    st.session_state["predicted_price"] = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, model_config = load_model(ARTIFACTS_DIR)
zip_to_district, district_map, property_map, scaler, imputer, properties = (
    load_resources()
)


# Constants from model config
numeric_cols = model_config["numeric_cols"]
cat_cols = model_config["cat_cols"]

states = {
    "To renovate": 0,
    "To be done up": 1,
    "To restore": 2,
    "Good": 3,
    "Just renovated": 4,
    "As new": 5,
    "Unknown": np.nan,
}

# Title
st.title("🏠 Property Price Prediction Model")


# Input section
with st.expander("🔍 Enter Property Details", expanded=True):
    search_by_zip_code = st.toggle("Search by Zip Code", value=False)
    col1, col2 = st.columns(2)

    with col1:
        if not search_by_zip_code:
            district = st.selectbox(
                "District",
                sorted(district_map.keys()),
                help="Select the district where the property is located.",
            )
        else:
            zip_code = st.selectbox(
                "Zip Code",
                sorted(zip_to_district.keys()),
                help="Select the zip code to search by.",
            )
            st.write(zip_to_district[zip_code])
        nb_bedrooms = st.number_input(
            "Number of Bedrooms",
            step=1,
            min_value=1,
            max_value=10,
            value=3,
            help="Enter the number of bedrooms.",
        )
        living_area = st.number_input(
            "Living Area (sqm)",
            min_value=10,
            value=100,
            help="Select the living area in square meters.",
        )
        surface_of_the_plot = st.number_input(
            "Surface of the Plot (sqm)",
            min_value=0,
            value=500,
            help="Select the surface area of the plot in square meters.",
        )

    with col2:
        state_of_building = st.selectbox(
            "State of the Building",
            sorted(states.keys()),
            help="Select the current state of the building.",
        )
        property_sub_type = st.selectbox(
            "Property Type",
            sorted(property_map.keys()),
            help="Select the type of property.",
        )
        equipped_kitchen = st.toggle("Equipped Kitchen", value=False)
        garden = st.toggle("Garden", value=False)
        terrace = st.toggle("Terrace", value=False)
        swimming_pool = st.toggle("Swimming Pool", value=False)
        furnished = st.toggle("Furnished", value=False)

if st.button("Generate Prediction"):
    st.session_state["chart_generated"] = True


if st.session_state["chart_generated"]:
    with st.spinner("Generating prediction..."):
        collected_input = {
            "nb_bedrooms": nb_bedrooms,
            "living_area": living_area,
            "surface_of_the_plot": surface_of_the_plot,
            "state_of_building": states[state_of_building],
            "equipped_kitchen": equipped_kitchen,
            "district": zip_to_district[zip_code] if search_by_zip_code else district,
            "property_sub_type": property_sub_type,
            "garden": garden,
            "swimming_pool": swimming_pool,
            "terrace": terrace,
            "furnished": furnished,
        }

        processed_input = preprocess_input(
            collected_input,
            property_map,
            district_map,
            numeric_cols,
            cat_cols,
            scaler,
            imputer,
        )

        predicted_price = np.int32(
            predict_price(
                processed_input,
                model,
                numeric_cols,
                cat_cols,
                device,
            )
        )

        st.session_state["predicted_price"] = predicted_price[0]
        st.success(f"**Predicted Price:** {st.session_state['predicted_price']} €")

    # Visualization
    col1, col2 = st.columns([2, 1])
    with col1:
        filtered_properties = properties[
            properties["district"] == collected_input["district"]
        ]
        hist = (
            alt.Chart(filtered_properties)
            .mark_bar(color="#1E90FF")
            .encode(
                alt.X("price", bin=alt.Bin(maxbins=30), title="Price (€)"),
                alt.Y("count()", title="Number of Properties"),
                tooltip=["count()", "price"],
            )
            .properties(width=600, height=400)
        )

        rule = (
            alt.Chart(pd.DataFrame({"price": [st.session_state["predicted_price"]]}))
            .mark_rule(color="red", strokeWidth=2)
            .encode(alt.X("price"))
        )
        price_distribution = hist + rule
        final_chart = price_distribution
        st.subheader(
            f"Price Distribution of Similar Properties in {collected_input['district']}"
        )
        st.altair_chart(final_chart, use_container_width=True)

    with col2:
        st.markdown("### 📝 Details of the Predicted Property")
        st.write(f"**Number of Bedrooms:** {collected_input['nb_bedrooms']}")
        st.write(f"**Living Area:** {collected_input['living_area']} sqm")
        st.write(
            f"**Surface of the Plot:** {collected_input['surface_of_the_plot']} sqm"
        )
        st.write(f"**State of Building:** {state_of_building}")
        st.write(f"**Property Type:** {property_sub_type}")
        amenities = []
        if collected_input["equipped_kitchen"]:
            amenities.append("Equipped Kitchen")
        if collected_input["garden"]:
            amenities.append("Garden")
        if collected_input["terrace"]:
            amenities.append("Terrace")
        if collected_input["swimming_pool"]:
            amenities.append("Swimming Pool")
        if collected_input["furnished"]:
            amenities.append("Furnished")
        if len(amenities):
            st.write("**Amenities:** " + ", ".join(amenities) if amenities else "None")
