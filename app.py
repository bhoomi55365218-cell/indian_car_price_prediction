import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ---------------------------------------------------------
# STREAMLIT CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Indian Pre-Owned Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# ---------------------------------------------------------
# FILE PATH
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "car_price_model_small.pkl"

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


model = load_model()

if model is None:
    st.error(
        "Model file not found: car_price_model_small.pkl"
    )
    st.stop()

# ---------------------------------------------------------
# MODEL FEATURES
# ---------------------------------------------------------
if not hasattr(model, "feature_names_in_"):
    st.error(
        "The saved model does not contain feature_names_in_. "
        "Please use the same model file that worked in Colab."
    )
    st.stop()

expected_features = list(model.feature_names_in_)

# ---------------------------------------------------------
# COLUMNS IN ORIGINAL DATASET
# ---------------------------------------------------------
categorical_bases = [
    "Maker",
    "model",
    "Location",
    "Owner Type",
    "body_type",
    "transmission",
    "fuel_type"
]

numeric_features = [
    "ID",
    "Distance ",
    "manufacture_year",
    "Age of car",
    "engine_displacement",
    "engine_power",
    "Vroom Audit Rating",
    "door_count",
    "seat_count"
]

# ---------------------------------------------------------
# GET CATEGORIES FROM ONE-HOT ENCODED MODEL FEATURES
# ---------------------------------------------------------
def get_categories(base_name):
    prefix = base_name + "_"
    values = []

    for feature in expected_features:
        if feature.startswith(prefix):
            values.append(feature[len(prefix):])

    return sorted(values)


# ---------------------------------------------------------
# BUILD MODEL INPUT FROM USER VALUES
# ---------------------------------------------------------
def build_model_input(values):
    # Start with all model features set to zero
    row = {feature: 0 for feature in expected_features}

    # Numeric columns
    for feature in numeric_features:
        if feature in expected_features and feature in values:
            row[feature] = values[feature]

    # Categorical columns
    for feature in categorical_bases:
        if feature not in values:
            continue

        category = str(values[feature])
        dummy_name = f"{feature}_{category}"

        if dummy_name in row:
            row[dummy_name] = 1

    return pd.DataFrame([row], columns=expected_features)


# ---------------------------------------------------------
# CONVERT BATCH DATA TO MODEL FEATURES
# ---------------------------------------------------------
def prepare_batch_for_model(df):
    df = df.copy()

    # Fix Distance column if CSV has no trailing space
    if "Distance" in df.columns and "Distance " not in df.columns:
        df = df.rename(columns={"Distance": "Distance "})

    # Remove target if present
    if "Price" in df.columns:
        df = df.drop(columns=["Price"])

    output = pd.DataFrame(
        0,
        index=df.index,
        columns=expected_features
    )

    # Numeric columns
    for feature in numeric_features:
        if feature in df.columns and feature in output.columns:
            output[feature] = pd.to_numeric(
                df[feature],
                errors="coerce"
            ).fillna(0)

    # Categorical columns
    for feature in categorical_bases:
        if feature not in df.columns:
            continue

        for idx, value in df[feature].items():
            dummy_name = f"{feature}_{value}"

            if dummy_name in output.columns:
                output.loc[idx, dummy_name] = 1

    return output


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------
st.title("🚗 Indian Pre-Owned Car Price Prediction")
st.write(
    "Use machine learning to estimate the market price of a pre-owned car."
)

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    [
        "🎯 Price Prediction",
        "📁 Batch Prediction",
        "📊 Model Insights"
    ]
)

# =========================================================
# TAB 1 - PRICE PREDICTION
# =========================================================
with tab1:

    st.subheader("Enter Car Details")

    col1, col2, col3 = st.columns(3)

    # Maker
    maker_options = get_categories("Maker")
    with col1:
        maker = st.selectbox(
            "Maker",
            maker_options if maker_options else ["Unknown"]
        )

    # Model
    model_options = get_categories("model")
    with col2:
        model_name = st.selectbox(
            "Model",
            model_options if model_options else ["Unknown"]
        )

    # Location
    location_options = get_categories("Location")
    with col3:
        location = st.selectbox(
            "Location",
            location_options if location_options else ["Unknown"]
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        owner_options = get_categories("Owner Type")
        owner_type = st.selectbox(
            "Owner Type",
            owner_options if owner_options else ["Unknown"]
        )

    with col5:
        fuel_options = get_categories("fuel_type")
        fuel_type = st.selectbox(
            "Fuel Type",
            fuel_options if fuel_options else ["Unknown"]
        )

    with col6:
        transmission_options = get_categories("transmission")
        transmission = st.selectbox(
            "Transmission",
            transmission_options if transmission_options else ["Unknown"]
        )

    col7, col8, col9 = st.columns(3)

    with col7:
        body_options = get_categories("body_type")
        body_type = st.selectbox(
            "Body Type",
            body_options if body_options else ["Unknown"]
        )

    with col8:
        distance = st.number_input(
            "Distance driven (km)",
            min_value=0.0,
            value=65000.0,
            step=1000.0
        )

    with col9:
        age = st.number_input(
            "Age of car (years)",
            min_value=0,
            max_value=50,
            value=5,
            step=1
        )

    col10, col11, col12 = st.columns(3)

    with col10:
        manufacture_year = st.number_input(
            "Manufacture year",
            min_value=1900,
            max_value=2100,
            value=2020,
            step=1
        )

    with col11:
        engine_displacement = st.number_input(
            "Engine displacement",
            min_value=0.0,
            value=1896.0,
            step=50.0
        )

    with col12:
        engine_power = st.number_input(
            "Engine power",
            min_value=0.0,
            value=91.0,
            step=1.0
        )

    col13, col14, col15 = st.columns(3)

    with col13:
        vroom_rating = st.number_input(
            "Vroom Audit Rating",
            min_value=0,
            value=6,
            step=1
        )

    with col14:
        door_count = st.number_input(
            "Door Count",
            min_value=1,
            max_value=10,
            value=4,
            step=1
        )

    with col15:
        seat_count = st.number_input(
            "Seat Count",
            min_value=1,
            max_value=20,
            value=5,
            step=1
        )

    st.divider()

    if st.button(
        "🔮 Predict Price",
        type="primary",
        use_container_width=True
    ):

        values = {
            "ID": 0,
            "Maker": maker,
            "model": model_name,
            "Location": location,
            "Distance ": distance,
            "Owner Type": owner_type,
            "manufacture_year": manufacture_year,
            "Age of car": age,
            "engine_displacement": engine_displacement,
            "engine_power": engine_power,
            "body_type": body_type,
            "Vroom Audit Rating": vroom_rating,
            "transmission": transmission,
            "door_count": door_count,
            "seat_count": seat_count,
            "fuel_type": fuel_type
        }

        try:
            model_input = build_model_input(values)

            prediction = float(
                model.predict(model_input)[0]
            )

            lower = prediction * 0.95
            upper = prediction * 1.05

            st.success("Prediction completed successfully!")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Estimated Price",
                f"₹{prediction:,.0f}"
            )

            c2.metric(
                "5% Lower Range",
                f"₹{lower:,.0f}"
            )

            c3.metric(
                "5% Upper Range",
                f"₹{upper:,.0f}"
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# =========================================================
# TAB 2 - BATCH PREDICTION
# =========================================================
with tab2:

    st.subheader("Batch Prediction")

    uploaded_file = st.file_uploader(
        "Upload Test CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            batch = pd.read_csv(uploaded_file)

            st.write("Uploaded Data")
            st.dataframe(
                batch.head(10),
                use_container_width=True
            )

            model_input = prepare_batch_for_model(batch)

            predictions = model.predict(model_input)

            result = batch.copy()
            result["Price"] = predictions

            if "ID" in result.columns:
                result = result[["ID", "Price"]]

            st.success(
                f"Predicted {len(result):,} rows successfully."
            )

            st.dataframe(
                result.head(20),
                use_container_width=True
            )

            csv_data = result.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Predictions",
                data=csv_data,
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as e:
            st.error(
                f"Batch prediction failed: {e}"
            )


# =========================================================
# TAB 3 - MODEL INSIGHTS
# =========================================================
with tab3:

    st.subheader("Model Insights")

    st.write(
        "This deployed version derives the available categorical "
        "options directly from the trained model."
    )

    st.write("### Model Information")

    info1, info2 = st.columns(2)

    with info1:
        st.metric(
            "Number of Model Features",
            len(expected_features)
        )

    with info2:
        st.metric(
            "Number of Categorical Fields",
            len(categorical_bases)
        )

    st.write("### Available Categories")

    for feature in categorical_bases:
        categories = get_categories(feature)

        if categories:
            st.write(
                f"**{feature}:** {len(categories)} categories"
            )

    st.write("### Model Feature Names")

    with st.expander("Show model features"):
        st.code(
            "\n".join(expected_features)
        )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.sidebar.divider()
st.sidebar.caption(
    "Indian Pre-Owned Car Price Prediction"
)
