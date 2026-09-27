import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path


# =========================================================
# STREAMLIT CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Indian Pre-Owned Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)


# =========================================================
# PROJECT FILE PATHS
# =========================================================

# app.py is stored in the GitHub repository root
ROOT = Path(__file__).resolve().parent

# Your current model filename
MODEL_PATH = ROOT / "car_price_model_small.pkl"

# Optional training data
TRAIN_PATH = ROOT / "Cap_Training_Data_2025.csv"

TARGET = "Price"
DISTANCE_COL = "Distance "


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    # First try the current model
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    # Fallback if you rename it later
    alternate_model = ROOT / "car_price_model.pkl"

    if alternate_model.exists():
        return joblib.load(alternate_model)

    return None


# =========================================================
# LOAD TRAINING DATA
# =========================================================

@st.cache_data
def load_training_data():

    if not TRAIN_PATH.exists():
        return None

    try:
        df = pd.read_csv(TRAIN_PATH)

        # Check that the file actually contains data
        if df.empty or len(df.columns) == 0:
            return None

        return df

    except Exception:
        return None


# =========================================================
# LOAD MODEL AND DATA
# =========================================================

model = load_model()
train = load_training_data()


# =========================================================
# CHECK MODEL
# =========================================================

if model is None:

    st.error(
        "Trained model not found. "
        "Please upload car_price_model_small.pkl "
        "to the GitHub repository root."
    )

    st.stop()


# =========================================================
# GET MODEL FEATURE NAMES
# =========================================================

if not hasattr(model, "feature_names_in_"):

    st.error(
        "The saved model does not contain feature_names_in_. "
        "This app expects the same model that was used during training."
    )

    st.stop()


expected_features = list(model.feature_names_in_)


# =========================================================
# CATEGORICAL FEATURES
# =========================================================

categorical_bases = [
    "Maker",
    "model",
    "Location",
    "Owner Type",
    "body_type",
    "transmission",
    "fuel_type"
]


# =========================================================
# NUMERICAL FEATURES
# =========================================================

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


# =========================================================
# GET CATEGORIES FROM MODEL
# =========================================================

def get_model_categories(base_name):

    prefix = base_name + "_"

    categories = []

    for feature in expected_features:

        if feature.startswith(prefix):

            value = feature[len(prefix):]

            categories.append(value)

    return sorted(categories)


# =========================================================
# GET CATEGORIES
# =========================================================

def get_categories(base_name):

    # Use real training data when available
    if train is not None and base_name in train.columns:

        values = (
            train[base_name]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if values:
            return sorted(values)

    # Otherwise use the model's one-hot feature names
    return get_model_categories(base_name)


# =========================================================
# GET DEFAULT NUMERIC VALUE
# =========================================================

def get_default_value(column, default):

    if train is not None and column in train.columns:

        series = pd.to_numeric(
            train[column],
            errors="coerce"
        ).dropna()

        if len(series) > 0:
            return float(series.median())

    return default


# =========================================================
# BUILD SINGLE PREDICTION INPUT
# =========================================================

def build_model_input(values):

    # Create all expected model columns
    row = {
        feature: 0
        for feature in expected_features
    }

    # -----------------------------------------------------
    # Numeric features
    # -----------------------------------------------------

    for feature in numeric_features:

        if feature in expected_features:

            if feature in values:

                row[feature] = values[feature]


    # -----------------------------------------------------
    # Categorical features
    # -----------------------------------------------------

    for feature in categorical_bases:

        if feature not in values:
            continue

        category = str(values[feature])

        dummy_name = f"{feature}_{category}"

        if dummy_name in row:

            row[dummy_name] = 1


    # -----------------------------------------------------
    # DataFrame in exact training order
    # -----------------------------------------------------

    return pd.DataFrame(
        [row],
        columns=expected_features
    )


# =========================================================
# PREPARE BATCH DATA
# =========================================================

def prepare_batch_for_model(df):

    df = df.copy()

    # -----------------------------------------------------
    # Fix Distance column
    # -----------------------------------------------------

    if (
        "Distance" in df.columns
        and "Distance " not in df.columns
    ):

        df = df.rename(
            columns={
                "Distance": "Distance "
            }
        )


    # -----------------------------------------------------
    # Remove target if uploaded CSV contains Price
    # -----------------------------------------------------

    if TARGET in df.columns:

        df = df.drop(
            columns=[TARGET]
        )


    # -----------------------------------------------------
    # Create DataFrame with model columns
    # -----------------------------------------------------

    output = pd.DataFrame(
        0,
        index=df.index,
        columns=expected_features
    )


    # -----------------------------------------------------
    # Numeric columns
    # -----------------------------------------------------

    for feature in numeric_features:

        if (
            feature in df.columns
            and feature in output.columns
        ):

            output[feature] = pd.to_numeric(
                df[feature],
                errors="coerce"
            ).fillna(0)


    # -----------------------------------------------------
    # Categorical columns
    # -----------------------------------------------------

    for feature in categorical_bases:

        if feature not in df.columns:
            continue

        for index, value in df[feature].items():

            dummy_name = f"{feature}_{value}"

            if dummy_name in output.columns:

                output.loc[
                    index,
                    dummy_name
                ] = 1


    return output


# =========================================================
# APPLICATION TITLE
# =========================================================

st.title(
    "🚗 Indian Pre-Owned Car Price Prediction"
)

st.write(
    "Enter the car details to estimate its market price."
)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🎯 Price Prediction",
        "📁 Batch Prediction",
        "📊 Insights"
    ]
)


# =========================================================
# TAB 1 - PRICE PREDICTION
# =========================================================

with tab1:

    st.subheader("Car Details")

    col1, col2, col3 = st.columns(3)


    # -----------------------------------------------------
    # Maker
    # -----------------------------------------------------

    maker_options = get_categories("Maker")

    with col1:

        maker = st.selectbox(
            "Maker",
            maker_options
            if maker_options
            else ["Unknown"]
        )


    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model_options = get_categories("model")

    with col2:

        model_name = st.selectbox(
            "Model",
            model_options
            if model_options
            else ["Unknown"]
        )


    # -----------------------------------------------------
    # Location
    # -----------------------------------------------------

    location_options = get_categories("Location")

    with col3:

        location = st.selectbox(
            "Location",
            location_options
            if location_options
            else ["Unknown"]
        )


    # -----------------------------------------------------
    # Second row
    # -----------------------------------------------------

    col4, col5, col6 = st.columns(3)


    with col4:

        owner_options = get_categories(
            "Owner Type"
        )

        owner_type = st.selectbox(
            "Owner Type",
            owner_options
            if owner_options
            else ["Unknown"]
        )


    with col5:

        fuel_options = get_categories(
            "fuel_type"
        )

        fuel_type = st.selectbox(
            "Fuel Type",
            fuel_options
            if fuel_options
            else ["Unknown"]
        )


    with col6:

        transmission_options = get_categories(
            "transmission"
        )

        transmission = st.selectbox(
            "Transmission",
            transmission_options
            if transmission_options
            else ["Unknown"]
        )


    # -----------------------------------------------------
    # Third row
    # -----------------------------------------------------

    col7, col8, col9 = st.columns(3)


    with col7:

        body_options = get_categories(
            "body_type"
        )

        body_type = st.selectbox(
            "Body Type",
            body_options
            if body_options
            else ["Unknown"]
        )


    with col8:

        distance = st.number_input(
            "Distance driven (km)",
            min_value=0.0,
            max_value=20_000_000.0,
            value=get_default_value(
                DISTANCE_COL,
                65000.0
            ),
            step=1000.0
        )


    with col9:

        age = st.number_input(
            "Age of car (years)",
            min_value=0,
            max_value=100,
            value=int(
                get_default_value(
                    "Age of car",
                    5
                )
            ),
            step=1
        )


    # -----------------------------------------------------
    # Fourth row
    # -----------------------------------------------------

    col10, col11, col12 = st.columns(3)


    with col10:

        manufacture_year = st.number_input(
            "Manufacture year",
            min_value=1900,
            max_value=2100,
            value=int(
                get_default_value(
                    "manufacture_year",
                    2020
                )
            ),
            step=1
        )


    with col11:

        engine_displacement = st.number_input(
            "Engine displacement",
            min_value=0.0,
            max_value=100000.0,
            value=get_default_value(
                "engine_displacement",
                1896.0
            ),
            step=50.0
        )


    with col12:

        engine_power = st.number_input(
            "Engine power",
            min_value=0.0,
            max_value=2000.0,
            value=get_default_value(
                "engine_power",
                91.0
            ),
            step=1.0
        )


    # -----------------------------------------------------
    # Fifth row
    # -----------------------------------------------------

    col13, col14, col15 = st.columns(3)


    with col13:

        vroom_rating = st.number_input(
            "Vroom Audit Rating",
            min_value=0,
            max_value=20,
            value=int(
                get_default_value(
                    "Vroom Audit Rating",
                    6
                )
            ),
            step=1
        )


    with col14:

        door_count = st.number_input(
            "Door Count",
            min_value=1,
            max_value=10,
            value=int(
                get_default_value(
                    "door_count",
                    4
                )
            ),
            step=1
        )


    with col15:

        seat_count = st.number_input(
            "Seat Count",
            min_value=1,
            max_value=20,
            value=int(
                get_default_value(
                    "seat_count",
                    5
                )
            ),
            step=1
        )


    st.divider()


    # =====================================================
    # PREDICTION BUTTON
    # =====================================================

    if st.button(
        "🔮 Predict Price",
        type="primary",
        use_container_width=True
    ):

        values = {

            # ID is required by your trained model
            "ID": 0,

            "Maker": maker,

            "model": model_name,

            "Location": location,

            "Distance ": distance,

            "Owner Type": owner_type,

            "manufacture_year": manufacture_year,

            "Age of car": age,

            "engine_displacement":
                engine_displacement,

            "engine_power":
                engine_power,

            "body_type":
                body_type,

            "Vroom Audit Rating":
                vroom_rating,

            "transmission":
                transmission,

            "door_count":
                door_count,

            "seat_count":
                seat_count,

            "fuel_type":
                fuel_type
        }


        try:

            model_input = build_model_input(
                values
            )

            prediction = float(
                model.predict(model_input)[0]
            )


            # -------------------------------------------------
            # Valuation range
            # -------------------------------------------------

            lower = prediction * 0.95
            upper = prediction * 1.05


            st.success(
                "Prediction completed successfully!"
            )


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


            st.subheader(
                "Selected Car"
            )

            st.write(
                f"**Maker:** {maker}"
            )

            st.write(
                f"**Model:** {model_name}"
            )

            st.write(
                f"**Location:** {location}"
            )


        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )


# =========================================================
# TAB 2 - BATCH PREDICTION
# =========================================================

with tab2:

    st.subheader(
        "📁 Batch Prediction"
    )

    st.write(
        "Upload your test CSV file to generate "
        "prices for multiple cars."
    )


    uploaded_file = st.file_uploader(
        "Upload Test CSV",
        type=["csv"]
    )


    if uploaded_file is not None:

        try:

            batch = pd.read_csv(
                uploaded_file
            )


            st.write(
                "Uploaded Data"
            )


            st.dataframe(
                batch.head(10),
                use_container_width=True
            )


            # ---------------------------------------------
            # Prepare data
            # ---------------------------------------------

            model_input = prepare_batch_for_model(
                batch
            )


            # ---------------------------------------------
            # Predict
            # ---------------------------------------------

            predictions = model.predict(
                model_input
            )


            # ---------------------------------------------
            # Output
            # ---------------------------------------------

            result = batch.copy()

            result["Price"] = predictions


            # Required submission format
            if "ID" in result.columns:

                result = result[
                    ["ID", "Price"]
                ]


            st.success(
                f"Predicted {len(result):,} rows successfully."
            )


            st.dataframe(
                result.head(20),
                use_container_width=True
            )


            # ---------------------------------------------
            # Download CSV
            # ---------------------------------------------

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
# TAB 3 - INSIGHTS
# =========================================================

with tab3:

    st.subheader(
        "📊 Insights Dashboard"
    )


    # -----------------------------------------------------
    # If training data exists
    # -----------------------------------------------------

    if train is not None:

        # Age vs price
        if (
            "Age of car" in train.columns
            and "Price" in train.columns
        ):

            age_price = (
                train
                .groupby("Age of car")["Price"]
                .mean()
                .sort_index()
            )

            st.write(
                "### Average Price by Car Age"
            )

            st.line_chart(
                age_price
            )


        # Maker vs price
        if (
            "Maker" in train.columns
            and "Price" in train.columns
        ):

            maker_price = (
                train
                .groupby("Maker")["Price"]
                .mean()
                .sort_values(
                    ascending=False
                )
            )

            st.write(
                "### Average Price by Maker"
            )

            st.bar_chart(
                maker_price
            )


        # Distance vs price
        if (
            DISTANCE_COL in train.columns
            and "Price" in train.columns
        ):

            st.write(
                "### Price vs Distance"
            )

            distance_data = train[
                [
                    DISTANCE_COL,
                    "Price"
                ]
            ].dropna()


            # Keep chart responsive
            if len(distance_data) > 5000:

                step = max(
                    1,
                    len(distance_data) // 5000
                )

                distance_data = (
                    distance_data.iloc[::step]
                )


            distance_data = (
                distance_data
                .sort_values(DISTANCE_COL)
            )


            st.line_chart(
                distance_data.set_index(
                    DISTANCE_COL
                )["Price"]
            )


        st.write(
            "### Training Data Preview"
        )

        st.dataframe(
            train.head(20),
            use_container_width=True
        )


    # -----------------------------------------------------
    # No training CSV
    # -----------------------------------------------------

    else:

        st.info(
            "Training CSV is not available in the deployed "
            "repository. Prediction and batch prediction "
            "still work using the trained model."
        )


        st.write(
            "### Model Information"
        )


        c1, c2 = st.columns(2)


        c1.metric(
            "Model Features",
            len(expected_features)
        )


        c2.metric(
            "Categorical Features",
            len(categorical_bases)
        )


        with st.expander(
            "Show Model Features"
        ):

            st.code(
                "\n".join(
                    expected_features
                )
            )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.divider()

st.sidebar.title(
    "About"
)

st.sidebar.write(
    "Indian Pre-Owned Car Price Prediction"
)

st.sidebar.write(
    "Machine Learning + Streamlit"
)

st.sidebar.write(
    "Single Prediction | Batch Prediction | Insights"
)
