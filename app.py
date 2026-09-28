import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Indian Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

# IMPORTANT:
# app.py, model and optional training CSV are in the
# GitHub repository root.

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "car_price_model_small.pkl"
ALTERNATE_MODEL_PATH = ROOT / "car_price_model.pkl"

# Training data is OPTIONAL for deployment.
TRAIN_PATH = ROOT / "Cap_Training_Data_2025.csv"

TARGET = "Price"
DISTANCE_COL = "Distance "


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    """Load the trained model."""

    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    if ALTERNATE_MODEL_PATH.exists():
        return joblib.load(ALTERNATE_MODEL_PATH)

    return None


model = load_model()


# ============================================================
# CHECK MODEL
# ============================================================

if model is None:
    st.error(
        "❌ Model file not found.\n\n"
        "Make sure `car_price_model_small.pkl` "
        "is uploaded to the GitHub repository root."
    )
    st.stop()


# ============================================================
# MODEL FEATURE NAMES
# ============================================================

expected_features = list(
    getattr(model, "feature_names_in_", [])
)

if not expected_features:
    st.error(
        "❌ The saved model does not contain "
        "`feature_names_in_`.\n\n"
        "Please use the same trained model that "
        "worked in Google Colab."
    )
    st.stop()


# ============================================================
# LOAD TRAINING DATA - OPTIONAL
# ============================================================

@st.cache_data
def load_training_data():
    """
    Load the training dataset when it exists.

    The Streamlit app can still run without the large CSV.
    """

    if not TRAIN_PATH.exists():
        return None

    try:

        # Ignore empty / placeholder GitHub files
        if TRAIN_PATH.stat().st_size < 100:
            return None

        df = pd.read_csv(TRAIN_PATH)

        if df.empty:
            return None

        if TARGET not in df.columns:
            return None

        return df

    except Exception:
        return None


train = load_training_data()


# ============================================================
# FEATURES
# ============================================================

CATEGORICAL_COLUMNS = [
    "Maker",
    "model",
    "Location",
    "Owner Type",
    "body_type",
    "transmission",
    "fuel_type"
]

NUMERIC_COLUMNS = [
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


# ============================================================
# DETECT ONE-HOT MODEL
# ============================================================

def uses_one_hot_encoding():
    """Check whether the model expects one-hot columns."""

    for feature in expected_features:

        feature = str(feature)

        for column in CATEGORICAL_COLUMNS:

            if feature.startswith(column + "_"):
                return True

    return False


ONE_HOT_MODEL = uses_one_hot_encoding()


# ============================================================
# GET CATEGORY VALUES
# ============================================================

def get_categories(column):
    """
    Get dropdown values from training data when possible.
    Otherwise get them from model feature names.
    """

    # --------------------------------------------------------
    # Training data available
    # --------------------------------------------------------

    if (
        train is not None
        and column in train.columns
    ):

        values = (
            train[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if values:
            return sorted(values)


    # --------------------------------------------------------
    # Training data unavailable
    # --------------------------------------------------------

    prefix = column + "_"

    values = []

    for feature in expected_features:

        feature = str(feature)

        if feature.startswith(prefix):

            values.append(
                feature[len(prefix):]
            )

    return sorted(set(values))


# ============================================================
# NUMERIC DEFAULT
# ============================================================

def get_numeric_default(
    column,
    fallback
):
    """
    Get median values from the training dataset.
    If the dataset isn't available, use fallback.
    """

    if (
        train is not None
        and column in train.columns
    ):

        values = pd.to_numeric(
            train[column],
            errors="coerce"
        ).dropna()

        if not values.empty:

            return float(
                values.median()
            )

    return float(fallback)


# ============================================================
# NORMALIZE DATASET COLUMNS
# ============================================================

def normalize_columns(df):
    """
    Fix the known trailing-space issue in Distance.
    """

    df = df.copy()

    if (
        "Distance" in df.columns
        and DISTANCE_COL not in df.columns
    ):

        df = df.rename(
            columns={
                "Distance": DISTANCE_COL
            }
        )

    return df


# ============================================================
# BUILD SINGLE MODEL INPUT
# ============================================================

def build_single_input(values):
    """
    Build exactly the same feature structure expected
    by the trained model.
    """

    # ========================================================
    # ONE-HOT MODEL
    # ========================================================

    if ONE_HOT_MODEL:

        # Start with every model feature at 0
        row = {
            feature: 0
            for feature in expected_features
        }


        # ----------------------------------------------------
        # Numeric features
        # ----------------------------------------------------

        for column in NUMERIC_COLUMNS:

            if (
                column in expected_features
                and column in values
            ):

                row[column] = values[column]


        # ----------------------------------------------------
        # Categorical features
        # ----------------------------------------------------

        for column in CATEGORICAL_COLUMNS:

            if column not in values:
                continue

            category = str(
                values[column]
            )

            dummy_column = (
                f"{column}_{category}"
            )


            if dummy_column in row:

                row[dummy_column] = 1


        return pd.DataFrame(
            [row],
            columns=expected_features
        )


    # ========================================================
    # RAW MODEL
    # ========================================================

    row = {
        feature: np.nan
        for feature in expected_features
    }


    # Numeric values
    for column in NUMERIC_COLUMNS:

        if (
            column in row
            and column in values
        ):

            row[column] = values[column]


    # Categorical values
    for column in CATEGORICAL_COLUMNS:

        if (
            column in row
            and column in values
        ):

            row[column] = values[column]


    # Your model expects ID
    if "ID" in row:

        row["ID"] = values.get(
            "ID",
            0
        )


    return pd.DataFrame(
        [row],
        columns=expected_features
    )


# ============================================================
# PREPARE BATCH INPUT
# ============================================================

def prepare_batch_for_model(df):
    """
    Convert uploaded CSV to the exact feature format
    expected by the trained model.
    """

    df = normalize_columns(df)


    # Remove target if it exists
    if TARGET in df.columns:

        df = df.drop(
            columns=[TARGET]
        )


    # ========================================================
    # ONE-HOT MODEL
    # ========================================================

    if ONE_HOT_MODEL:

        output = pd.DataFrame(
            0,
            index=df.index,
            columns=expected_features
        )


        # ----------------------------------------------------
        # Numeric columns
        # ----------------------------------------------------

        for column in NUMERIC_COLUMNS:

            if (
                column in df.columns
                and column in output.columns
            ):

                output[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                ).fillna(0)


        # ----------------------------------------------------
        # Categorical columns
        # ----------------------------------------------------

        for column in CATEGORICAL_COLUMNS:

            if column not in df.columns:
                continue


            for index, value in df[
                column
            ].items():

                dummy_column = (
                    f"{column}_{value}"
                )


                if dummy_column in output.columns:

                    output.loc[
                        index,
                        dummy_column
                    ] = 1


        return output


    # ========================================================
    # RAW MODEL
    # ========================================================

    output = pd.DataFrame(
        index=df.index
    )


    for feature in expected_features:

        if feature in df.columns:

            output[feature] = df[
                feature
            ]

        elif feature == "ID":

            output[feature] = 0

        else:

            output[feature] = np.nan


    return output[
        expected_features
    ]


# ============================================================
# TITLE
# ============================================================

st.title(
    "🚗 Indian Pre-Owned Car Price Prediction"
)

st.write(
    "Enter the car details to estimate its market price."
)


# ============================================================
# INFORMATION MESSAGE
# ============================================================

if train is None:

    st.info(
        "ℹ️ Training CSV is not available in the deployed "
        "repository. The app is using the trained model "
        "to populate the input options."
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🎯 Price Prediction",
        "📁 Batch Prediction",
        "📊 Insights"
    ]
)


# ============================================================
# TAB 1 - PRICE PREDICTION
# ============================================================

with tab1:

    st.subheader(
        "Car Details"
    )


    # --------------------------------------------------------
    # DROPDOWN OPTIONS
    # --------------------------------------------------------

    maker_options = get_categories(
        "Maker"
    )

    model_options = get_categories(
        "model"
    )

    location_options = get_categories(
        "Location"
    )

    owner_options = get_categories(
        "Owner Type"
    )

    fuel_options = get_categories(
        "fuel_type"
    )

    transmission_options = get_categories(
        "transmission"
    )

    body_options = get_categories(
        "body_type"
    )


    # ========================================================
    # ROW 1
    # ========================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        maker = st.selectbox(
            "Maker",
            maker_options
            if maker_options
            else ["Unknown"]
        )


    with col2:

        # When training data is available,
        # filter models by maker.

        if (
            train is not None
            and "Maker" in train.columns
            and "model" in train.columns
        ):

            filtered_models = sorted(
                train.loc[
                    train["Maker"].astype(str)
                    == maker,
                    "model"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

        else:

            filtered_models = model_options


        model_name = st.selectbox(
            "Model",
            filtered_models
            if filtered_models
            else ["Unknown"]
        )


    with col3:

        location = st.selectbox(
            "Location",
            location_options
            if location_options
            else ["Unknown"]
        )


    # ========================================================
    # ROW 2
    # ========================================================

    col4, col5, col6 = st.columns(3)


    with col4:

        owner_type = st.selectbox(
            "Owner Type",
            owner_options
            if owner_options
            else ["Unknown"]
        )


    with col5:

        fuel_type = st.selectbox(
            "Fuel Type",
            fuel_options
            if fuel_options
            else ["Unknown"]
        )


    with col6:

        transmission = st.selectbox(
            "Transmission",
            transmission_options
            if transmission_options
            else ["Unknown"]
        )


    # ========================================================
    # ROW 3
    # ========================================================

    col7, col8, col9 = st.columns(3)


    with col7:

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
            value=get_numeric_default(
                DISTANCE_COL,
                65000.0
            ),
            step=1000.0
        )


    with col9:

        age = st.number_input(
            "Age of car",
            min_value=0,
            max_value=100,
            value=int(
                get_numeric_default(
                    "Age of car",
                    5
                )
            ),
            step=1
        )


    # ========================================================
    # ROW 4
    # ========================================================

    col10, col11, col12 = st.columns(3)


    with col10:

        manufacture_year = st.number_input(
            "Manufacture year",
            min_value=1900,
            max_value=2100,
            value=int(
                get_numeric_default(
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
            value=get_numeric_default(
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
            value=get_numeric_default(
                "engine_power",
                91.0
            ),
            step=1.0
        )


    # ========================================================
    # ROW 5
    # ========================================================

    col13, col14, col15 = st.columns(3)


    with col13:

        vroom_rating = st.number_input(
            "Vroom Audit Rating",
            min_value=0,
            max_value=20,
            value=int(
                get_numeric_default(
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
                get_numeric_default(
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
                get_numeric_default(
                    "seat_count",
                    5
                )
            ),
            step=1
        )


    st.divider()


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    if st.button(
        "🔮 Predict Price",
        type="primary",
        use_container_width=True
    ):

        values = {

            "ID": 0,

            "Maker":
                maker,

            "model":
                model_name,

            "Location":
                location,

            DISTANCE_COL:
                distance,

            "Owner Type":
                owner_type,

            "manufacture_year":
                manufacture_year,

            "Age of car":
                age,

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

            # Build the exact model input
            model_input = build_single_input(
                values
            )


            # Predict
            prediction = float(
                model.predict(
                    model_input
                )[0]
            )


            # ------------------------------------------------
            # VALUATION RANGE
            # ------------------------------------------------

            lower = prediction * 0.95
            upper = prediction * 1.05


            # ------------------------------------------------
            # DISPLAY RESULTS
            # ------------------------------------------------

            st.success(
                "✅ Prediction completed!"
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


            # Selected car
            st.subheader(
                "Selected Car"
            )


            a, b, c = st.columns(3)


            a.write(
                f"**Maker:** {maker}"
            )


            b.write(
                f"**Model:** {model_name}"
            )


            c.write(
                f"**Location:** {location}"
            )


        except Exception as e:

            st.error(
                f"❌ Prediction failed: {e}"
            )


            with st.expander(
                "Technical Details"
            ):

                st.write(
                    "The model expects:"
                )

                st.code(
                    "\n".join(
                        expected_features
                    )
                )


# ============================================================
# TAB 2 - BATCH PREDICTION
# ============================================================

with tab2:

    st.subheader(
        "📁 Batch Prediction"
    )


    st.write(
        "Upload your test CSV to generate predictions "
        "for multiple cars."
    )


    uploaded_file = st.file_uploader(
        "Upload Test CSV",
        type=["csv"]
    )


    if uploaded_file is not None:

        try:

            # Read uploaded CSV
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


            # Convert to model format
            model_input = (
                prepare_batch_for_model(
                    batch
                )
            )


            # Predict
            predictions = (
                model.predict(
                    model_input
                )
            )


            # Create result
            result = batch.copy()

            result[
                TARGET
            ] = predictions


            # Submission format
            if "ID" in result.columns:

                result = result[
                    [
                        "ID",
                        TARGET
                    ]
                ]


            st.success(
                f"✅ Predicted {len(result):,} rows."
            )


            st.dataframe(
                result.head(20),
                use_container_width=True
            )


            # Download CSV
            csv_data = (
                result
                .to_csv(index=False)
                .encode("utf-8")
            )


            st.download_button(
                "⬇️ Download Predictions",
                data=csv_data,
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True
            )


        except Exception as e:

            st.error(
                f"❌ Batch prediction failed: {e}"
            )


# ============================================================
# TAB 3 - INSIGHTS
# ============================================================

with tab3:

    st.subheader(
        "📊 Insights Dashboard"
    )


    # ========================================================
    # TRAINING DATA AVAILABLE
    # ========================================================

    if train is not None:


        # ----------------------------------------------------
        # Average Price by Car Age
        # ----------------------------------------------------

        if (
            "Age of car" in train.columns
            and TARGET in train.columns
        ):

            age_price = (
                train
                .groupby(
                    "Age of car"
                )[TARGET]
                .mean()
                .sort_index()
            )


            st.write(
                "### Average Price by Car Age"
            )


            st.line_chart(
                age_price
            )


        # ----------------------------------------------------
        # Average Price by Maker
        # ----------------------------------------------------

        if (
            "Maker" in train.columns
            and TARGET in train.columns
        ):

            maker_price = (
                train
                .groupby(
                    "Maker"
                )[TARGET]
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


        # ----------------------------------------------------
        # Price vs Distance
        # ----------------------------------------------------

        if (
            DISTANCE_COL in train.columns
            and TARGET in train.columns
        ):

            st.write(
                "### Price vs Distance"
            )


            distance_data = (
                train[
                    [
                        DISTANCE_COL,
                        TARGET
                    ]
                ]
                .dropna()
                .sort_values(
                    DISTANCE_COL
                )
            )


            # Downsample for performance
            if len(distance_data) > 5000:

                step = max(
                    1,
                    len(distance_data) // 5000
                )

                distance_data = (
                    distance_data.iloc[
                        ::step
                    ]
                )


            st.line_chart(
                distance_data.set_index(
                    DISTANCE_COL
                )[TARGET]
            )


        # ----------------------------------------------------
        # Training Data
        # ----------------------------------------------------

        st.write(
            "### Training Data Preview"
        )


        st.dataframe(
            train.head(20),
            use_container_width=True
        )


    # ========================================================
    # TRAINING DATA NOT AVAILABLE
    # ========================================================

    else:

        st.info(
            "The full training CSV is not available in the "
            "deployed GitHub repository."
        )


        st.write(
            "### Model Information"
        )


        c1, c2, c3 = st.columns(3)


        c1.metric(
            "Model Features",
            len(expected_features)
        )


        c2.metric(
            "Model Type",
            type(model).__name__
        )


        c3.metric(
            "One-Hot Encoding",
            "Yes"
            if ONE_HOT_MODEL
            else "No"
        )


        with st.expander(
            "Show Expected Model Features"
        ):

            st.code(
                "\n".join(
                    expected_features
                )
            )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.divider()

st.sidebar.title(
    "About"
)

st.sidebar.write(
    "🚗 Indian Pre-Owned Car Price Prediction"
)

st.sidebar.write(
    "Machine Learning + Streamlit"
)

st.sidebar.write(
    "🎯 Single Prediction"
)

st.sidebar.write(
    "📁 Batch Prediction"
)

st.sidebar.write(
    "📊 Insights Dashboard"
)

st.sidebar.caption(
    "Dynamic valuation range: ±5%"
)
