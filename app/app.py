import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Indian Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# ============================================================
# GITHUB / STREAMLIT CLOUD PATHS
# ============================================================
# This app is designed for the following repository structure:
#
# indian_pre_owned_car_price_prediction/
# ├── app.py
# ├── car_price_model.pkl
# ├── Cap_Training_Data_2025.csv
# ├── predictions.csv
# ├── requirements.txt
# └── README.md

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "car_price_model.pkl"
TRAIN_PATH = ROOT / "Cap_Training_Data_2025.csv"

TARGET = "Price"
DISTANCE_COL = "Distance "

CATEGORICAL_COLUMNS = [
    "Maker",
    "model",
    "Location",
    "Owner Type",
    "body_type",
    "transmission",
    "fuel_type"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    """Load the trained Random Forest model."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

@st.cache_data
def load_training_data():
    """Load the training CSV used to reproduce the training features."""
    if not TRAIN_PATH.exists():
        return None

    try:
        if TRAIN_PATH.stat().st_size < 100:
            return None

        df = pd.read_csv(TRAIN_PATH)
        if df.empty or TARGET not in df.columns:
            return None

        return df
    except Exception:
        return None


model = load_model()
train = load_training_data()

if model is None:
    st.error(
        "Model file not found. Make sure `car_price_model.pkl` "
        "is in the GitHub repository root."
    )
    st.stop()

if train is None:
    st.error(
        "Training CSV not found or is empty. Make sure the real "
        "`Cap_Training_Data_2025.csv` is in the GitHub repository root."
    )
    st.stop()


# ============================================================
# RECREATE THE EXACT TRAINING FEATURE ORDER
# ============================================================

X_train_raw = train.drop(columns=[TARGET]).copy()

# The original dataset contains a trailing space in this column name.
if "Distance" in X_train_raw.columns and DISTANCE_COL not in X_train_raw.columns:
    X_train_raw = X_train_raw.rename(columns={"Distance": DISTANCE_COL})

categorical_for_training = [
    c for c in CATEGORICAL_COLUMNS
    if c in X_train_raw.columns
]

TRAIN_FEATURES = pd.get_dummies(
    X_train_raw,
    columns=categorical_for_training,
    dtype=int
).columns.tolist()


# The RandomForest was trained on the one-hot encoded training matrix.
# It may not expose feature_names_in_, so n_features_in_ is used here.
model_feature_count = getattr(model, "n_features_in_", None)

if (
    model_feature_count is not None
    and len(TRAIN_FEATURES) != model_feature_count
):
    st.error(
        "Model/data mismatch: the model expects "
        f"{model_feature_count} input features, but the uploaded "
        f"training data creates {len(TRAIN_FEATURES)} features."
    )
    st.stop()


# ============================================================
# PREPARE DATA FOR THE MODEL
# ============================================================

def prepare_for_model(df):
    """Apply the same one-hot encoding and feature order used in training."""

    df = df.copy()

    # Handle Distance vs Distance(space)
    if "Distance" in df.columns and DISTANCE_COL not in df.columns:
        df = df.rename(columns={"Distance": DISTANCE_COL})

    # Remove target if present
    if TARGET in df.columns:
        df = df.drop(columns=[TARGET])

    categorical_cols = [
        c for c in CATEGORICAL_COLUMNS
        if c in df.columns
    ]

    encoded = pd.get_dummies(
        df,
        columns=categorical_cols,
        dtype=int
    )

    # Exact same columns and order as training
    encoded = encoded.reindex(
        columns=TRAIN_FEATURES,
        fill_value=0
    )

    # Convert everything to numeric
    for column in encoded.columns:
        encoded[column] = pd.to_numeric(
            encoded[column],
            errors="coerce"
        ).fillna(0)

    return encoded


# ============================================================
# NUMERIC DEFAULTS
# ============================================================

def get_default(column, fallback):
    values = pd.to_numeric(
        train[column],
        errors="coerce"
    ).dropna()

    if values.empty:
        return float(fallback)

    return float(values.median())


# ============================================================
# APP HEADER
# ============================================================

st.title("🚗 Indian Pre-Owned Car Price Prediction")
st.write("Enter the car details to estimate its market price.")


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
# TAB 1 - SINGLE PRICE PREDICTION
# ============================================================

with tab1:
    st.subheader("Selected Car")

    # --------------------------------------------------------
    # Row 1
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    maker = st.selectbox(
        "Maker",
        sorted(
            train["Maker"].dropna().astype(str).unique()
        )
    )

    maker_models = sorted(
        train.loc[
            train["Maker"].astype(str) == maker,
            "model"
        ].dropna().astype(str).unique()
    )

    with col1:
        pass

    # Re-render in the intended columns.
    # Streamlit widgets created outside a column cannot be moved, so use
    # the values above only as an internal fallback if needed.
    # The actual visible widgets are created below with unique labels.

    # Reset is not possible, so use the already-created values for the first
    # row and place remaining widgets in columns.
    with col1:
        st.caption(f"Maker: {maker}")

    with col2:
        model_name = st.selectbox(
            "Model",
            maker_models if maker_models else ["Unknown"],
            key="model_select"
        )

    with col3:
        location = st.selectbox(
            "Location",
            sorted(train["Location"].dropna().astype(str).unique()),
            key="location_select"
        )

    # --------------------------------------------------------
    # Row 2
    # --------------------------------------------------------

    col4, col5, col6 = st.columns(3)

    with col4:
        owner_type = st.selectbox(
            "Owner Type",
            sorted(train["Owner Type"].dropna().astype(str).unique()),
            key="owner_select"
        )

    with col5:
        fuel_type = st.selectbox(
            "Fuel Type",
            sorted(train["fuel_type"].dropna().astype(str).unique()),
            key="fuel_select"
        )

    with col6:
        transmission = st.selectbox(
            "Transmission",
            sorted(train["transmission"].dropna().astype(str).unique()),
            key="transmission_select"
        )

    # --------------------------------------------------------
    # Row 3
    # --------------------------------------------------------

    col7, col8, col9 = st.columns(3)

    with col7:
        body_type = st.selectbox(
            "Body Type",
            sorted(train["body_type"].dropna().astype(str).unique()),
            key="body_select"
        )

    with col8:
        distance = st.number_input(
            "Distance driven (km)",
            min_value=0.0,
            max_value=20_000_000.0,
            value=get_default(DISTANCE_COL, 65000),
            step=1000.0,
            key="distance_input"
        )

    with col9:
        age = st.number_input(
            "Age of car (years)",
            min_value=0,
            max_value=100,
            value=int(get_default("Age of car", 5)),
            step=1,
            key="age_input"
        )

    # --------------------------------------------------------
    # Row 4
    # --------------------------------------------------------

    col10, col11, col12 = st.columns(3)

    with col10:
        manufacture_year = st.number_input(
            "Manufacture year",
            min_value=1900,
            max_value=2100,
            value=int(get_default("manufacture_year", 2020)),
            step=1,
            key="year_input"
        )

    with col11:
        engine_displacement = st.number_input(
            "Engine displacement",
            min_value=0.0,
            max_value=100000.0,
            value=get_default("engine_displacement", 1896),
            step=50.0,
            key="engine_input"
        )

    with col12:
        engine_power = st.number_input(
            "Engine power",
            min_value=0.0,
            max_value=2000.0,
            value=get_default("engine_power", 91),
            step=1.0,
            key="power_input"
        )

    # --------------------------------------------------------
    # Row 5
    # --------------------------------------------------------

    col13, col14, col15 = st.columns(3)

    with col13:
        vroom_rating = st.number_input(
            "Vroom Audit Rating",
            min_value=int(train["Vroom Audit Rating"].min()),
            max_value=int(train["Vroom Audit Rating"].max()),
            value=int(get_default("Vroom Audit Rating", 6)),
            step=1,
            key="vroom_input"
        )

    with col14:
        door_count = st.number_input(
            "Door Count",
            min_value=1,
            max_value=10,
            value=int(get_default("door_count", 4)),
            step=1,
            key="door_input"
        )

    with col15:
        seat_count = st.number_input(
            "Seat Count",
            min_value=1,
            max_value=20,
            value=int(get_default("seat_count", 5)),
            step=1,
            key="seat_input"
        )

    st.divider()

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Price",
        type="primary",
        use_container_width=True
    ):

        input_row = pd.DataFrame([
            {
                "ID": int(train["ID"].median()),
                "Maker": maker,
                "model": model_name,
                "Location": location,
                DISTANCE_COL: distance,
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
                "fuel_type": fuel_type,
            }
        ])

        try:
            model_input = prepare_for_model(input_row)
            prediction = float(model.predict(model_input)[0])

            lower = prediction * 0.95
            upper = prediction * 1.05

            st.success("✅ Prediction completed!")

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

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

            with st.expander("Technical Details"):
                st.write(
                    f"Model input contains {model_input.shape[1]} features."
                    if "model_input" in locals()
                    else "The model input could not be created."
                )
                st.write("Expected feature count:", len(TRAIN_FEATURES))


# ============================================================
# TAB 2 - BATCH PREDICTION
# ============================================================

with tab2:
    st.subheader("📁 Batch Prediction")
    st.write(
        "Upload `Cap_Test_Data_2025.csv` to generate prices for multiple cars."
    )

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

            model_input = prepare_for_model(batch)
            predictions = model.predict(model_input)

            output = batch.copy()
            output[TARGET] = predictions

            if "ID" in output.columns:
                output = output[["ID", TARGET]]

            st.success(
                f"✅ Predicted {len(output):,} rows."
            )

            st.dataframe(
                output.head(20),
                use_container_width=True
            )

            st.download_button(
                "⬇️ Download Predictions",
                output.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as exc:
            st.error(f"Batch prediction failed: {exc}")


# ============================================================
# TAB 3 - INSIGHTS
# ============================================================

with tab3:
    st.subheader("📊 Insights Dashboard")

    age_price = (
        train.groupby("Age of car")[TARGET]
        .mean()
        .sort_index()
    )

    maker_price = (
        train.groupby("Maker")[TARGET]
        .mean()
        .sort_values(ascending=False)
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write("Average Price by Car Age")
        st.line_chart(age_price)

    with col2:
        st.write("Average Price by Maker")
        st.bar_chart(maker_price)

    distance_data = (
        train[[DISTANCE_COL, TARGET]]
        .dropna()
        .sort_values(DISTANCE_COL)
    )

    if len(distance_data) > 5000:
        step = max(1, len(distance_data) // 5000)
        distance_data = distance_data.iloc[::step]

    st.write("Price vs Distance")
    st.line_chart(
        distance_data.set_index(DISTANCE_COL)[TARGET]
    )

    st.write("Training Data")
    st.dataframe(
        train.head(20),
        use_container_width=True
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.divider()
st.sidebar.title("About")
st.sidebar.write("🚗 Indian Pre-Owned Car Price Prediction")
st.sidebar.write("Machine Learning + Streamlit")
st.sidebar.write("🎯 Single Prediction")
st.sidebar.write("📁 Batch Prediction")
st.sidebar.write("📊 Insights Dashboard")
st.sidebar.caption("Dynamic valuation range: ±5%")
