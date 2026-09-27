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

# ---------------------------------------------------------
# GitHub / Streamlit Cloud file layout
# app.py, model, and the optional training CSV are in the
# repository root.
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "car_price_model_small.pkl"
ALTERNATE_MODEL_PATH = ROOT / "car_price_model.pkl"
TRAIN_PATH = ROOT / "Cap_Training_Data_2025.csv"

TARGET = "Price"
DISTANCE_COL = "Distance "


@st.cache_resource
def load_model():
    """Load the model stored in the GitHub repository root."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    if ALTERNATE_MODEL_PATH.exists():
        return joblib.load(ALTERNATE_MODEL_PATH)

    return None


@st.cache_data
def load_training_data():
    """Load training data only when a real CSV is available."""
    if not TRAIN_PATH.exists():
        return None

    try:
        if TRAIN_PATH.stat().st_size < 100:
            return None

        df = pd.read_csv(TRAIN_PATH)

        if df.empty or len(df.columns) == 0 or TARGET not in df.columns:
            return None

        return df

    except Exception:
        return None


def get_model_categories(model_obj, base_name):
    """
    Recover one-hot categories directly from model.feature_names_in_.
    This lets the app run even when the large training CSV is not stored
    in GitHub.
    """
    if not hasattr(model_obj, "feature_names_in_"):
        return []

    prefix = f"{base_name}_"
    return sorted(
        {
            str(feature)[len(prefix):]
            for feature in model_obj.feature_names_in_
            if str(feature).startswith(prefix)
        }
    )


def get_categories(train_df, model_obj, column):
    """Use training CSV categories when available; otherwise use model names."""
    if train_df is not None and column in train_df.columns:
        values = (
            train_df[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        if values:
            return sorted(values)

    return get_model_categories(model_obj, column)


def get_numeric_default(train_df, column, fallback):
    """Return a robust numeric default."""
    if train_df is not None and column in train_df.columns:
        values = pd.to_numeric(
            train_df[column], errors="coerce"
        ).dropna()
        if not values.empty:
            return float(values.median())
    return float(fallback)


def build_input_row(train_df, values):
    feature_cols = [c for c in train_df.columns if c != TARGET]
    row = {c: np.nan for c in feature_cols}

    if "ID" in row:
        row["ID"] = int(train_df["ID"].median())

    for key, value in values.items():
        if key in row:
            row[key] = value

    return pd.DataFrame([row], columns=feature_cols)


def prepare_for_model(df, model):
    """
    Match the input data to the feature representation used by the
    saved model. This handles models trained after pd.get_dummies().
    """
    df = df.copy()

    # Dataset has a trailing space in this column name.
    if "Distance" in df.columns and DISTANCE_COL not in df.columns:
        df = df.rename(columns={"Distance": DISTANCE_COL})

    if TARGET in df.columns:
        df = df.drop(columns=[TARGET])

    categorical_cols = [
        "Maker",
        "model",
        "Location",
        "Owner Type",
        "body_type",
        "transmission",
        "fuel_type"
    ]

    categorical_cols = [c for c in categorical_cols if c in df.columns]

    # If the saved model expects raw columns, don't one-hot encode.
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        expected_set = set(expected)

        raw_present = any(c in expected_set for c in categorical_cols)

        # The error in this project shows the model expects one-hot columns
        # such as Location_Bangalore, so encode only when needed.
        one_hot_expected = any(
            any(c + "_" in expected_name for c in categorical_cols)
            for expected_name in expected
        )

        if one_hot_expected:
            df = pd.get_dummies(
                df,
                columns=categorical_cols,
                dtype=int
            )

        # Make exact feature names and order match the training model.
        df = df.reindex(columns=expected, fill_value=0)

        return df

    # Fallback for models without feature_names_in_
    return pd.get_dummies(
        df,
        columns=categorical_cols,
        dtype=int
    )


st.title("🚗 Indian Pre-Owned Car Price Prediction")
st.write("Enter the car details to estimate its market price.")

model = load_model()
train = load_training_data()

if model is None:
    st.error(
        "Model file not found. Make sure "
        "`car_price_model_small.pkl` is in the repository root."
    )
    st.stop()

if train is None:
    st.info(
        "The full training CSV is not available in the deployed repository. "
        "The app will use the trained model's feature names for dropdowns. "
        "Upload a CSV in Batch Prediction for bulk inference."
    )

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Car Details")

maker_options = get_categories(
    train, model, "Maker"
)

model_options = get_categories(
    train, model, "model"
)

location_options = get_categories(
    train, model, "Location"
)

owner_options = get_categories(
    train, model, "Owner Type"
)

fuel_options = get_categories(
    train, model, "fuel_type"
)

transmission_options = get_categories(
    train, model, "transmission"
)

body_options = get_categories(
    train, model, "body_type"
)

maker = st.sidebar.selectbox(
    "Maker",
    maker_options if maker_options else ["Unknown"]
)

model_name = st.sidebar.selectbox(
    "Model",
    model_options if model_options else ["Unknown"]
)

location = st.sidebar.selectbox(
    "Location",
    location_options if location_options else ["Unknown"]
)

owner_type = st.sidebar.selectbox(
    "Owner Type",
    owner_options if owner_options else ["Unknown"]
)

fuel_type = st.sidebar.selectbox(
    "Fuel Type",
    fuel_options if fuel_options else ["Unknown"]
)

transmission = st.sidebar.selectbox(
    "Transmission",
    transmission_options if transmission_options else ["Unknown"]
)

body_type = st.sidebar.selectbox(
    "Body Type",
    body_options if body_options else ["Unknown"]
)

distance = st.sidebar.number_input(
    "Distance driven (km)",
    min_value=0.0,
    max_value=20_000_000.0,
    value=get_numeric_default(train, DISTANCE_COL, 65000.0),
    step=1000.0
)

age = st.sidebar.number_input(
    "Age of car",
    min_value=0,
    max_value=100,
    value=int(get_numeric_default(train, "Age of car", 5)),
    step=1
)

manufacture_year = st.sidebar.number_input(
    "Manufacture year",
    min_value=1900,
    max_value=2100,
    value=int(get_numeric_default(train, "manufacture_year", 2020)),
    step=1
)

engine_displacement = st.sidebar.number_input(
    "Engine displacement",
    min_value=0.0,
    max_value=100000.0,
    value=get_numeric_default(train, "engine_displacement", 1896.0),
    step=50.0
)

engine_power = st.sidebar.number_input(
    "Engine power",
    min_value=0.0,
    max_value=2000.0,
    value=get_numeric_default(train, "engine_power", 91.0),
    step=1.0
)

vroom_default = int(
    get_numeric_default(train, "Vroom Audit Rating", 6)
)

vroom_rating = st.sidebar.number_input(
    "Vroom Audit Rating",
    min_value=0,
    max_value=20,
    value=vroom_default,
    step=1
)

door_count = st.sidebar.number_input(
    "Door count",
    min_value=1,
    max_value=10,
    value=int(get_numeric_default(train, "door_count", 4)),
    step=1
)

seat_count = st.sidebar.number_input(
    "Seat count",
    min_value=1,
    max_value=20,
    value=int(get_numeric_default(train, "seat_count", 5)),
    step=1
)

values = {
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

tab1, tab2, tab3 = st.tabs(
    ["🎯 Price Prediction", "📁 Batch Prediction", "📊 Insights"]
)

with tab1:
    st.subheader("Selected Car")

    a, b, c = st.columns(3)
    a.metric("Maker", maker.title())
    b.metric("Model", model_name.title())
    c.metric("Location", location)

    if st.button(
        "🔮 Predict Price",
        type="primary",
        use_container_width=True
    ):
        if model is None:
            st.error("Model file not found.")
        else:
            try:
                input_df = build_input_row(train, values)
                model_input = prepare_for_model(input_df, model)

                prediction = float(
                    model.predict(model_input)[0]
                )

                low = prediction * 0.95
                high = prediction * 1.05

                st.success("Prediction completed!")

                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Estimated Price",
                    f"₹{prediction:,.0f}"
                )
                c2.metric(
                    "5% Lower Range",
                    f"₹{low:,.0f}"
                )
                c3.metric(
                    "5% Upper Range",
                    f"₹{high:,.0f}"
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")

                if hasattr(model, "feature_names_in_"):
                    st.write(
                        "Model expects these features:"
                    )
                    st.code(
                        "\n".join(
                            list(model.feature_names_in_)
                        )
                    )

with tab2:
    st.subheader("Batch Prediction")

    uploaded_file = st.file_uploader(
        "Upload test CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            batch = pd.read_csv(uploaded_file)

            st.dataframe(
                batch.head(),
                use_container_width=True
            )

            if model is None:
                st.error("Model file not found.")
            else:
                model_input = prepare_for_model(
                    batch,
                    model
                )

                predictions = model.predict(
                    model_input
                )

                output = batch.copy()
                output[TARGET] = predictions

                if "ID" in output.columns:
                    output = output[["ID", TARGET]]

                st.success(
                    f"Predicted {len(output)} rows."
                )

                st.dataframe(
                    output.head(20),
                    use_container_width=True
                )

                st.download_button(
                    "⬇️ Download Predictions",
                    output.to_csv(index=False).encode("utf-8"),
                    file_name="car_price_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

with tab3:
    st.subheader("Dataset Insights")

    if train is not None:

        age_price = (
            train.groupby("Age of car")["Price"]
            .mean()
            .sort_index()
        )

        maker_price = (
            train.groupby("Maker")["Price"]
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

        if DISTANCE_COL in train.columns:
            st.write("Price vs Distance")

            distance_data = train[
                [DISTANCE_COL, "Price"]
            ].dropna().sort_values(DISTANCE_COL)

            if len(distance_data) > 5000:
                step = max(1, len(distance_data) // 5000)
                distance_data = distance_data.iloc[::step]

            st.line_chart(
                distance_data.set_index(DISTANCE_COL)["Price"]
            )

        st.write("Training Data")
        st.dataframe(
            train.head(20),
            use_container_width=True
        )

    else:
        st.info(
            "Training CSV is not included in the deployed GitHub repository. "
            "Prediction and batch prediction use the trained model directly."
        )

        st.metric(
            "Model Features",
            len(expected_features)
        )

        with st.expander("Show model features"):
            st.code("\n".join(expected_features))

st.sidebar.divider()
st.sidebar.caption(
    "Indian Pre-Owned Car Price Prediction"
)
