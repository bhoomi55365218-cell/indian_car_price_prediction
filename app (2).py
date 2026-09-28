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

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "car_price_model.pkl"
PIPELINE_PATH = ROOT / "models" / "car_price_pipeline.pkl"
TRAIN_PATH = ROOT / "data" / "Cap_Training_Data_2025.csv"

TARGET = "Price"
DISTANCE_COL = "Distance "


@st.cache_resource
def load_model():
    # Prefer the pipeline if one exists; otherwise use the existing model.
    if PIPELINE_PATH.exists():
        return joblib.load(PIPELINE_PATH)
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data
def load_training_data():
    if not TRAIN_PATH.exists():
        return None
    return pd.read_csv(TRAIN_PATH)


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

if train is None:
    st.error("Training CSV not found.")
    st.code("data/Cap_Training_Data_2025.csv")
    st.stop()

if model is None:
    st.warning(
        "Model file not found. Put car_price_model.pkl or "
        "car_price_pipeline.pkl inside the models folder."
    )

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Car Details")

maker = st.sidebar.selectbox(
    "Maker",
    sorted(train["Maker"].dropna().astype(str).unique())
)

maker_models = sorted(
    train.loc[
        train["Maker"].astype(str) == maker,
        "model"
    ].dropna().astype(str).unique()
)

model_name = st.sidebar.selectbox("Model", maker_models)

location = st.sidebar.selectbox(
    "Location",
    sorted(train["Location"].dropna().astype(str).unique())
)

owner_type = st.sidebar.selectbox(
    "Owner Type",
    sorted(train["Owner Type"].dropna().astype(str).unique())
)

fuel_type = st.sidebar.selectbox(
    "Fuel Type",
    sorted(train["fuel_type"].dropna().astype(str).unique())
)

transmission = st.sidebar.selectbox(
    "Transmission",
    sorted(train["transmission"].dropna().astype(str).unique())
)

body_type = st.sidebar.selectbox(
    "Body Type",
    sorted(train["body_type"].dropna().astype(str).unique())
)

distance_default = float(
    train[DISTANCE_COL].median()
    if DISTANCE_COL in train.columns
    else train["Distance"].median()
)

distance = st.sidebar.number_input(
    "Distance driven (km)",
    min_value=0.0,
    max_value=20_000_000.0,
    value=distance_default,
    step=1000.0
)

age = st.sidebar.number_input(
    "Age of car",
    min_value=0,
    max_value=100,
    value=int(train["Age of car"].median()),
    step=1
)

manufacture_year = st.sidebar.number_input(
    "Manufacture year",
    min_value=1900,
    max_value=2100,
    value=int(train["manufacture_year"].median()),
    step=1
)

engine_displacement = st.sidebar.number_input(
    "Engine displacement",
    min_value=0,
    max_value=100000,
    value=int(train["engine_displacement"].median()),
    step=50
)

engine_power = st.sidebar.number_input(
    "Engine power",
    min_value=0.0,
    max_value=2000.0,
    value=float(train["engine_power"].median()),
    step=1.0
)

vroom_rating = st.sidebar.number_input(
    "Vroom Audit Rating",
    min_value=int(train["Vroom Audit Rating"].min()),
    max_value=int(train["Vroom Audit Rating"].max()),
    value=int(train["Vroom Audit Rating"].median()),
    step=1
)

door_count = st.sidebar.number_input(
    "Door count",
    min_value=1,
    max_value=10,
    value=int(train["door_count"].median()),
    step=1
)

seat_count = st.sidebar.number_input(
    "Seat count",
    min_value=1,
    max_value=20,
    value=int(train["seat_count"].median()),
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

    st.write("Training Data")
    st.dataframe(
        train.head(20),
        use_container_width=True
    )

st.sidebar.divider()
st.sidebar.caption(
    "Indian Pre-Owned Car Price Prediction"
)
