# Indian Pre-Owned Car Price Prediction

## Project Overview

An end-to-end machine learning project that predicts the price of pre-owned cars in India and deploys the model through a Streamlit web application.

## GitHub Repository Structure

```text
indian_pre_owned_car_price_prediction/
├── app.py
├── car_price_model.pkl
├── Cap_Training_Data_2025.csv
├── predictions.csv
├── requirements.txt
├── README.md
└── indian_pre_owned_car_price_prediction.ipynb
```

## Streamlit App

### Price Prediction
The user enters/selects vehicle details including maker, model, location, distance, owner type, year, age, engine details, body type, audit rating, transmission, doors, seats, and fuel type. The app returns an estimated price and a dynamic ±5% valuation range.

### Batch Prediction
Upload `Cap_Test_Data_2025.csv` and download the generated predictions in `ID,Price` format.

### Insights
The app displays average price by car age, average price by maker, price versus distance, and a training-data preview.

## Model

`car_price_model.pkl` is a compact Random Forest model created from the trained Random Forest used in development. It contains 20 trees and is approximately 14.6 MB, making it suitable for normal GitHub browser upload.

The original model was trained using `pd.get_dummies()` on the training data. The deployed app reconstructs the same one-hot encoded feature columns and exact column order before prediction.

## Requirements

Install the app dependencies with:

```bash
pip install -r requirements.txt
```

The deployment requirements are pinned to the model's training environment versions for the main persistence-sensitive packages:

```text
Python 3.13
scikit-learn 1.6.1
numpy 2.3.1
scipy 1.16.3
joblib 1.6.0
```

## Run Locally

From the repository root:

```bash
streamlit run app.py
```

## Run in Google Colab

Install dependencies:

```python
!pip install -q -r /content/indian_pre_owned_car_price_prediction/requirements.txt
```

Start Streamlit in the background:

```python
import subprocess
import time

subprocess.Popen(
    [
        "streamlit",
        "run",
        "/content/indian_pre_owned_car_price_prediction/app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
    ],
    stdout=open("/content/streamlit.log", "w"),
    stderr=subprocess.STDOUT,
)

time.sleep(5)
print("Streamlit started")
```

For a temporary public Colab link:

```python
from pyngrok import ngrok

ngrok.kill()
public_url = ngrok.connect(8501)
print("App URL:", public_url)
```

## Streamlit Community Cloud

Deploy the repository with:

```text
Repository: bhoomi55365218-cell/indian_pre-owned_car_price_prediction
Branch: main
Main file path: app.py
```

Make sure `requirements.txt`, `app.py`, `car_price_model.pkl`, and the valid training CSV are in the repository root.

## Important About the Training CSV

The training CSV is required by this version of the app because the saved Random Forest does not expose `feature_names_in_`. The app rebuilds the exact training feature order from `Cap_Training_Data_2025.csv`.

The valid training file used for this project is approximately 5 MB. Do not replace it with an empty placeholder file.

## Output

The example batch output is:

```text
predictions.csv
```

Expected columns:

```text
ID,Price
```

## Notebook

Keep the existing machine-learning notebook unchanged. The notebook is a separate project deliverable and is not modified by this Streamlit deployment setup.

## Submission Checklist

- [ ] Machine learning notebook
- [ ] `app.py`
- [ ] `car_price_model.pkl`
- [ ] `requirements.txt`
- [ ] `README.md`
- [ ] `predictions.csv`
- [ ] Streamlit app link
