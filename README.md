# Indian Pre-Owned Car Price Prediction — Streamlit App

## Folder structure

```text
Indian-Car-Price-Prediction/
├── app/
│   └── app.py
├── data/
│   └── Cap_Training_Data_2025.csv
├── models/
│   └── car_price_pipeline.pkl
└── requirements.txt
```

## Important

The model file should be a **single saved preprocessing + ML pipeline** named:

`models/car_price_pipeline.pkl`

It should accept the 15 raw feature columns used by the app and return predicted `Price` values.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Run in Google Colab

Install packages:

```python
!pip install -q streamlit pyngrok joblib scikit-learn xgboost lightgbm
```

Then start the app with your preferred public-tunnel method.
