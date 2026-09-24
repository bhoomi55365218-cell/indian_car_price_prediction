# Indian Pre-Owned Car Price Prediction

## Project Name
**Indian Pre-Owned Car Price Prediction**

This project is an end-to-end Machine Learning application that predicts the price of a pre-owned car using historical Indian car-market data. The machine-learning model is deployed through an interactive Streamlit web application.

---

## Project Folder

The current project folder used in Google Colab is:

```text
/content/Indian-Car-Price-Prediction/
```

### Current Project Structure

```text
/content/Indian-Car-Price-Prediction/
│
├── app/
│   ├── app.py
│   └── app_backup.py
│
├── models/
│   └── car_price_model.pkl
│
├── data/
│   └── Cap_Training_Data_2025.csv
│
├── outputs/
│   └── predictions.csv
│
├── requirements.txt
└── README.md
```

The test dataset currently used from Colab is:

```text
/content/Cap_Test_Data_2025.csv
```

The sample submission file currently used from Colab is:

```text
/content/Cap_Sample_Submission_2025 (1).csv
```

The ML notebook currently used in Colab is:

```text
car price prediction.ipynb
```

---

# Component 1A - Machine Learning Notebook

The notebook contains the machine-learning workflow for the project.

### Main stages

1. Import Python libraries
2. Load training and test datasets
3. Understand the dataset
4. Perform Exploratory Data Analysis (EDA)
5. Analyze missing values
6. Check duplicates and outliers
7. Clean the data
8. Perform feature engineering
9. Preprocess numerical and categorical features
10. Train regression models
11. Evaluate models
12. Select/save the final model
13. Predict prices for the test dataset
14. Generate the submission file

### Target Variable

```text
Price
```

### Important Input Features

```text
ID
Maker
model
Location
Distance
Owner Type
manufacture_year
Age of car
engine_displacement
engine_power
body_type
Vroom Audit Rating
transmission
door_count
seat_count
fuel_type
```

---

# Component 1B - Streamlit Application

The Streamlit source code is:

```text
/content/Indian-Car-Price-Prediction/app/app.py
```

The trained model used by the app is:

```text
/content/Indian-Car-Price-Prediction/models/car_price_model.pkl
```

## Current Streamlit Features

### 1. Price Prediction

The user can enter/select:

- Maker
- Model
- Location
- Distance driven
- Owner Type
- Manufacture year
- Age of car
- Engine displacement
- Engine power
- Body Type
- Vroom Audit Rating
- Transmission
- Door count
- Seat count
- Fuel type

The app displays:

```text
Estimated Price
5% Lower Range
5% Upper Range
```

### 2. Batch Prediction

The app accepts a CSV upload through the **Batch Prediction** tab.

The test dataset used during development is:

```text
/content/Cap_Test_Data_2025.csv
```

The app predicts all rows and allows the user to download the results.

The generated prediction file is:

```text
/content/predictions.csv
```

The copy kept in the final project folder is:

```text
/content/Indian-Car-Price-Prediction/outputs/predictions.csv
```

The expected submission columns are:

```text
ID,Price
```

### 3. Insights Dashboard

The **Insights** tab currently provides:

- Average Price by Car Age
- Average Price by Maker
- Training-data preview

---

# Requirements

The project uses the following Python packages:

```text
streamlit
pandas
numpy
joblib
scikit-learn
xgboost
lightgbm
```

They are listed in:

```text
/content/Indian-Car-Price-Prediction/requirements.txt
```

---

# How to Run the Streamlit App Locally

Open a terminal in the project root and run:

```bash
pip install -r requirements.txt
```

Then start Streamlit:

```bash
streamlit run app/app.py
```

Streamlit will display a local URL. Open that URL in a browser.

---

# How to Run the App in Google Colab

## Step 1 - Mount Google Drive (optional but recommended)

```python
from google.colab import drive
drive.mount('/content/drive')
```

The project can be stored permanently in:

```text
/content/drive/MyDrive/Indian-Car-Price-Prediction/
```

## Step 2 - Install packages

```python
!pip install -q streamlit pyngrok joblib pandas numpy scikit-learn xgboost lightgbm
```

## Step 3 - Start Streamlit

```python
import subprocess

subprocess.Popen(
    [
        "streamlit",
        "run",
        "/content/Indian-Car-Price-Prediction/app/app.py",
        "--server.port",
        "8501",
        "--server.address",
        "0.0.0.0",
    ],
    stdout=open("/content/streamlit.log", "w"),
    stderr=subprocess.STDOUT,
)

print("Streamlit started on port 8501")
```

## Step 4 - Create the public ngrok link

```python
from pyngrok import ngrok

ngrok.kill()
public_url = ngrok.connect(8501)

print("Streamlit App URL:", public_url)
```

Open the generated HTTPS URL in a browser.

### Current development/demo URL

```text
https://arise-krypton-gangly.ngrok-free.dev
```

**Important:** This is a temporary Colab/ngrok URL. It only works while the corresponding Colab runtime and ngrok tunnel are active. A new URL may be generated after the runtime or tunnel is restarted.

---

# Model File

The Streamlit application expects the trained model at:

```text
/content/Indian-Car-Price-Prediction/models/car_price_model.pkl
```

The model must be compatible with the preprocessing and feature representation used during training.

---

# Output File

The final batch prediction output is:

```text
/content/Indian-Car-Price-Prediction/outputs/predictions.csv
```

Expected format:

```text
ID,Price
211000001,<predicted price>
211000002,<predicted price>
211000003,<predicted price>
...
```

---

# Recommended Submission Structure

## Component 1A

```text
car price prediction.ipynb
```

## Component 1B

```text
Component_1B/
│
├── app.py
├── car_price_model.pkl
├── requirements.txt
└── README.md
```

## Output

```text
predictions.csv
```

---

# Submission Checklist

```text
[ ] Machine Learning notebook (.ipynb)
[ ] Streamlit source code (app.py)
[ ] Trained model (car_price_model.pkl)
[ ] requirements.txt
[ ] README.md
[ ] predictions.csv
[ ] Working app/demo link or run instructions
```

---

## Notes

- Keep the folder names and relative paths unchanged when running the app.
- The Streamlit app reads the training CSV to populate dropdown options and generate the insights dashboard.
- The trained model is loaded from the `models` folder.
- The batch prediction output is written to the `outputs` folder when the project copy is maintained.
- For long-term access, keep the project files in Google Drive and use a persistent deployment platform for a permanent web URL instead of relying on a temporary Colab/ngrok session.
