# Social Media Impact Application

This project analyzes the impact of social media usage on student mental health and academic performance. It includes a machine learning pipeline to predict addiction risk and mental health scores, wrapped in a modern Flask web application.

## 🚀 Features
-   **Addiction Risk Prediction**: Classifies users as "High Risk" or "Low Risk" (98% Accuracy).
-   **Mental Health Score Prediction**: Estimates a mental health score (1-10) based on usage habits (R² ~0.95).
-   **Visual Comparison**: Compare your mental health score against the population distribution.
-   **Interactive UI**: Clean, glassmorphism-inspired interface.

## 📦 Project Structure
```
/ASIF
├── app.py                # Flask Backend (Server)
├── train_models.py       # Script to train ML models and save artifacts
├── models/               # Saved models (.joblib files)
├── static/               # CSS and Assets
├── templates/            # HTML Templates (Frontend)
├── requirements.txt      # Python Dependencies
└── Students Social Media Addiction.csv # Dataset
```

## 🛠️ Installation & Setup

> **Model compatibility note**
> The pre-trained models were created with scikit-learn **1.8.0**. If your
> runtime environment uses an older version (e.g. 1.7.x), you may see warnings
> during startup or errors such as ``'LogisticRegression' object has no
> attribute 'multi_class'`` when making predictions. You can resolve this by
> either:
>
> 1. Upgrading scikit-learn to the same version: ``pip install scikit-learn>=1.8``
> 2. Retraining the models using the bundled ``train_models.py`` script with
>    your current version. This will regenerate the joblib files.
>
> The application itself now sets ``multi_class='ovr'`` on loaded logistic
> models as a fallback, but aligning versions is recommended for stability.

## 🛠️ Installation & Setup

### 1. Prerequisites
-   Python 3.8 or higher installed.

### 2. Set Up Virtual Environment
It is recommended to use a virtual environment to keep dependencies isolated.

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Run the following command to install all required libraries. Make sure you execute it from the
`ASIF/ASIF` directory (where the `requirements.txt` lives):

```bash
cd ASIF/ASIF
pip install -r requirements.txt
```

Alternatively you can run the top–level helper from the workspace root:

```bash
pip install -r requirements.txt
```

(That file simply forwards to the inner one.)

> **⚠️ Important:** the models depend on `xgboost` which isn't a standard
> library in some Python distributions. If you see an error such as
> `ModuleNotFoundError: No module named 'xgboost'` when starting the server,
> make sure the package is installed in **the same** interpreter used to run
> `app.py` (for example your virtualenv or Anaconda environment). You can
> install it manually with:
>
> ```bash
> pip install xgboost
> ```


## 🏃‍♂️ How to Run

### 1. Train the Models (First Time Only)
Before running the app, ensure the models are trained and saved in the `models/` directory.
```bash
python train_models.py
```
*This will generate `models/model_addiction.joblib`, `models/model_mental_health.joblib`, and necessary scalers.*

### 2. Start the Application
Run the Flask server:
```bash
python app.py
```

### 3. Access the App
Open your web browser and go to:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

## 📊 How It Works
1.  **Input Data**: Enter details like Age, Average Usage, Sleep Hours, etc.
2.  **Analysis**: The app processes your data using the pre-trained XGBoost and Logistic Regression models.
3.  **Results**: You receive a predicted Mental Health Score and Addiction Risk.
4.  **Comparison**: A graph shows where you stand compared to other students in the dataset.

---
*Created by Antigravity*
