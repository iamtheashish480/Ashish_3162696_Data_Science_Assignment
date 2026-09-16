# Telco Customer Churn Prediction

End-to-end customer churn prediction using the **same IBM Telco Customer Churn dataset supplied with the assignment**. The submission includes data preparation, EDA, leakage-safe feature engineering, Decision Tree modelling, class-imbalance handling, hyperparameter tuning, additional model comparison, model interpretation, a saved preprocessing+model pipeline, a FastAPI REST API, and an optional Streamlit user interface.

## Assignment coverage

- 70:30 stratified train/test split with `random_state=42`.
- Data types, missing values, duplicates, target balance and preprocessing checks.
- 5+ meaningful EDA visualizations with business insights.
- Three engineered features: `AvgMonthlyCharge`, `NumServices`, and `TenureGroup`.
- Required Decision Tree with multiple configurations and final tuned model.
- Evaluation: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC and PR-AUC.
- Model interpretation: feature importance and Decision Tree visualization.
- Bonus: class imbalance via `class_weight='balanced'`, 5-fold `GridSearchCV` hyperparameter tuning, and comparison with Logistic Regression and Random Forest.
- Saved full pipeline in `model/churn_pipeline.pkl`, so feature engineering and preprocessing are identical during API/UI inference.
- **Additional UI:** Streamlit interface for interactive customer-level prediction and model-performance/feature-importance display.

## Project structure

```text
customer_churn_project/
├── data/
│   ├── TelcoCustomerChurn.csv
│   └── TelcoCustomerChurn - Data Dictionary.csv
├── notebook/
│   └── churn_analysis.ipynb
├── model/
│   ├── churn_pipeline.pkl
│   ├── feature_importance.csv
│   └── metrics.json
├── app.py                 # FastAPI REST API
├── streamlit_app.py       # Optional interactive UI
├── model_utils.py
├── train.py
├── requirements.txt
├── sample_request.json
├── sample_response.json
└── README.md
```

## Setup

From the project root:

```bash
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
# venv\\Scripts\\activate

pip install -r requirements.txt
```

## Train / reproduce

```bash
python train.py
```

This rebuilds the saved pipeline and evaluation artifacts. The notebook contains the complete analysis and modelling workflow and can be run top-to-bottom.

## Run the FastAPI REST API

```bash
python app.py
```

Open:

- API landing page: `http://127.0.0.1:8000/`
- Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

### POST `/predict`

Send the JSON structure in `sample_request.json`.

Example response:

```json
{
  "prediction": "Yes",
  "churn_probability": 0.8627
}
```

### POST `/predict_batch`

Accepts a JSON array of customer objects and returns one prediction per customer.

## Run the Streamlit UI

In a second terminal, from the project root:

```bash
streamlit run streamlit_app.py
```

Streamlit will display a local browser URL, normally:

`http://localhost:8501`

The UI provides:

- Customer profile and service inputs.
- Contract and billing inputs.
- One-click churn prediction.
- Churn probability and Yes/No prediction.
- Final model performance metrics.
- Top feature-importance visualization.
- Submitted payload preview.

The Streamlit UI loads the **same `model/churn_pipeline.pkl`** used by the FastAPI API. It does not train a separate model.

## Modelling notes

The churn target is imbalanced. The project compares an unbalanced Decision Tree with a class-balanced Decision Tree and uses class-balanced models for the tuned tree, Logistic Regression and Random Forest. Hyperparameters are selected only using the training split through 5-fold cross-validation, preventing test-set leakage. The final submitted model is the tuned, class-balanced Decision Tree because a Decision Tree is explicitly required by the assignment. F1 is used for tuning because it balances precision and recall; recall is also important because missing a true churner can mean a missed retention opportunity.

## API and UI architecture

```text
                    ┌────────────────────────┐
                    │   Streamlit UI          │
                    │   streamlit_app.py      │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │ Saved ML Pipeline       │
                    │ churn_pipeline.pkl      │
                    └───────────▲────────────┘
                                │
                    ┌───────────┴────────────┐
                    │ FastAPI REST API       │
                    │ app.py /predict        │
                    └────────────────────────┘
```

Both interfaces use the same saved preprocessing + feature-engineering + Decision Tree pipeline.

## Submission workflow

**Business Problem → Data → Preparation → EDA → Feature Engineering → Model → Evaluation → Interpretation → Saved Model → API → Interactive UI**
