# Customer Churn Predictor

A Streamlit web app that predicts whether a telecom customer is likely to churn (leave), built with scikit-learn on the Telco Customer Churn dataset.

**Live demo:** https://churn-predictor-huzaif.streamlit.app

## What it does

Enter a customer's details (demographics, services, contract, billing) and the app returns a churn probability along with a risk level:

- **Low:** below 40%
- **Medium:** 40% to 70%
- **High:** 70% and above

It is a decision-support tool to help a retention team rank customers by churn risk, not a fully automated decision-maker.

## Model

- **Task:** Binary classification (churn / no churn)
- **Algorithm:** Logistic Regression (`class_weight="balanced"`, `C=0.001`, `solver="liblinear"`)
- **Preprocessing:** StandardScaler on tenure, MonthlyCharges and TotalCharges; OneHotEncoder on 15 categorical columns; SeniorCitizen passed through
- **Dataset:** Telco Customer Churn, 7,043 customers and 21 columns

See [MODEL_CARD.md](MODEL_CARD.md) for full details, intended use and limitations.

## Project structure

| File | Purpose |
|------|---------|
| `app.py` | Streamlit user interface |
| `validation.py` | Input validation and custom error handling |
| `churn_schema.json` | Expected input columns and allowed values |
| `churn_model.joblib` | Trained scikit-learn pipeline (preprocessing + model) |
| `telco_churn.csv` | Training dataset |
| `requirements.txt` | Python dependencies |

## Run locally

```
git clone https://github.com/ghuzi/churn-predictor.git
cd churn-predictor
pip install -r requirements.txt
streamlit run app.py
```

## Tech stack

Python, pandas, scikit-learn 1.6.1, joblib, Streamlit

## Author

Huzaif (Muhammad Ghuzaif Bin Arif)
[GitHub](https://github.com/ghuzi) | [LinkedIn](https://www.linkedin.com/in/muhammad-ghuzaif-arif-b68bb8360)
