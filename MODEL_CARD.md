# Model Card: Customer Churn Predictor

*Created: 2026-11-07 | scikit-learn 1.6.1*

## Model details
- **Task:** Binary classification, predicting whether a telecom customer will churn (leave).
- **Model:** Logistic Regression (`class_weight="balanced"`, `C=0.001`, `solver="liblinear"`) inside a scikit-learn Pipeline.
- **Pipeline:** `StandardScaler` on tenure, MonthlyCharges and TotalCharges, `OneHotEncoder` on the 15 categorical columns, `SeniorCitizen` passed through, then the classifier.
- **Output:** a churn probability, converted to a risk level: Low (below 40%), Medium (40% to 70%), High (70% and above).
- **File:** `churn_model.joblib`. Input rules are in `validation.py` and `churn_schema.json`.

## Intended use
- Help a retention team rank customers by churn risk and decide who gets a retention offer.
- A decision-support tool for a portfolio and learning project, not a fully automated decision-maker.

## Out of scope
- Any decision that affects a customer's price, service or eligibility without human review.
- Other companies, countries or time periods: the model has only seen this one dataset.
- Explaining *why* a single customer will leave. The model finds patterns, not causes.

## Data
- Telco Customer Churn dataset from Kaggle: 7,043 customers, 21 columns, 26.5% churned.
- `customerID` dropped. 11 blank `TotalCharges` values (customers with tenure 0) filled with 0.
- Split 80/20 with `stratify=y` and `random_state=42`: 5,634 training rows and 1,409 test rows.

## Training
- Three algorithms compared on the same split, then `class_weight="balanced"` added to handle the 26.5% / 73.5% class imbalance.
- Best model (Logistic Regression) tuned with `GridSearchCV`, 5-fold cross-validation on the training set only, scored by recall.

## Performance (test set, 1,409 customers, 374 real churners)

| Metric | Value |
|---|---|
| Recall (churners caught) | 0.807 |
| Precision (flagged customers who really left) | 0.486 |
| F1 | 0.606 |
| Accuracy | 0.722 |
| Recall for customers who stayed | 0.691 |

| | Predicted stay | Predicted churn |
|---|---|---|
| **Actually stayed** | 715 | 320 |
| **Actually churned** | 72 | 302 |

The model caught 302 of 374 real churners, missed 72, and raised 320 false alarms.

### Model comparison (test set)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression (default) | 0.806 | 0.657 | 0.559 | 0.604 |
| Decision Tree (default) | 0.728 | 0.488 | 0.500 | 0.494 |
| Random Forest (default) | 0.786 | 0.623 | 0.487 | 0.547 |
| Logistic Regression (balanced) | 0.738 | 0.504 | 0.783 | 0.614 |
| Decision Tree (balanced) | 0.736 | 0.503 | 0.481 | 0.492 |
| Random Forest (balanced) | 0.778 | 0.607 | 0.463 | 0.525 |

## Input validation
Every record is checked before prediction: all 19 fields present, no unexpected fields, exact category spelling, numeric values inside sensible limits, and fields that agree with each other. A warning is shown when a number falls outside the range seen in training.

## Limitations
- **Trade-off:** the model favours recall, so about half of the customers it flags will not actually leave (320 false alarms against 302 correct catches). It is only worth using if a retention offer costs much less than losing a customer.
- **Test set reuse:** the test set was also used to compare the three algorithms and choose Logistic Regression, so the scores above are slightly optimistic. The tuning step itself used cross-validation on the training data only.
- **Small tuning gain:** tuning moved recall from 0.783 to 0.807 on 374 churners (about 9 customers), which may be within random variation. A single split was used, with no repeated runs.
- **Probabilities are not calibrated:** because of `class_weight="balanced"`, a score of 70% does not mean a 70% real chance of churn. Treat the score as a ranking.
- **Patterns, not causes:** features such as payment method are associated with churn, but changing them may not prevent it. Offers should be tested on a small group first.
- **Narrow data:** one snapshot from one company. The model will need re-checking if prices, plans or customer behaviour change.
- **Fields not cross-checked:** `TotalCharges` is not compared with tenure x monthly bill, so inconsistent values can pass validation.

## Ethical considerations
- The model uses `gender` and `SeniorCitizen` as inputs. No fairness audit across groups has been done, and these fields should be reviewed (or removed) before any real use.
- Risk scores should lead to a **helpful action** (a better offer or a support call), not to worse treatment of a customer.

## How to use
```python
import joblib
from validation import validate_customer

model = joblib.load("churn_model.joblib")
clean, warnings = validate_customer(customer_dict)   # raises ChurnInputError on bad input
probability = model.predict_proba(pd.DataFrame([clean]))[0, 1]
```
Load it with the same scikit-learn version (1.6.1) used for training.
