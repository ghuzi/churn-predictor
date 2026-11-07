import json
import math
import numbers

# Load the schema that was built from the training data
with open("churn_schema.json") as f:
    SCHEMA = json.load(f)

INTERNET_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                 "TechSupport", "StreamingTV", "StreamingMovies"]


class ChurnInputError(ValueError):
    """Raised when a customer record does not match the expected schema."""


def _is_number(v):
    # True for real, finite numbers (int/float/numpy numbers), but not for bool
    return isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isfinite(v)


def validate_customer(data):
    """Validate one customer dictionary.

    Returns (clean_record, warnings). Raises ChurnInputError listing ALL problems found.
    """
    if not isinstance(data, dict):
        raise ChurnInputError("Input must be a dictionary of customer fields.")

    errors, warnings, clean = [], [], {}
    features = SCHEMA["features"]

    # 1. Missing and unexpected fields
    missing = [f for f in features if f not in data]
    extra = [k for k in data if k not in features]
    if missing:
        errors.append(f"Missing fields: {missing}")
    if extra:
        errors.append(f"Unexpected fields: {extra}")

    # 2. Categorical fields must use one of the known values (exact spelling)
    for col, allowed in SCHEMA["categorical"].items():
        if col in data:
            value = data[col].strip() if isinstance(data[col], str) else data[col]
            if value in allowed:
                clean[col] = value
            else:
                errors.append(f"{col}: {data[col]!r} is not valid. Allowed values: {allowed}")

    # 3. Numeric fields must be real numbers inside sensible limits
    for col, rule in SCHEMA["numeric"].items():
        if col not in data:
            continue
        value = data[col]
        if not _is_number(value):
            errors.append(f"{col}: must be a finite number, got {value!r}")
            continue
        if col == "SeniorCitizen":
            if value in (0, 1):
                clean[col] = int(value)
            else:
                errors.append(f"SeniorCitizen: must be 0 or 1, got {value!r}")
            continue
        if col == "tenure" and value != int(value):
            errors.append(f"tenure: must be a whole number of months, got {value!r}")
            continue
        if value < rule["hard_min"] or value > rule["hard_max"]:
            errors.append(f"{col}: {value} is outside the allowed range "
                          f"[{rule['hard_min']}, {rule['hard_max']}]")
            continue
        if value < rule["train_min"] or value > rule["train_max"]:
            warnings.append(f"{col}: {value} is outside the range seen in training "
                            f"[{rule['train_min']}, {rule['train_max']}], so the prediction may be less reliable.")
        clean[col] = int(value) if col == "tenure" else float(value)

    # 4. Cross-field consistency (only when the fields involved are valid)
    phone, lines = clean.get("PhoneService"), clean.get("MultipleLines")
    if phone == "No" and lines is not None and lines != "No phone service":
        errors.append("MultipleLines must be 'No phone service' when PhoneService is 'No'.")
    if phone == "Yes" and lines == "No phone service":
        errors.append("MultipleLines cannot be 'No phone service' when PhoneService is 'Yes'.")

    internet = clean.get("InternetService")
    for col in INTERNET_COLS:
        v = clean.get(col)
        if internet == "No" and v is not None and v != "No internet service":
            errors.append(f"{col} must be 'No internet service' when InternetService is 'No'.")
        if internet in ("DSL", "Fiber optic") and v == "No internet service":
            errors.append(f"{col} cannot be 'No internet service' when the customer has internet.")

    if errors:
        raise ChurnInputError("\n".join(f"- {e}" for e in errors))

    # Return the record in the exact column order used for training
    return {f: clean[f] for f in features}, warnings
