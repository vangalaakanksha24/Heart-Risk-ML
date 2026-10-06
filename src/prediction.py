"""Optional terminal input for a single patient prediction."""
from __future__ import annotations

import pandas as pd

from data_understanding import infer_feature_types


def _read_value(column: str, is_categorical: bool, allowed_values: list, is_integer: bool):
    """Prompt until a valid value is supplied for one dataset feature."""
    allowed_text = f" [{', '.join(map(str, allowed_values))}]" if is_categorical else ""
    while True:
        raw = input(f"{column}{allowed_text}: ").strip()
        try:
            value = int(raw) if is_integer else float(raw)
        except ValueError:
            print("Please enter a valid numeric value.")
            continue
        if is_categorical and value not in allowed_values:
            print(f"Please choose one of: {', '.join(map(str, allowed_values))}.")
            continue
        return value


def collect_patient_values(training_features: pd.DataFrame) -> pd.DataFrame:
    """Collect a record matching the original dataset schema, including age."""
    _, categorical_columns = infer_feature_types(training_features)
    values: dict[str, float | int] = {}
    print("\nPATIENT INPUT\n-------------")
    print("Enter values using the displayed coding for categorical fields.")
    for column in training_features.columns:
        series = training_features[column]
        is_categorical = column in categorical_columns
        allowed = sorted(series.dropna().unique().tolist()) if is_categorical else []
        is_integer = pd.api.types.is_integer_dtype(series)
        values[column] = _read_value(column, is_categorical, allowed, is_integer)
    return pd.DataFrame([values], columns=training_features.columns)


def offer_patient_prediction(model, training_features: pd.DataFrame, clinical_feature_adder) -> None:
    """Offer a user-controlled prediction after test evaluation finishes."""
    answer = input("\nDo you want to predict heart disease for a new patient? (y/n): ").strip().lower()
    if answer not in {"y", "yes"}:
        print("New-patient prediction skipped.")
        return
    patient = collect_patient_values(training_features)
    patient_with_clinical_features, _ = clinical_feature_adder(patient)
    prediction = int(model.predict(patient_with_clinical_features)[0])
    result = "Yes (heart disease likely)" if prediction == 1 else "No (heart disease not predicted)"
    print("\nNEW PATIENT PREDICTION\n----------------------")
    print(f"Heart Disease: {result}")
