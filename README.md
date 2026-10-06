# Heart Disease Prediction

## Objective

This terminal-only Python project compares four heart-disease classification architectures while preserving a final, untouched 20% test set. Its target is at least 90% test accuracy, but it reports only measured performance and makes no accuracy guarantee.

## Dataset

The project uses the provided `data/heart_2020_cleaned.csv` only. At runtime it inspects the CSV, verifies the target column, reports its schema and distribution, and removes duplicate rows before splitting.

## Architecture

`Dataset → Data Understanding → Preprocessing → 80/20 Stratified Split → General Feature Engineering → Clinical Features → Feature Selection → Tuning → Models/Ensembles → Final Test Evaluation`

The models are Random Forest, RF + SVM soft voting, RF + SVM + XGBoost stacking, and RF + SVM + XGBoost + LightGBM stacking. Stacking uses Logistic Regression as its meta-classifier.

## Preprocessing and selection

Imputation, encoding, scaling, general feature engineering, and feature selection occur within sklearn pipelines so they are learned from training folds only. The project compares Polynomial Features with PowerTransformer, and Mutual Information with RFE, using five-fold stratified cross-validation. It adds clinical features only when the relevant source columns exist.

## Hyperparameter optimization and evaluation

RandomizedSearchCV tunes Random Forest, SVM, XGBoost, and LightGBM using five-fold stratified cross-validation on the training partition. Final evaluation reports Accuracy, Precision, Recall, and F1-Score. Mutual Information, RFE, and L1 model-based selection are compared by cross-validated Precision, Recall, and F1-Score in `outputs/feature_selection_scores.png`. The final best model alone receives test-set ROC and confusion-matrix charts.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Run

From the `heart_disease_prediction` directory:

```bash
python src/main.py
```

After evaluation, choose `y` when the program asks about a new-patient prediction.
It will request age and all other dataset fields, then print `Yes` or `No`.
"# Heart-Risk-ML" 
