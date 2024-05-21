import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


# Load the models and validation data
lr_model = joblib.load('../train_models/lr_model.joblib')
rf_model = joblib.load('../train_models/rf_best_model.joblib')
xgb_model = joblib.load('../train_models/xgb_model.joblib')
X_val, y_val = joblib.load('../processing data/val.joblib')

# Load the feature names and selected features
feature_names = joblib.load('../processing data/feature_names.joblib')
selected_features = joblib.load('../processing data/selected_features.joblib')

# Prepare validation data with selected features
X_val_selected = pd.DataFrame(X_val, columns=feature_names)[selected_features]

# Function to calculate net benefit
def net_benefit(y_true, y_pred_proba, threshold):
    pred_labels = y_pred_proba >= threshold
    tp = np.sum((y_true == 1) & (pred_labels == 1))
    fp = np.sum((y_true == 0) & (pred_labels == 1))
    nb = tp / len(y_true) - fp / len(y_true) * (threshold / (1 - threshold))
    return nb

# Calculate net benefit for different thresholds
thresholds = np.linspace(0, 1, 100)

# Logistic Regression model
y_val_lr_pred_proba = lr_model.predict_proba(X_val_selected)[:, 1]
net_benefits_lr = [net_benefit(y_val, y_val_lr_pred_proba, t) for t in thresholds]

# Random Forest model
y_val_rf_pred_proba = rf_model.predict_proba(X_val_selected)[:, 1]
net_benefits_rf = [net_benefit(y_val, y_val_rf_pred_proba, t) for t in thresholds]

# XGBoost model
y_val_xgb_pred_proba = xgb_model.predict_proba(X_val_selected)[:, 1]
net_benefits_xgb = [net_benefit(y_val, y_val_xgb_pred_proba, t) for t in thresholds]

# Plot DCA curves for validation set
plt.figure(figsize=(10, 6))
plt.plot(thresholds, net_benefits_lr, label='Logistic Regression Model')
plt.plot(thresholds, net_benefits_rf, label='Random Forest Model')
plt.plot(thresholds, net_benefits_xgb, label='XGBoost Model')
plt.plot(thresholds, thresholds - thresholds, linestyle='--', color='black', label='Treat All')
plt.plot(thresholds, thresholds * 0, linestyle='--', color='blue', label='Treat None')
plt.xlabel('Threshold Probability')
plt.ylabel('Net Benefit')
plt.title('Decision Curve Analysis (Validation Set)')
plt.legend()
plt.show()

# Load the test data
X_test, y_test = joblib.load('../processing data/test.joblib')

# Prepare test data with selected features
X_test_selected = pd.DataFrame(X_test, columns=feature_names)[selected_features]

# Logistic Regression model
y_test_lr_pred_proba = lr_model.predict_proba(X_test_selected)[:, 1]
net_benefits_lr = [net_benefit(y_test, y_test_lr_pred_proba, t) for t in thresholds]

# Random Forest model
y_test_rf_pred_proba = rf_model.predict_proba(X_test_selected)[:, 1]
net_benefits_rf = [net_benefit(y_test, y_test_rf_pred_proba, t) for t in thresholds]

# XGBoost model
y_test_xgb_pred_proba = xgb_model.predict_proba(X_test_selected)[:, 1]
net_benefits_xgb = [net_benefit(y_test, y_test_xgb_pred_proba, t) for t in thresholds]

# Plot DCA curves for test set
plt.figure(figsize=(10, 6))
plt.plot(thresholds, net_benefits_lr, label='Logistic Regression Model')
plt.plot(thresholds, net_benefits_rf, label='Random Forest Model')
plt.plot(thresholds, net_benefits_xgb, label='XGBoost Model')
plt.plot(thresholds, thresholds - thresholds, linestyle='--', color='black', label='Treat All')
plt.plot(thresholds, thresholds * 0, linestyle='--', color='blue', label='Treat None')
plt.xlabel('Threshold Probability')
plt.ylabel('Net Benefit')
plt.title('Decision Curve Analysis (Test Set)')
plt.legend()
plt.show()
