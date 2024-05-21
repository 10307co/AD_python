import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, f1_score, precision_score,
                             recall_score, cohen_kappa_score, roc_auc_score, brier_score_loss)
from sklearn.utils import resample

# Load the model and test dataset
rf_optimal = joblib.load('../train_models/rf_best_model.joblib')
X_test, y_test = joblib.load('../processing data/test.joblib')

# Load the feature names and selected features
feature_names = joblib.load('../processing data/feature_names.joblib')
selected_features = joblib.load('../processing data/selected_features.joblib')

# Prepare test data with selected features
X_test_selected = pd.DataFrame(X_test, columns=feature_names)[selected_features]

# Evaluate the model on the test set
rf_test_pred = rf_optimal.predict(X_test_selected)
rf_test_pred_prob = rf_optimal.predict_proba(X_test_selected)[:, 1]

# Print classification report and confusion matrix
print("Classification Report on Test Set:\n", classification_report(y_test, rf_test_pred))
cm = confusion_matrix(y_test, rf_test_pred)
print("Confusion Matrix on Test Set:\n", cm)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, rf_test_pred)
print("Accuracy on Test Set:", accuracy)

# Calculate and print ROC AUC Score
roc_auc = roc_auc_score(y_test, rf_test_pred_prob)
print("ROC AUC Score:", roc_auc)

# Calculate other performance metrics
precision = precision_score(y_test, rf_test_pred)
recall = recall_score(y_test, rf_test_pred)
kappa = cohen_kappa_score(y_test, rf_test_pred)
f1 = f1_score(y_test, rf_test_pred)

print("Precision:", precision)
print("Recall (Sensitivity):", recall)
print("Kappa Score:", kappa)
print("F1 Score:", f1)

# Specificity calculation
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
print("Specificity: ", specificity)

# Calculate Brier score
brier = brier_score_loss(y_test, rf_test_pred_prob)
print("Brier Score:", brier)

# Save the predicted probabilities for further analysis
joblib.dump(rf_test_pred_prob, '../test_models/rf_test_probs.joblib')

# Convert y_test and rf_test_pred to numpy arrays if they are not already
y_test = np.array(y_test)
rf_test_pred = np.array(rf_test_pred)

# Function to calculate confidence intervals using bootstrapping
def bootstrap_metric(y_true, y_pred, metric_func, y_pred_proba=None, n_bootstraps=1000, alpha=0.95):
    bootstrapped_scores = []
    rng = np.random.RandomState(seed=42)
    for _ in range(n_bootstraps):
        indices = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        if metric_func == roc_auc_score:
            score = metric_func(y_true[indices], y_pred_proba[indices])
        else:
            score = metric_func(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    lower_bound = sorted_scores[int((1.0 - alpha) / 2 * len(sorted_scores))]
    upper_bound = sorted_scores[int((1.0 + alpha) / 2 * len(sorted_scores))]
    return lower_bound, upper_bound

# Function to calculate specificity
def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)

# Calculate confidence intervals
accuracy_ci = bootstrap_metric(y_test, rf_test_pred, accuracy_score)
precision_ci = bootstrap_metric(y_test, rf_test_pred, precision_score)
recall_ci = bootstrap_metric(y_test, rf_test_pred, recall_score)
f1_ci = bootstrap_metric(y_test, rf_test_pred, f1_score)
roc_auc_ci = bootstrap_metric(y_test, rf_test_pred_prob, roc_auc_score, y_pred_proba=rf_test_pred_prob)
specificity_ci = bootstrap_metric(y_test, rf_test_pred, specificity_score)

print(f"Accuracy: {accuracy} (95% CI: {accuracy_ci})")
print(f"Precision: {precision} (95% CI: {precision_ci})")
print(f"Recall (Sensitivity): {recall} (95% CI: {recall_ci})")
print(f"F1 Score: {f1} (95% CI: {f1_ci})")
print(f"ROC AUC Score: {roc_auc} (95% CI: {roc_auc_ci})")
print(f"Specificity: {specificity} (95% CI: {specificity_ci})")
