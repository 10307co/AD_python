import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report,
                             confusion_matrix, roc_auc_score, cohen_kappa_score, brier_score_loss, roc_curve)
from sklearn.utils import resample

# Load the model and validation dataset
xgb_model = joblib.load('../train_models/xgb_model.joblib')
X_val, y_val = joblib.load('../processing data/val.joblib')

# Load the feature names and selected features
feature_names = joblib.load('../processing data/feature_names.joblib')
selected_features = joblib.load('../processing data/selected_features.joblib')

# Prepare validation data with selected features
X_val_selected = pd.DataFrame(X_val, columns=feature_names)[selected_features]

# Predict on the validation set
y_val_pred = xgb_model.predict(X_val_selected)
y_val_pred_prob = xgb_model.predict_proba(X_val_selected)[:, 1]

# Classification report and confusion matrix
print("Classification Report on Validation Set:\n", classification_report(y_val, y_val_pred))
cm = confusion_matrix(y_val, y_val_pred)
print("Confusion Matrix on Validation Set:\n", cm)

# Calculate and print accuracy
accuracy = accuracy_score(y_val, y_val_pred)
print("Accuracy on Validation Set:", accuracy)

# Calculate and print ROC AUC Score
roc_auc = roc_auc_score(y_val, y_val_pred_prob)
print("ROC AUC Score:", roc_auc)

# Calculate other performance metrics
precision = precision_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)
kappa = cohen_kappa_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)

print("Precision:", precision)
print("Recall (Sensitivity):", recall)
print("Kappa Score:", kappa)
print("F1 Score:", f1)

# Specificity calculation
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
print("Specificity: ", specificity)

# Calculate Brier score
brier = brier_score_loss(y_val, y_val_pred_prob)
print("Brier Score:", brier)

# Save the predicted probabilities for further analysis
joblib.dump(y_val_pred_prob, '../validate_models/xgb_probs.joblib')

# Convert y_val and y_val_pred to numpy arrays
y_val_np = np.array(y_val)
y_pred_np = np.array(y_val_pred)

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
accuracy_ci = bootstrap_metric(y_val_np, y_pred_np, accuracy_score)
precision_ci = bootstrap_metric(y_val_np, y_pred_np, precision_score)
recall_ci = bootstrap_metric(y_val_np, y_pred_np, recall_score)
f1_ci = bootstrap_metric(y_val_np, y_pred_np, f1_score)
roc_auc_ci = bootstrap_metric(y_val_np, y_val_pred_prob, roc_auc_score, y_pred_proba=y_val_pred_prob)
specificity_ci = bootstrap_metric(y_val_np, y_pred_np, specificity_score)

print(f"Accuracy: {accuracy} (95% CI: {accuracy_ci})")
print(f"Precision: {precision} (95% CI: {precision_ci})")
print(f"Recall (Sensitivity): {recall} (95% CI: {recall_ci})")
print(f"F1 Score: {f1} (95% CI: {f1_ci})")
print(f"ROC AUC Score: {roc_auc} (95% CI: {roc_auc_ci})")
print(f"Specificity: {specificity} (95% CI: {specificity_ci})")

# Plot ROC curve
fpr_xgb, tpr_xgb, _ = roc_curve(y_val, y_val_pred_prob)

plt.figure()
plt.plot(fpr_xgb, tpr_xgb, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No AD', 'AD'], yticklabels=['No AD', 'AD'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix on Validation set')
plt.show()
