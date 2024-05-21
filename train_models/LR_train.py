import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
import joblib

# Load training dataset
X_train, y_train = joblib.load('../processing data/train.joblib')

# Load feature names and selected features
feature_names = joblib.load('../processing data/feature_names.joblib')
selected_features = joblib.load('../processing data/selected_features.joblib')

# Prepare training data with selected features
X_train_selected = pd.DataFrame(X_train, columns=feature_names)[selected_features]

# Define the logistic regression model within a pipeline
model_pipeline = Pipeline([
    ('logisticregression', LogisticRegression(random_state=42, max_iter=10000))
])

# Setup GridSearchCV for hyperparameter tuning
param_grid = {
    'logisticregression__C': np.logspace(-4, 4, 20),
    'logisticregression__penalty': ['l1', 'l2'],
    'logisticregression__solver': ['liblinear', 'saga']
}

# Use StratifiedKFold for cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Setup GridSearchCV
grid_search = GridSearchCV(model_pipeline, param_grid=param_grid, cv=skf, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train_selected, y_train)

# Best parameters and model
print("Best parameters found by grid search are:", grid_search.best_params_)
lr_model = grid_search.best_estimator_

# Evaluate the model using cross-validation on the training set
cv_scores = cross_val_score(lr_model, X_train_selected, y_train, cv=5, scoring='accuracy')
mean_cv_score = cv_scores.mean()

print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean Cross-Validation Score: {mean_cv_score:.4f}')

# Save the best model to disk for future evaluation
joblib.dump(lr_model, 'lr_model.joblib')
