import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Load datasets
X_train, y_train = joblib.load('../processing data/train.joblib')


# Load feature names and selected features
feature_names = joblib.load('../processing data/feature_names.joblib')
selected_features = joblib.load('../processing data/selected_features.joblib')

# Prepare training and validation data with selected features
X_train_selected = pd.DataFrame(X_train, columns=feature_names)[selected_features]

# Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Initialize RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit GridSearchCV
grid_search.fit(X_train_selected, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best parameters found by GridSearchCV: {best_params}")

# Train the RandomForestClassifier with the best parameters
rf_best = RandomForestClassifier(**best_params)
rf_best.fit(X_train_selected, y_train)

# Perform Cross-Validation to check for overfitting
cv_scores = cross_val_score(rf_best, X_train_selected, y_train, cv=5, scoring='accuracy')
mean_cv_score = cv_scores.mean()

print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean Cross-Validation Score: {mean_cv_score}')

# Save the trained model to disk
joblib.dump(rf_best, 'rf_best_model.joblib')
