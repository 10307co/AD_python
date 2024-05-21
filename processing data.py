import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from imblearn.over_sampling import ADASYN
import joblib

# Load the dataset
df = pd.read_csv('nacc_AD.csv')

# Define continuous and categorical variables
continuous_vars = ['EDUC', 'Age', 'BMI', 'sdp', 'Drugstaken']
categorical_vars = ['SEX', 'Marital', 'Mom', 'Stenting', 'Hypertensive', 'Diabetes', 'DP', 'Neurosis', 'REM', 'Insomnia']

# Preprocessor for continuous features
continuous_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler())
])

# Preprocessor for categorical features
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessors into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', continuous_transformer, continuous_vars),
        ('cat', categorical_transformer, categorical_vars)
    ]
)

# Apply preprocessing to the data
X = df.drop('AD', axis=1)  # Assuming 'AD' is the target variable and should not be included in X
y = df['AD']
X_preprocessed = preprocessor.fit_transform(X)

# Get the transformed feature names
feature_names = preprocessor.get_feature_names_out()

# Apply ADASYN to balance the data set
adasyn = ADASYN(random_state=42, n_neighbors=2)
X_resampled, y_resampled = adasyn.fit_resample(X_preprocessed, y)

# Split the data into training, validation, and testing sets
X_train_7, X_test, y_train_7, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_7, y_train_7, test_size=0.2, random_state=42)

# Save datasets to disk
joblib.dump((X_train_7, y_train_7), 'train_7.joblib')
joblib.dump((X_test, y_test), 'test.joblib')
joblib.dump((X_train, y_train), 'train.joblib')
joblib.dump((X_val, y_val), 'val.joblib')

# Feature selection using RFE
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=rf, n_features_to_select=10, step=1)
rfe.fit(X_train, y_train)

# Get RFE selected features
selected_features = np.array(feature_names)[rfe.support_].tolist()
print("Selected Features: ", selected_features)

# Save feature names and selected features
joblib.dump(feature_names.tolist(), 'feature_names.joblib')
joblib.dump(selected_features, 'selected_features.joblib')
