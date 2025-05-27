from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import xgboost as xgb

# Load the iris dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

print("First five rows of the Iris dataset:")
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import xgboost as xgb

# Load the iris dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

print("First five rows of the Iris dataset:")
print(df.head())

# Part A: Split the dataset
X = df
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Part B: Create default XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
print("\nXGBoost model created.")

# Part C: Fit the model
model.fit(X_train, y_train)
print("\n✅ Model has been trained.")
