from sklearn.datasets import load_iris
import pandas as pd

# Load the iris dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

print("\nTraining set:")
print(X_train.head())

print("\nTest set:")
print(X_test.head())
print("First five rows of the Iris dataset:")
print(df.head())

