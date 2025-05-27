from sklearn.datasets import load_iris
import pandas as pd

# Load the iris dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

print("First five rows of the Iris dataset:")
print(df.head())

