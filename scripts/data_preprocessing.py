import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("../data/BostonHousing.csv")
# Handle missing values (if any)
df_cleaned = df.dropna()

# No categorical variables in this dataset, so no encoding is needed

# Normalize/standardize numerical features
features = df_cleaned.drop("medv", axis=1)
target = df_cleaned["medv"]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

print("\nTraining and testing sets created:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)