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

from sklearn.linear_model import LinearRegression
# Initialize and train the model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Hyperparameter tuning using Ridge regression with cross-validation
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge = Ridge()
param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_ridge = grid_search.best_estimator_
y_pred = best_ridge.predict(X_test)

print("Best Alpha:", grid_search.best_params_['alpha'])




