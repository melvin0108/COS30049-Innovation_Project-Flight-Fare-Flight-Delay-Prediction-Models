import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
import pickle
import os

# Load the dataset
data_path = os.path.join("dataset", "Cleaned_Dataset.csv")
data = pd.read_csv(data_path)

# # Convert price to AUD (assuming price is in rupees)
# data['price_aud'] = data['price'] / 57

# # Save the cleaned dataset to a CSV file
# data.to_csv('Cleaned_Dataset.csv', index=False)

# Preprocess categorical columns using Label Encoding
label_columns = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']
le = LabelEncoder()

for column in label_columns:
    if data[column].dtype == 'object':  # Check if the column is categorical
        data[column] = le.fit_transform(data[column])

# Verify data types to ensure all columns are numeric
#print(data.dtypes)

# Prepare features (X) and target (Y)
X = data.drop(['price_aud', 'price', 'flight'], axis=1)
Y = data['price_aud']

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=108)

# Initialize the RandomForestRegressor with optimized parameters
model = RandomForestRegressor(
    n_estimators=100,          
    criterion="squared_error",
    min_samples_split=5,
    min_samples_leaf=1,
    max_depth=10,             
    random_state=108,
    n_jobs = -1
)

# Train the model
model.fit(X_train, Y_train)

# Save the trained model using pickle
with open('model/rf_regressor.pkl', 'wb') as file:
    pickle.dump(model, file)

# Make predictions on the test set
Y_pred = model.predict(X_test)

# Evaluate the model performance using various metrics
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
medae = median_absolute_error(Y_test, Y_pred)

# Perform cross-validation to check for consistency
cv_scores = cross_val_score(model, X, Y, cv=3, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)

# Print performance metrics
print("\nPerformance Metrics:")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared Score: {r2:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Median Absolute Error: {medae:.2f}")
print(f"Cross-validation RMSE: {cv_rmse.mean():.2f} (+/- {cv_rmse.std() * 2:.2f})")

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 Most Important Features:")
print(feature_importance.head())

# Visualizations

# Plot: Actual vs Predicted Prices
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred, alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Price (AUD)")
plt.ylabel("Predicted Price (AUD)")
plt.title("Actual vs Predicted Prices")
plt.tight_layout()
plt.savefig("Visualization/actual_vs_predicted.png")
plt.close()

# Plot: Feature Importance (Top 10)
plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title("Top 10 Most Important Features")
plt.tight_layout()
plt.savefig("Visualization/feature_importance.png")
plt.close()

# Plot: Residuals
residuals = Y_test - Y_pred
plt.figure(figsize=(10, 6))
plt.scatter(Y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Price (AUD)")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("Visualization/residual_plot.png")
plt.close()

print("\nVisualization plots have been saved as PNG files.")


# Performance Metrics:
# Root Mean Squared Error: 75.78
# R-squared Score: 0.96
# Mean Absolute Error: 42.72
# Median Absolute Error: 20.36
# Cross-validation RMSE: 291.99 (+/- 707.92)

#Create the parameter grid using sklearn's ParameterGrid
# grid = ParameterGrid(param_grid)

# Function to evaluate one set of parameters
# def evaluate_params(params, X_train, X_test, Y_train, Y_test):
    # Create a new RandomForestRegressor model
    # model = RandomForestRegressor(random_state=108)
    # model.set_params(**params)
    
    # Fit the model on training data
    # model.fit(X_train, Y_train)
    
    # Predict on test data
    # Y_pred = model.predict(X_test)
    
    # Calculate the negative mean squared error (as we want to maximize the score)
    # score = -mean_squared_error(Y_test, Y_pred)
    
    # return score, params

# Use joblib's Parallel to run evaluations in parallel
# results = Parallel(n_jobs=-1)(
#     delayed(evaluate_params)(params, X_train.copy(), X_test.copy(), Y_train.copy(), Y_test.copy()) 
#     for params in tqdm(grid, desc="Grid Search Progress")
# )

# Find the best result
# best_score, best_params = max(results, key=lambda x: x[0])

# Output best parameters and best score
# print("Best parameters found: ", best_params)
# print("Best score: ", best_score)
#Best parameters found:  {'criterion': 'squared_error', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}


