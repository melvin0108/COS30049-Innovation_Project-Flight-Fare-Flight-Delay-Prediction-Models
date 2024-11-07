import pandas as pd
import os
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lbm
import joblib

# Load the data
data_path = os.path.join("dataset", "2015-Cleaned_flight_data.csv")
flight_data = pd.read_csv(data_path)

# Measure start time
start_time = time.time()

# Define columns to use as features
categorical_cols = ["MONTH", "DAY", "DAY_OF_WEEK", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
numerical_cols = ["SCHEDULED_DEPARTURE", "DEPARTURE_DELAY", "AIR_TIME", "DISTANCE", "SCHEDULED_ARRIVAL"]

# Extract the target variable
target = flight_data["ARRIVAL_DELAY"]

# Initialize dictionaries to store LabelEncoders and the StandardScaler
label_encoders = {}
scaler = StandardScaler()

# Encode categorical columns using LabelEncoder
for col in categorical_cols:
    le = LabelEncoder()
    flight_data[col] = le.fit_transform(flight_data[col].astype(str))  # Ensure values are strings
    label_encoders[col] = le  # Store the encoder for future use if needed

# Scale numerical columns using StandardScaler
flight_data[numerical_cols] = scaler.fit_transform(flight_data[numerical_cols])

# Combine the selected features (categorical + numerical)
X = flight_data[categorical_cols + numerical_cols]  # Features
y = target  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=108)

# Measure start time for training
train_start_time = time.time()

# Create and train the LightGBM regressor
regressor = lbm.LGBMRegressor(
    learning_rate=0.1,
    max_depth=5,
    n_estimators=200,
    min_child_samples=1
)
regressor.fit(X_train, y_train)

# Measure end time for training
train_end_time = time.time()
train_time = train_end_time - train_start_time
print(f"Total training time: {train_time:.2f} seconds")

# Make predictions on the test set
y_pred = regressor.predict(X_test)

# Calculate evaluation metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Output the results
print("\nEvaluation Metrics:")
print(f"R-squared (R²): {r2:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Save the preprocessor and preprocessed data to pickle
preprocessor = {
    "label_encoders": label_encoders,
    "scaler": scaler,
}

with open('preprocessor/preprocessorfordelays.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
print("\nPreprocessor saved as 'preprocessorfordelays.pkl'")

# Save the trained model to a pickle file
with open("model/lgbm_regressor_delay.pkl", "wb") as f:
    pickle.dump(regressor, f)
print("Model saved as 'lgbm_regressor_delay.pkl'")

# Measure total script execution time
end_time = time.time()
running_time = end_time - start_time
print(f"Total script execution time: {running_time:.2f} seconds")


#Best hyperparameter combination found with GridsearchCV:
#{'regressor__learning_rate': 0.1, 'regressor__max_depth': 5, 'regressor__min_samples_leaf': 1, 'regressor__min_samples_split': 5, 'regressor__n_estimators': 200}



# profile = ProfileReport(dropped_data, title="My report")
# profile.to_file("Visualization/rgs_flight_data_report.html")

# Total training time: 1.17 seconds

# Evaluation Metrics:
# R-squared (R²): 0.91
# Mean Absolute Error (MAE): 7.90
# Mean Squared Error (MSE): 139.12

# Model saved as '2015-LGBMRegressor_FlightDelay.pkl'
# Total script execution time: 1.31 seconds
