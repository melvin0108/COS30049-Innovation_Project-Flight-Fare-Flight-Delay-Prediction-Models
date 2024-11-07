import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load the data
airports_data = os.path.join("dataset", "airports_with_utc_offsets.csv")
airports_data = pd.read_csv(airports_data)

dropped_data = os.path.join("dataset", "flights.csv")
dropped_data = pd.read_csv(dropped_data)


# Create a mapping from IATA_CODE to UTC_OFFSET
airport_timezone_mapping = airports_data.set_index('IATA_CODE')['UTC_OFFSET'].to_dict()

# Create ORIGIN_AIRPORT_TZ and DESTINATION_AIRPORT_TZ columns by mapping IATA_CODE from ORIGIN_AIRPORT and DESTINATION_AIRPORT
dropped_data['ORIGIN_AIRPORT_TZ'] = dropped_data['ORIGIN_AIRPORT'].map(airport_timezone_mapping)
dropped_data['DESTINATION_AIRPORT_TZ'] = dropped_data['DESTINATION_AIRPORT'].map(airport_timezone_mapping)

# Adjust delay calculations for edge cases
def calculate_departure_delay(departure_time, scheduled_departure):
    if departure_time > 1300 and scheduled_departure < 300:
        return (departure_time - 1440) - scheduled_departure
    else:
        return departure_time - scheduled_departure

def calculate_arrival_delay(arrival_time, scheduled_arrival):
    if arrival_time < 300 and scheduled_arrival > 1300:
        return arrival_time + 1440 - scheduled_arrival
    else:
        return arrival_time - scheduled_arrival

# AIR_TIME = (WHEELS_OFF - WHEELS_ON) + (DESTINATION_TZ - SOURCE_TZ) * 60
dropped_data.loc[:, 'AIR_TIME'] = (dropped_data['WHEELS_ON'] - dropped_data['WHEELS_OFF']) + ((dropped_data['ORIGIN_AIRPORT_TZ'] - dropped_data['DESTINATION_AIRPORT_TZ']) * 60)

# ELAPSED_TIME = AIR_TIME + TAXI_OUT + TAXI_IN
dropped_data.loc[:, 'ELAPSED_TIME'] = dropped_data['AIR_TIME'] + dropped_data['TAXI_OUT'] + dropped_data['TAXI_IN']

# Calculate DEPARTURE_DELAY and ARRIVAL_DELAY using the custom functions
dropped_data.loc[:, 'DEPARTURE_DELAY'] = dropped_data.apply(lambda row: calculate_departure_delay(row['DEPARTURE_TIME'], row['SCHEDULED_DEPARTURE']), axis=1)
dropped_data.loc[:, 'ARRIVAL_DELAY'] = dropped_data.apply(lambda row: calculate_arrival_delay(row['ARRIVAL_TIME'], row['SCHEDULED_ARRIVAL']), axis=1)

# Measure start time
start_time = time.time()

# Define which columns to use for encoding and scaling
categorical_cols = ["MONTH", "DAY", "DAY_OF_WEEK", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
numerical_cols = ["SCHEDULED_DEPARTURE", "DEPARTURE_DELAY", "TAXI_OUT", "WHEELS_OFF", "AIR_TIME", "DISTANCE", "WHEELS_ON", "TAXI_IN", "SCHEDULED_ARRIVAL"]

# Apply StandardScaler to numerical columns
scaler = StandardScaler()
numerical_scaled = pd.DataFrame(scaler.fit_transform(dropped_data[numerical_cols]), columns=numerical_cols)

# Apply OneHotEncoder to categorical columns
encoder = OneHotEncoder(sparse_output=False, drop='first')
categorical_encoded = pd.DataFrame(encoder.fit_transform(dropped_data[categorical_cols]),
                                   columns=encoder.get_feature_names_out(categorical_cols))

# Combine scaled numerical data, encoded categorical data, and target (ARRIVAL_DELAY)
final_data = pd.concat([numerical_scaled, categorical_encoded, dropped_data["ARRIVAL_DELAY"].reset_index(drop=True)], axis=1)

# Measure end time
end_time = time.time()
running_time = end_time - start_time
print(f"Total encoding: {running_time:.2f} seconds")

# #Optionally, saving the processed data to a new CSV file
# final_data.to_csv("Processed_flight_data.csv", index=False)

# #Display the first few rows of the processed dataset
# print(final_data.head())

# Split data into features and target
x = final_data.drop(columns=["ARRIVAL_DELAY"])  # Features
y = final_data["ARRIVAL_DELAY"]  # Target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=108)

# Initialize the Linear Regression model
model = LinearRegression()

# Measure start time for training
train_start_time = time.time()

# Train the Linear Regression model
model.fit(X_train, y_train)

# Measure end time for training
train_end_time = time.time()
train_time = train_end_time - train_start_time
print(f"Total training time: {train_time:.2f} seconds")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Output the results
print(f"R-squared (R²): {r2:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Save the trained model to a pickle file
with open("model/2015-LinearRegression_FlightDelay.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as '2015-LinearRegression_FlightDelay.pkl'")
