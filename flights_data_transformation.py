import pandas as pd
import os

# Load the data
airports_data = os.path.join("dataset", "airports_with_utc_offsets.csv")
airports_data = pd.read_csv(airports_data)

flights_data = os.path.join("dataset", "flights.csv")
flights_data = pd.read_csv(flights_data)

# Dropping rows with missing values in relevant columns
dropped_flight_data = flights_data.dropna(subset=[
    'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT', 
    'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 
    'DISTANCE', 'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL', "ARRIVAL_TIME", "ARRIVAL_DELAY"
])

# Function to convert time points (SCHEDULED_DEPARTURE, DEPARTURE_TIME, etc.) to minutes from midnight
def convert_time_to_minutes(time):
    time = f"{int(time):04d}"  # Ensure the time is 4 digits
    hours, minutes = int(time[:2]), int(time[2:])
    return hours * 60 + minutes

# Convert time-related columns to minutes from midnight using direct assignment
time_columns = ['SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'WHEELS_OFF', 'WHEELS_ON', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME']
for col in time_columns:
    dropped_flight_data[col] = dropped_flight_data[col].apply(convert_time_to_minutes)

#dropping redundant columns
column_to_drop = ["AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY", "YEAR", "ELAPSED_TIME", "SCHEDULED_TIME" , "DEPARTURE_TIME", "ARRIVAL_TIME", "FLIGHT_NUMBER", "AIRLINE"]
dropped_flight_data = dropped_flight_data.drop(columns= column_to_drop)

# Randomly drop half of the rows
dropped_flight_data = dropped_flight_data.sample(frac=0.05, random_state=108).reset_index(drop=True)

# Save the updated dataframe
dropped_flight_data.to_csv("dataset/2015-Cleaned_flight_data.csv", index=False)

# Display the first few rows to verfy
#print(dropped_flight_data.describe())
