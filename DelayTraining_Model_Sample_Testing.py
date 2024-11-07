import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from math import radians, cos, sin, asin, sqrt

# Load the saved model from the pickle file
model_path = "model/2015-LinearRegression_FlightDelay.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load the airports dataset
airports_path = "dataset/airports.csv"
airports_data = pd.read_csv(airports_path)
airport_iata_codes = airports_data['IATA_CODE'].tolist()

# Input validation functions
def validate_int(prompt, min_val=None, max_val=None):
    while True:
        try:
            value = int(input(prompt))
            if (min_val is not None and value < min_val) or (max_val is not None and value > max_val):
                raise ValueError(f"Value must be between {min_val} and {max_val}.")
            return value
        except ValueError as e:
            print(f"Invalid input: {e}")

def validate_airport(prompt):
    while True:
        airport_code = input(prompt).upper()
        if airport_code not in airport_iata_codes:
            print("Invalid airport IATA code. Please enter a valid IATA code from the airports dataset.")
        else:
            return airport_code

# Create a mapping of airport IATA code to their latitude and longitude
airport_locations = airports_data.set_index('IATA_CODE')[['LATITUDE', 'LONGITUDE']].to_dict(orient='index')

# Haversine formula to calculate distance between two latitude/longitude points
def haversine(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 3956  # Radius of Earth in miles
    return c * r


# User input
def get_user_input():
    print("Please provide the following flight information:")
    user_input = {}
    user_input["MONTH"] = validate_int("Month (1-12): ", 1, 12)
    user_input["DAY"] = validate_int("Day of the month (1-31): ", 1, 31)
    user_input["DAY_OF_WEEK"] = validate_int("Day of the week (1=Monday, 7=Sunday): ", 1, 7)
    
    while True:
        user_input["ORIGIN_AIRPORT"] = validate_airport("Origin Airport IATA code: ")
        user_input["DESTINATION_AIRPORT"] = validate_airport("Destination Airport IATA code: ")
        if user_input["ORIGIN_AIRPORT"] != user_input["DESTINATION_AIRPORT"]:
            break
        else:
            print("Origin and destination airports must not be the same.")
        
    while True:
        user_input["SCHEDULED_DEPARTURE"] = validate_int("Scheduled Departure time (in minutes from midnight, e.g., 1439 for 23:59): ", 0, 1439)
        user_input["SCHEDULED_ARRIVAL"] = validate_int("Scheduled Arrival time (in minutes from midnight, e.g., 1439 for 23:59): ", 0, 1439)
        if user_input["ORIGIN_AIRPORT"] != user_input["DESTINATION_AIRPORT"]:
            break
        else:
            print("Scheduled departure and arrival times must not be the same.")

    user_input["DEPARTURE_DELAY"] = validate_int("Departure Delay (in minutes): ")
    user_input["TAXI_OUT"] = None
    user_input["WHEELS_OFF"] = None
    user_input["AIR_TIME"] = user_input["SCHEDULED_ARRIVAL"] - user_input["SCHEDULED_DEPARTURE"] 

    # Calculate distance based on the origin and destination airports' lat/long
    origin_coords = airport_locations[user_input["ORIGIN_AIRPORT"]]
    destination_coords = airport_locations[user_input["DESTINATION_AIRPORT"]]
    user_input["DISTANCE"] = haversine(
        origin_coords['LATITUDE'], origin_coords['LONGITUDE'],
        destination_coords['LATITUDE'], destination_coords['LONGITUDE']
    )

    user_input["WHEELS_ON"] = None
    user_input["TAXI_IN"] = None
    
    
    return user_input

def preprocess_input(user_input):
    # Convert the input dictionary to a DataFrame
    user_data = pd.DataFrame([user_input])
    
    #Defining categorical_cols, numerical_cols
    categorical_cols = ["MONTH", "DAY", "DAY_OF_WEEK", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
    numerical_cols = ["SCHEDULED_DEPARTURE", "DEPARTURE_DELAY", "TAXI_OUT", "WHEELS_OFF", "AIR_TIME", "DISTANCE", "WHEELS_ON", "TAXI_IN", "SCHEDULED_ARRIVAL"]
        
    # Apply OneHotEncoding to categorical columns (this should match the encoder used during training)
    encoder = OneHotEncoder(drop='first')  # Recreate the same encoder setup
    categorical_data = pd.DataFrame(encoder.fit_transform(user_data[categorical_cols]).toarray())
    
    # Apply StandardScaler to numerical columns (this should match the scaler used during training)
    scaler = StandardScaler()  # Recreate the same scaler setup
    numerical_data = pd.DataFrame(scaler.fit_transform(user_data[numerical_cols]))
    
    # Combine both numerical and categorical data for prediction
    preprocessed_data = pd.concat([numerical_data, categorical_data], axis=1)
    
    return preprocessed_data


# Main function
if __name__ == "__main__":
    # Get user input
    user_input = get_user_input()

    # Preprocess input for prediction
    preprocessed_data = preprocess_input(user_input)

    # Make the prediction using the loaded model
    prediction = model.predict(preprocessed_data)
    
    # Display the result
    print(f"Predicted ARRIVAL_DELAY: {prediction[0]:.2f} minutes")
