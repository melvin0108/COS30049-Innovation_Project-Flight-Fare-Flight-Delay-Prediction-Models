import pandas as pd
from timezonefinder import TimezoneFinder
from zoneinfo import ZoneInfo
from datetime import datetime
import os

# Load the data
airport_data = os.path.join("dataset", "airports.csv")
airport_data = pd.read_csv(airport_data)

# Initialize TimezoneFinder instance
tf = TimezoneFinder()

# Function to get the UTC offset from latitude and longitude
def get_utc_offset(lat, lng):
    try:
        # Get the time zone string from latitude and longitude
        timezone_str = tf.timezone_at(lat=lat, lng=lng)
        if timezone_str:
            # Use zoneinfo to calculate the UTC offset in hours
            tz = ZoneInfo(timezone_str)
            now = datetime.now(tz)
            utc_offset = now.utcoffset().total_seconds() / 3600  # Convert from seconds to hours
            return utc_offset
        else:
            return "Unknown"
    except Exception as e:
        print(f"Error fetching timezone for lat: {lat}, lng: {lng} - {e}")
        return "Unknown"

# Apply the function to get the UTC offset for each airport
airport_data['UTC_OFFSET'] = airport_data.apply(lambda row: get_utc_offset(row['LATITUDE'], row['LONGITUDE']), axis=1)

# Save the updated dataframe
airport_data.to_csv("dataset//airports_with_utc_offsets.csv", index=False)

# Print the first few rows to verify
print(airport_data[['AIRPORT', 'LATITUDE', 'LONGITUDE', 'UTC_OFFSET']].head())


