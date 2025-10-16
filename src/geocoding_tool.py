# File Path: Disaster_ML_Fusion/src/geocoding_tool.py

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import os

# --- Define Paths ---
# Note: This script runs from the 'src' folder, so paths use '../data/'
INPUT_PATH = "C:\\Users\\indhu\\Downloads\\archive (1)\\Disaster2021.csv"
OUTPUT_PATH = "../data/Disaster2021_Geocoded.csv"

def run_geocoding():
    """
    Attempts to geocode records where coordinates were originally missing
    using Nominatim (OpenStreetMap).
    """
    try:
        df = pd.read_csv(INPUT_PATH)
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_PATH}. Please run the EDA notebook first.")
        return

    # Identify records to check (where 'Location' is not null, and we're looking for better coordinates)
    df_to_geocode = df[df['Location'].notna()].copy()
    
    # Simple strategy: only try to geocode if location is provided
    # A more advanced check would be needed to ensure the current Lat/Lon is the imputed median.
    if df_to_geocode.empty:
        print("No valid locations found for geocoding. Skipping step.")
        df.to_csv(OUTPUT_PATH, index=False)
        return

    print(f"\nStarting Geocoding for {len(df_to_geocode)} records...")

    # Initialize Nominatim Geocoder
    geolocator = Nominatim(user_agent="disaster_risk_fusion_project_v1")
    # Rate limit: max 1 call per second to comply with Nominatim policy
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.05)

    # Prepare query column
    df_to_geocode['Query'] = df_to_geocode['Location'] + ', ' + df_to_geocode['Country']
    
    # Function to apply geocoding
    def attempt_geocode(query):
        try:
            location = geocode(query)
            if location:
                return location.latitude, location.longitude
        except Exception:
            return None, None
        return None, None

    # Apply geocoding (This step will be SLOW)
    results = df_to_geocode['Query'].apply(lambda x: attempt_geocode(x))
    
    # Unpack results
    geocoded_coords = pd.DataFrame(results.tolist(), columns=['Lat_New', 'Lon_New'], index=df_to_geocode.index)
    
    # --- Merge and Consolidate Coordinates ---
    df = df.merge(geocoded_coords, left_index=True, right_index=True, how='left')
    
    # Fill original Latitude/Longitude with new geocoded values only where geocoding was successful
    df['Latitude'] = df['Lat_New'].fillna(df['Latitude'])
    df['Longitude'] = df['Lon_New'].fillna(df['Longitude'])
    
    df.drop(columns=['Lat_New', 'Lon_New'], inplace=True)
    
    print("\nGeocoding complete. Saving data.")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Geocoded data saved to: {OUTPUT_PATH}")

if __name__ == '__main__':
    run_geocoding()