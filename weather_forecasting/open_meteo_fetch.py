#!/usr/bin/env python3
"""
Weather Forecast Data Collector
Fetches hourly weather forecast data for the next day (00:00 to 23:00)
and saves it to a CSV file.
"""

import requests
import csv
import pandas as pd
from datetime import datetime, timedelta
import sys
import logging

# Configure logging
logger = logging.getLogger("openmeteo")
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter("WEATHER FORECAST - OPENMETEO FETCH - %(levelname)s - %(message)s")
ch.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(ch)

    

def process_weather_data(weather_data, verbose = 1):
    """
    Process raw weather forecast data from Open-Meteo API into a clean DataFrame.

    Args:
        weather_data (dict): Raw weather data returned by fetch_weather_forecast()

    Returns:
        pd.DataFrame or None: Processed weather DataFrame, or None if data is invalid
    """
    if not weather_data or 'hourly' not in weather_data:
        logger.error("✗ No valid weather data to process", file=sys.stderr)
        return None

    hourly = weather_data['hourly']
    rows = []

    for i in range(len(hourly['time'])):
        try:
            dt = datetime.fromisoformat(hourly['time'][i])

            # Convert units
            irradiance_kj = (
                hourly['shortwave_radiation'][i] * 3.6
                if hourly['shortwave_radiation'][i] is not None
                else None
            )
            wind_ms = (
                hourly['wind_speed_10m'][i] / 3.6
                if hourly['wind_speed_10m'][i] is not None
                else None
            )

            row = {
                'datetime': dt,
                'hour': dt.hour,
                'day': dt.day,
                'month': dt.month,
                'air_temperature': hourly['temperature_2m'][i],
                'humidity': hourly['relative_humidity_2m'][i],
                'irradiance': round(irradiance_kj, 2) if irradiance_kj is not None else None,
                'pressure': hourly['surface_pressure'][i],
                'rain': hourly['rain'][i],
                'wind_direction': hourly['wind_direction_10m'][i],
                'wind_velocity': round(wind_ms, 2) if wind_ms is not None else None,
            }
            rows.append(row)
        except Exception as e:
            logger.error(f"⚠ Skipping record {i} due to error: {e}", file=sys.stderr)

    if not rows:
        logger.error("✗ No valid records processed", file=sys.stderr)
        return None

    df = pd.DataFrame(rows)
    if verbose >= 1:
        logger.info(f"✓ Processed {len(df)} records into DataFrame")
    return df

def fetch_weather_forecast(latitude=46.0664, longitude=11.1257, verbose = 1):
    """
    Fetch hourly weather forecast for the next day using Open-Meteo API.
    
    Default location: Trento, Italy
    You can change latitude and longitude to your desired location.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        verbose: Verbose level
    
    Returns:
        dict: Weather data or None if request fails
    """
    
    # Calculate tomorrow's date range
    tomorrow = datetime.now() + timedelta(days=1)
    start_date = tomorrow.strftime('%Y-%m-%d')
    end_date = start_date  # Same day for start and end
    
    # Open-Meteo API endpoint
    url = "https://api.open-meteo.com/v1/forecast"
    
    # Parameters for the API request
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': [
            'temperature_2m',           # Air temperature at 2m
            'relative_humidity_2m',     # Relative humidity at 2m
            'surface_pressure',         # Surface pressure
            'rain',                     # Rain
            'wind_speed_10m',          # Wind speed at 10m
            'wind_direction_10m',      # Wind direction at 10m
            'shortwave_radiation'      # Solar irradiance
        ],
        'start_date': start_date,
        'end_date': end_date,
        'timezone': 'auto'  # Automatically detect timezone based on coordinates
    }
    
    try:
        if verbose >= 1:
            logger.info(f"Fetching weather forecast for {start_date}...")
            logger.info(f"Location: Lat {latitude}, Lon {longitude}")
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()  # Raise exception for bad status codes
        
        data = response.json()
        if verbose >= 1:
            logger.info("✓ Data fetched successfully!")

        # Process JSON file to get a DataFrame structure
        df = process_weather_data(data, verbose)
        if verbose >= 1:
            logger.info("✓ Organized data in DataFrame")
        return df
        
    except requests.exceptions.RequestException as e:
        logger.error(f"✗ Error fetching data: {e}", file=sys.stderr)
        return None
    except ValueError as e:
        logger.error(f"✗ Error parsing JSON response: {e}", file=sys.stderr)
        return None




#=====================================================================
# Suplementary Code
#=====================================================================
def main():
    """Main function to fetch and save weather forecast."""
    
    # Change these coordinates to your desired location
    # Default: Roncafort Trento, Italy
    LATITUDE = 46.09564
    LONGITUDE = 11.10136
    
    # Output filename
    OUTPUT_FILE = 'tomorrow_forecast.csv'
    
    print("=" * 60)
    print("Weather Forecast Data Collector")
    print("=" * 60)
    print()
    
    # Fetch data
    weather_data = fetch_weather_forecast(LATITUDE, LONGITUDE, verbose = 1)
    
    if weather_data is None or weather_data.empty:
        print("✗ No valid DataFrame to save", file=sys.stderr)
        return False

    try:
        weather_data.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        print(f"✓ DataFrame saved to {OUTPUT_FILE}")
        print(f"  Total records: {len(weather_data)}")
        return True

    except IOError as e:
        print(f"✗ Error writing to file: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    sys.exit(main())