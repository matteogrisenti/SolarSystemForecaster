#!/usr/bin/env python3
"""
Weather Forecast Data Collector
Fetches hourly weather forecast data for the next day (00:00 to 23:00)
and saves it to a CSV file.
"""

import requests
import csv
from datetime import datetime, timedelta
import sys

def fetch_weather_forecast(latitude=46.0664, longitude=11.1257):
    """
    Fetch hourly weather forecast for the next day using Open-Meteo API.
    
    Default location: Trento, Italy
    You can change latitude and longitude to your desired location.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
    
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
        print(f"Fetching weather forecast for {start_date}...")
        print(f"Location: Lat {latitude}, Lon {longitude}")
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()  # Raise exception for bad status codes
        
        data = response.json()
        print("✓ Data fetched successfully!")
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching data: {e}", file=sys.stderr)
        return None
    except ValueError as e:
        print(f"✗ Error parsing JSON response: {e}", file=sys.stderr)
        return None




#=====================================================================
# Suplementary Code
#=====================================================================
def save_to_csv(weather_data, filename='tomorrow_forecast.csv'):
    """
    Save weather forecast data to CSV file.
    
    Args:
        weather_data: Weather data from API
        filename: Output CSV filename
    """
    
    if not weather_data or 'hourly' not in weather_data:
        print("✗ No valid data to save", file=sys.stderr)
        return False
    
    hourly = weather_data['hourly']
    
    # Prepare CSV data
    rows = []
    for i in range(len(hourly['time'])):
        # Parse datetime
        dt = datetime.fromisoformat(hourly['time'][i])
        
        # Convert units
        # Irradiance: W/m² to kJ/m² (1 hour × W/m² = 3.6 kJ/m²)
        irradiance_kj = hourly['shortwave_radiation'][i] * 3.6 if hourly['shortwave_radiation'][i] is not None else None
        
        # Wind velocity: km/h to m/s
        wind_ms = hourly['wind_speed_10m'][i] / 3.6 if hourly['wind_speed_10m'][i] is not None else None
        
        row = {
            'hour': dt.hour,
            'day': dt.day,
            'month': dt.month,
            'air_temperature': hourly['temperature_2m'][i],
            'humidity': hourly['relative_humidity_2m'][i],
            'irradiance': round(irradiance_kj, 2) if irradiance_kj is not None else None,
            'pressure': hourly['surface_pressure'][i],
            'rain': hourly['rain'][i],
            'wind_direction': hourly['wind_direction_10m'][i],
            'wind_velocity': round(wind_ms, 2) if wind_ms is not None else None
        }
        rows.append(row)
    
    # Write to CSV
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'hour', 'day', 'month', 'air_temperature', 'humidity',
                'irradiance', 'pressure', 'rain', 'wind_direction', 'wind_velocity'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"✓ Data saved to {filename}")
        print(f"  Total records: {len(rows)}")
        return True
        
    except IOError as e:
        print(f"✗ Error writing to file: {e}", file=sys.stderr)
        return False


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
    weather_data = fetch_weather_forecast(LATITUDE, LONGITUDE)
    
    if weather_data:
        # Save to CSV
        success = save_to_csv(weather_data, OUTPUT_FILE)
        
        if success:
            print()
            print("=" * 60)
            print("✓ Process completed successfully!")
            print("=" * 60)
            return 0
    
    print()
    print("=" * 60)
    print("✗ Process failed!")
    print("=" * 60)
    return 1


if __name__ == "__main__":
    sys.exit(main())