#!/usr/bin/env python3
"""
Historical Weather Data Collector
Fetches historical hourly weather data from Open-Meteo Archive API.
Useful for building prediction models and training datasets.
"""

import requests
import csv
from datetime import datetime, timedelta
import sys
import argparse
import time

def fetch_forecast_historical_weather(latitude, longitude, start_date, end_date):
    """
    Fetch forecast historical hourly weather data using Open-Meteo Historical API.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        start_date: Start date (YYYY-MM-DD format or datetime object)
        end_date: End date (YYYY-MM-DD format or datetime object)
    
    Returns:
        dict: Weather data or None if request fails
    """
    
    # Convert datetime objects to strings if needed
    if isinstance(start_date, datetime):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, datetime):
        end_date = end_date.strftime('%Y-%m-%d')
    
    # Open-Meteo Forecast Historical API endpoint
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    
    # Parameters for the API request
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': [
            'temperature_2m',
            'relative_humidity_2m',
            'surface_pressure',
            'rain',
            'wind_speed_10m',
            'wind_direction_10m',
            'shortwave_radiation'
        ],
        'timezone': 'auto'
    }
    
    try:
        print(f"Fetching historical forecasted data from {start_date} to {end_date}...")
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Check if we got valid data
        if 'hourly' in data and 'time' in data['hourly']:
            num_records = len(data['hourly']['time'])
            print(f"✓ Fetched {num_records} hourly records")
            return data
        else:
            print("✗ No data returned from API", file=sys.stderr)
            return None
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching historical data: {e}", file=sys.stderr)
        return None
    except ValueError as e:
        print(f"✗ Error parsing JSON response: {e}", file=sys.stderr)
        return None


def fetch_historical_batch(latitude, longitude, start_date, end_date, batch_days=90):
    """
    Fetch historical data in batches to handle large date ranges.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        start_date: Start date (datetime object)
        end_date: End date (datetime object)
        batch_days: Number of days per batch (default: 90)
    
    Returns:
        list: List of all processed rows
    """
    
    all_rows = []
    current_start = start_date
    
    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=batch_days - 1), end_date)
        
        print(f"\nBatch: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")
        
        weather_data = fetch_forecast_historical_weather(latitude, longitude, current_start, current_end)
        
        if weather_data:
            rows = process_weather_data(weather_data)
            all_rows.extend(rows)
            print(f"✓ Processed {len(rows)} records")
        else:
            print(f"✗ Failed to fetch batch")
        
        # Move to next batch
        current_start = current_end + timedelta(days=1)
        
        # Small delay to be respectful to the API
        if current_start <= end_date:
            time.sleep(0.5)
    
    return all_rows


def process_weather_data(weather_data):
    """
    Process weather data and convert units.
    
    Args:
        weather_data: Raw weather data from API
    
    Returns:
        list: Processed rows ready for CSV
    """
    
    if not weather_data or 'hourly' not in weather_data:
        return []
    
    hourly = weather_data['hourly']
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
            'datetime': dt.strftime('%Y-%m-%d %H:%M:%S'),
            'month': dt.month,
            'day': dt.day,
            'hour': dt.hour,
            'air_temperature': hourly['temperature_2m'][i],
            'humidity': hourly['relative_humidity_2m'][i],
            'irradiance': round(irradiance_kj, 2) if irradiance_kj is not None else None,
            'pressure': hourly['surface_pressure'][i],
            'rain': hourly['rain'][i],
            'wind_direction': hourly['wind_direction_10m'][i],
            'wind_velocity': round(wind_ms, 2) if wind_ms is not None else None
        }
        rows.append(row)
    
    return rows


def save_to_csv(rows, filename='historical_forecast.csv'):
    """
    Save weather data to CSV file.
    
    Args:
        rows: List of data rows
        filename: Output CSV filename
    
    Returns:
        bool: Success status
    """
    
    if not rows:
        print("✗ No valid data to save", file=sys.stderr)
        return False
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'datetime', 'month', 'day', 'hour', 'air_temperature', 'humidity', 'irradiance', 'pressure',
                'rain', 'wind_direction', 'wind_velocity'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"\n✓ Data saved to {filename}")
        print(f"  Total records: {len(rows)}")
        return True
        
    except IOError as e:
        print(f"✗ Error writing to file: {e}", file=sys.stderr)
        return False


def parse_date(date_str):
    """Parse date string to datetime object."""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        print(f"✗ Invalid date format: {date_str}. Use YYYY-MM-DD", file=sys.stderr)
        sys.exit(1)


def main():
    """Main function to fetch and save historical weather data."""
    
    parser = argparse.ArgumentParser(
        description='Fetch historical weather data from Open-Meteo Archive API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Fetch last 30 days
            python %(prog)s --days 30
            
            # Fetch last year
            python %(prog)s --days 365
            
            # Fetch specific date range
            python %(prog)s --start 2023-01-01 --end 2023-12-31
            
            # Fetch last 90 days for custom location (Rome)
            python %(prog)s --days 90 --lat 41.9028 --lon 12.4964 --output rome_historical.csv
            
            # Fetch multiple years for model training
            python %(prog)s --start 2020-01-01 --end 2024-12-31 --output training_data.csv

            Data Units:
            - air_temperature: °C
            - humidity: % (0-100)
            - irradiance: kJ/m² (converted from W/m²)
            - pressure: hPa (hectopascals)
            - rain: mm (precipitation)
            - wind_direction: degrees (0-360, 0° = North)
            - wind_velocity: m/s (converted from km/h)
                    """
    )
    
    # Date range options
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument('--days', type=int,
                           help='Number of past days to fetch (from yesterday backwards)')
    date_group.add_argument('--start', type=str,
                           help='Start date (YYYY-MM-DD), requires --end')
    
    parser.add_argument('--end', type=str,
                       help='End date (YYYY-MM-DD), used with --start')
    
    # Location options
    parser.add_argument('--lat', type=float, default=46.0664,
                       help='Latitude (default: 46.0664, Trento)')
    parser.add_argument('--lon', type=float, default=11.1257,
                       help='Longitude (default: 11.1257, Trento)')
    
    # Output options
    parser.add_argument('--output', type=str, default='historical_weather.csv',
                       help='Output CSV filename (default: historical_weather.csv)')
    
    parser.add_argument('--batch-days', type=int, default=90,
                       help='Days per API batch (default: 90, max recommended: 365)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.start and not args.end:
        parser.error("--start requires --end")
    if args.end and not args.start:
        parser.error("--end requires --start")
    
    print("=" * 70)
    print("Historical Weather Data Collector")
    print("=" * 70)
    print(f"\nLocation: Lat {args.lat}, Lon {args.lon}")
    
    # Determine date range
    if args.days:
        end_date = datetime.now() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=args.days - 1)
        print(f"Fetching last {args.days} days")
    else:
        start_date = parse_date(args.start)
        end_date = parse_date(args.end)
        
        if start_date > end_date:
            print("✗ Error: start date must be before end date", file=sys.stderr)
            return 1
        
        if end_date > datetime.now():
            print("✗ Error: end date cannot be in the future", file=sys.stderr)
            return 1
    
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Calculate total days
    total_days = (end_date - start_date).days + 1
    print(f"Total days: {total_days}")
    print(f"Expected records: ~{total_days * 24} (hourly)")
    
    # Fetch data
    print("\n" + "=" * 70)
    print("Fetching data...")
    print("=" * 70)
    
    all_rows = fetch_historical_batch(
        args.lat, args.lon, start_date, end_date, args.batch_days
    )
    
    if all_rows:
        # Save to CSV
        print("\n" + "=" * 70)
        print("Saving data...")
        print("=" * 70)
        
        success = save_to_csv(all_rows, args.output)
        
        if success:
            print("\n" + "=" * 70)
            print("✓ Process completed successfully!")
            print("=" * 70)
            print(f"\nYou can now use '{args.output}' for model training and analysis.")
            return 0
    
    print("\n" + "=" * 70)
    print("✗ Process failed!")
    print("=" * 70)
    return 1


if __name__ == "__main__":
    sys.exit(main())