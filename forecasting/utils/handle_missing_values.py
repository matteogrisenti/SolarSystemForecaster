import pandas as pd

def identify_gaps(df, missing_mask):
    """
    Identify continuous gaps in the data
    
    Returns list of dictionaries with gap information:
    - start: start timestamp
    - end: end timestamp  
    - length: gap length in hours
    """
    gaps = []
    in_gap = False
    gap_start = None
    
    for idx, is_missing in zip(df.index, missing_mask):
        if is_missing and not in_gap:
            # Start of new gap
            gap_start = idx
            in_gap = True
        elif not is_missing and in_gap:
            # End of gap
            gap_end = df.index[df.index.get_loc(idx) - 1]
            gaps.append({
                'start': gap_start,
                'end': gap_end,
                'length': int((gap_end - gap_start).total_seconds() / 3600) + 1
            })
            in_gap = False
    
    # Handle case where data ends with a gap
    if in_gap:
        gaps.append({
            'start': gap_start,
            'end': df.index[-1],
            'length': int((df.index[-1] - gap_start).total_seconds() / 3600) + 1
        })
    
    return gaps


def handle_solar_missing_values_smart(
    df, 
    timestamp_column='datetime',
    short_gap_threshold=6,  # hours - gaps <= this will be interpolated
    long_gap_threshold=240  # hours (10 days) - gaps > this will be filled with 0
):
    """
    Handle missing values intelligently for solar power data
    
    Strategy:
    ---------
    - Short gaps (≤ threshold hours): Interpolate weather features + power
    - Long gaps (> threshold hours): Fill all values with 0 (system likely offline)
    
    Parameters:
    -----------
    short_gap_threshold : int
        Maximum gap length (in hours) to be considered "short" (default: 6 hours)
    long_gap_threshold : int
        Minimum gap length (in hours) to be considered "long" (default: 240 = 10 days)
    """
    print("="*60)
    print("SMART MISSING VALUE HANDLING FOR SOLAR DATA")
    print("="*60)

    # Load and prepare data
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    df = df.sort_values(timestamp_column).reset_index(drop=True)
    
    print(f"\nOriginal data points: {len(df)}")
    
    # Set timestamp index
    df = df.set_index(timestamp_column)
    
    # Full hourly datetime range
    start_time = df.index.min()
    end_time = df.index.max()
    full_range = pd.date_range(start=start_time, end=end_time, freq='H')
    
    expected_rows = len(full_range)
    missing_rows = expected_rows - len(df)
    
    print(f"Expected data points (hourly): {expected_rows}")
    print(f"Missing data points to create: {missing_rows}")
    print(f"Percentage missing: {(missing_rows/expected_rows)*100:.2f}%")

    # Reindex to hourly frequency
    df_hourly = df.reindex(full_range)
    
    # Identify newly created (missing) rows
    new_rows_mask = df_hourly.isnull().all(axis=1)

    print("\n" + "-"*60)
    print("ANALYZING GAP STRUCTURE")
    print("-"*60)
    
    # Identify gaps and their lengths
    gaps = identify_gaps(df_hourly, new_rows_mask)
    
    short_gaps = [g for g in gaps if g['length'] <= short_gap_threshold]
    medium_gaps = [g for g in gaps if short_gap_threshold < g['length'] <= long_gap_threshold]
    long_gaps = [g for g in gaps if g['length'] > long_gap_threshold]
    
    print(f"\nShort gaps (≤ {short_gap_threshold}h): {len(short_gaps)} gaps, "
          f"{sum(g['length'] for g in short_gaps)} total hours")
    print(f"Medium gaps ({short_gap_threshold}h - {long_gap_threshold}h): {len(medium_gaps)} gaps, "
          f"{sum(g['length'] for g in medium_gaps)} total hours")
    print(f"Long gaps (> {long_gap_threshold}h): {len(long_gaps)} gaps, "
          f"{sum(g['length'] for g in long_gaps)} total hours")
    
    print("\n" + "-"*60)
    print("FILLING STRATEGY")
    print("-"*60)
    
    # Define column groups
    weather_features = ['air_temperature', 'humidity', 'irradiance', 
                       'pressure', 'rain', 'wind_direction', 'wind_velocity']
    target = 'power'
    
    # Available columns (in case some are missing)
    weather_features = [col for col in weather_features if col in df_hourly.columns]
    
    # Step 1: Handle LONG gaps - fill with 0
    print(f"\n1. Long gaps (>{long_gap_threshold}h): Filling with 0 (system likely offline)")
    for gap in long_gaps:
        df_hourly.loc[gap['start']:gap['end'], :] = 0
    
    # Step 2: Handle MEDIUM gaps - fill with 0 (too long to interpolate reliably)
    print(f"2. Medium gaps ({short_gap_threshold}h-{long_gap_threshold}h): Filling with 0 (gap too large for interpolation)")
    for gap in medium_gaps:
        df_hourly.loc[gap['start']:gap['end'], :] = 0
    
    # Step 3: Handle SHORT gaps - interpolate
    print(f"3. Short gaps (≤{short_gap_threshold}h): Using interpolation")
    print(f"   - Weather features: Linear interpolation")
    print(f"   - Power: Linear interpolation (limited by irradiance)")
    
    # Interpolate weather features for short gaps
    for col in weather_features:
        df_hourly[col] = df_hourly[col].interpolate(
            method='linear', 
            limit=short_gap_threshold,  # Don't interpolate beyond threshold
            limit_area='inside'  # Only interpolate between valid values
        )
    
    # Interpolate power for short gaps
    if target in df_hourly.columns:
        df_hourly[target] = df_hourly[target].interpolate(
            method='linear',
            limit=short_gap_threshold,
            limit_area='inside'
        )
        
        # Ensure power is 0 when irradiance is very low (nighttime)
        if 'irradiance' in df_hourly.columns:
            night_mask = df_hourly['irradiance'] < 10  # threshold for nighttime
            df_hourly.loc[night_mask, target] = 0
    
    # Final safety: fill any remaining NaNs with 0
    remaining_before = df_hourly.isnull().sum().sum()
    if remaining_before > 0:
        print(f"\n4. Filling {remaining_before} remaining NaN values with 0")
        df_hourly = df_hourly.fillna(0)
    
    # Reset index
    df_hourly = df_hourly.reset_index().rename(columns={'index': timestamp_column})
    
    print("\n" + "-"*60)
    print("RESULTS")
    print("-"*60)
    print(f"Final data points: {len(df_hourly)}")
    print(f"Remaining missing values: {df_hourly.isnull().sum().sum()}")
    print("\nMissing values handled successfully! ✓")
    
    return df_hourly