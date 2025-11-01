import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import seaborn as sns

def add_time_and_season_columns(df, timestamp_column='datetime'):
    """
    Add two columns to the dataframe:
    - 'is_day': boolean indicating if the time is during day (6-18) or night
    - 'season': string indicating the season (spring, summer, autumn, winter)
    """
    df = df.copy()
    
    # Add is_day column (True if daytime, False if nighttime)
    df['is_day'] = df[timestamp_column].dt.hour.between(6, 18)
    
    # Add season column based on month
    def get_season(month):
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:  # 9, 10, 11
            return 'autumn'
    
    df['season'] = df[timestamp_column].dt.month.apply(get_season)
    
    return df


def process_wind_data(df):
    """
    Process wind direction and velocity to create meaningful features
    """
    # Create a copy to avoid modifying original data
    df_processed = df.copy()
    
    # Convert wind direction from degrees to radians
    # Assuming wind_direction is in degrees (0-360)
    dir_rad = np.radians(df_processed['wind_direction'])
    
    
    # Wind components (recommended for meteorological data)
    # u-component: east-west wind velocity (positive = eastward)
    # v-component: north-south wind velocity (positive = northward)
    df_processed['wind_u'] = df_processed['wind_velocity'] * np.cos(dir_rad)
    df_processed['wind_v'] = df_processed['wind_velocity'] * np.sin(dir_rad)
        
    return df_processed