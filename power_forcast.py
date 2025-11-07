"""
Solar Power Prediction System

This module provides functionality to predict solar power generation based on
weather forecasts using a Feed-Forward Neural Network (FFN) estimator.

Author: Your Name
Date: 2025-11-06
"""

import logging
import pandas as pd
from typing import Dict, Any
from datetime import datetime, timedelta
from pathlib import Path

from weather_forecasting.forecast import weather_forecast
from FFN.FFN_estimator import SolarPowerEstimator


# Configure logging
logger = logging.getLogger("power_forecast")
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter("%(levelname)s - %(message)s")
ch.setFormatter(formatter)

# Add handler if it hasn't been added yet
if not logger.hasHandlers():
    logger.addHandler(ch)


class SolarPowerPredictor:
    """
    A class to predict solar power generation based on weather forecasts.
    
    Attributes:
        estimator: The FFN-based solar power estimator model
        latitude: Location latitude
        longitude: Location longitude
    """
    
    def __init__(self, latitude: float, longitude: float):
        """
        Initialize the Solar Power Predictor.
        
        Args:
            latitude: Latitude of the solar installation (-90 to 90)
            longitude: Longitude of the solar installation (-180 to 180)
            
        Raises:
            ValueError: If latitude or longitude are out of valid range
        """
        self._validate_coordinates(latitude, longitude)
        
        self.latitude = latitude
        self.longitude = longitude
        self.estimator = None
        
        logger.info(f"Initialized SolarPowerPredictor for location: ({latitude}, {longitude})")
    
    @staticmethod
    def _validate_coordinates(latitude: float, longitude: float) -> None:
        if not -90 <= latitude <= 90:
            raise ValueError(f"Latitude must be between -90 and 90, got {latitude}")
        if not -180 <= longitude <= 180:
            raise ValueError(f"Longitude must be between -180 and 180, got {longitude}")
    
    def _initialize_estimator(self) -> None:
        """Initialize the FFN estimator if not already initialized."""
        if self.estimator is None:
            try:
                self.estimator = SolarPowerEstimator()
                logger.info("FFN estimator initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize FFN estimator: {e}")
                raise
    
    def predict_tomorrow(self) -> Dict[str, Any]:
        """
        Predict solar power generation for tomorrow.
        
        Returns:
            Dictionary containing:
                - estimated_power: Predicted power output
                - forecast_data: Weather forecast data used
                - timestamp: Prediction timestamp
                - location: Coordinates used
                
        Raises:
            RuntimeError: If weather forecast or prediction fails
        """
        try:
            logger.info("Fetching weather forecast for tomorrow...\n")
            tomorrow_weather_forecast = weather_forecast(self.latitude, self.longitude, verbose=0)

            # Remove the 'datetime' column
            tomorrow_weather_forecast.drop(columns=['datetime'], inplace=True)
            
            if tomorrow_weather_forecast is None:
                raise RuntimeError("Weather forecast returned no data")
            
            logger.info("Weather forecast retrieved successfully")
            
            # Initialize estimator if needed
            self._initialize_estimator()
            
            logger.info("Calculating power estimation...")
            estimated_power = self.estimator.predict(data=tomorrow_weather_forecast)

            # Round to 2 decimals and set values below 0.1 to 0
            estimated_power = [
                round(p, 2) if p >= 0.1 else 0 for p in estimated_power
            ]

            result = {
                'estimated_power': estimated_power,
                'forecast_data': tomorrow_weather_forecast,
                'forecast_date': (datetime.now() + timedelta(days=1)).date().isoformat()
            }
            
            logger.info(f"Prediction completed successfully: {estimated_power}")
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate solar power prediction: {e}") from e



def export_to_csv(data: pd.DataFrame, output_path: str) -> None:
    """
    Export prediction results to CSV file.
    
    Args:
        data: DataFrame containing forecast and prediction data
        output_path: Path to output CSV file
        
    Raises:
        IOError: If file cannot be written
    """
    try:
        # Create directory if it doesn't exist
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Export to CSV
        data.to_csv(output_path, index=False, float_format='%.2f')
        logger.info(f"Data successfully exported to: {output_path}")
        logger.info(f"Total records written: {len(data)}")
        
    except Exception as e:
        logger.error(f"Failed to export data to CSV: {e}")
        raise IOError(f"Cannot write to {output_path}: {e}") from e



def main():
    """
    Main entry point for the script.
    Demonstrates usage of the Solar Power Predictor.
    """
    # Example coordinates (Trento, Italy)
    LATITUDE = 46.0664
    LONGITUDE = 11.1257
    OUTPUT = 'forecast.csv'
    
    try:
        print("\n"+"=" * 60)
        print("Solar Power Prediction System - Started")
        print("=" * 60)
        
        # Initialize predictor
        predictor = SolarPowerPredictor(LATITUDE, LONGITUDE)
        
        # Get tomorrow's prediction
        result = predictor.predict_tomorrow()
        
        # Merge forecast data with predictions
        merge_df = result['forecast_data'].copy()
        merge_df['estimated_power'] = result['estimated_power']

        export_to_csv(merge_df, OUTPUT)
        
        logger.info("Solar Power Prediction System - Completed Successfully")
        
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"Prediction failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()