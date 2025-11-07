"""
Solar Power Production Estimator
=================================
This module provides a class to load a trained FNN model and make predictions
on new data, including proper input scaling.

Author: Your Name
Date: 2025
"""

import os 
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from typing import Union, Dict, List

import warnings
warnings.filterwarnings('ignore')


class SolarPowerEstimator:
    """
    Estimator for solar power production using a trained FNN model.
    
    This class handles:
    - Loading the trained model
    - Loading and applying the scaler
    - Input validation
    - Prediction with proper scaling
    """
    
    def __init__(self):
        """
        Initialize the Solar Power Estimator.
        """
        base_dir = os.path.dirname(__file__)  # folder containing FFN_estimator.py
        self.model_path = os.path.join(base_dir, 'train/v2/fnn_solar_final.h5')
        self.scaler_path = os.path.join(base_dir, 'dataset/scaler.pkl')

        self.model = None
        self.scaler = None
        
        self.feature_names = [
            'hour',
            'day',
            'month',
            'air_temperature',
            'humidity',
            'irradiance',
            'pressure',
            'rain',
            'wind_direction',
            'wind_velocity'
        ]
        
        self._load_model()
        self._load_scaler()


    def _load_model(self):
        """Load the trained Keras model."""
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"✅ Model loaded successfully from {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {str(e)}")
    

    def _load_scaler(self):
        try:
            self.scaler = joblib.load(self.scaler_path)
            # print(getattr(self.scaler, 'feature_names_in_', None))
            print(f"✅ Scaler loaded successfully from {os.path.abspath(self.scaler_path)}")
        except Exception as e:
            raise RuntimeError(f"Failed to load scaler from {self.scaler_path}: {str(e)}")
    

    def _validate_input(self, data: Union[Dict, pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Validate and convert input data to numpy array.
        
        Args:
            data: Input data as dict, DataFrame, or numpy array
            
        Returns:
            Validated numpy array
        """
        if isinstance(data, dict):
            # Convert dict to array in correct order
            try:
                array = np.array([data[feat] for feat in self.feature_names])
                return array.reshape(1, -1)
            except KeyError as e:
                raise ValueError(f"Missing required feature: {str(e)}")
                
        elif isinstance(data, pd.DataFrame):
            # Ensure correct column order
            try:
                return data[self.feature_names].values
            except KeyError as e:
                raise ValueError(f"Missing required column: {str(e)}")
                
        elif isinstance(data, np.ndarray):
            # Validate shape
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"Expected {len(self.feature_names)} features, got {data.shape[1]}"
                )
            return data
        else:
            raise TypeError(
                "Input must be dict, pandas DataFrame, or numpy array"
            )

    def predict(self, 
                hour: int = None,
                day: int = None,
                month: int = None,
                air_temperature: float = None,
                humidity: float = None,
                irradiance: float = None,
                pressure: float = None,
                rain: float = None,
                wind_direction: float = None,
                wind_velocity: float = None,
                data: Union[Dict, pd.DataFrame, np.ndarray] = None) -> Union[float, np.ndarray]:
        """
        Predict solar power production.
        
        You can provide inputs in two ways:
        1. Individual parameters (for single prediction)
        2. Using 'data' parameter (for batch prediction)
        
        Args:
            hour: hour of the prediction 0-24
            day: day of the prediction 1-31
            month: month of the prediction 1-12
            air_temperature: Air temperature in Celsius
            humidity: Relative humidity (%)
            irradiance: Solar irradiance (W/m²)
            pressure: Atmospheric pressure (hPa)
            rain: Rainfall (mm)
            wind_direction: Wind direction (degrees)
            wind_velocity: Wind speed (m/s)
            data: Alternative input as dict, DataFrame, or array
            
        Returns:
            Predicted power (kW) - float for single prediction, array for batch
            
        Examples:
            # Single prediction with individual parameters
            power = estimator.predict(
                hour = 12,
                day = 3,
                month = 1,
                air_temperature=30.12,
                humidity=39.25,
                irradiance=831.5,
                pressure=986.85,
                rain=0.0,
                wind_direction=179.83,
                wind_velocity=2.6
            )
            
            # Single prediction with dict
            power = estimator.predict(data={
                'hour': 12,
                'day': 3,
                'month': 1,
                'air_temperature': 30.12,
                'humidity': 39.25,
                'irradiance': 831.5,
                'pressure': 986.85,
                'rain': 0.0,
                'wind_direction': 179.83,
                'wind_velocity': 2.6
            })
            
            # Batch prediction with DataFrame
            df = pd.DataFrame({...})
            powers = estimator.predict(data=df)
        """
        # Handle input
        if data is not None:
            X = self._validate_input(data)
        else:
            # Check if all individual parameters are provided
            params_dict = {
                'hour': hour,
                'day': day,
                'month': month,
                'air_temperature': air_temperature,
                'humidity': humidity,
                'irradiance': irradiance,
                'pressure': pressure,
                'rain': rain,
                'wind_direction': wind_direction,
                'wind_velocity': wind_velocity
            }

            # Debug: print all parameters
            print("Debug: Individual parameters provided:")
            for name, value in params_dict.items():
                print(f"  {name}: {value}")
            
            # Find which parameters are missing
            missing_params = [name for name, value in params_dict.items() if value is None]
            
            if missing_params:
                raise ValueError(
                    f"Missing required parameter(s): {', '.join(missing_params)}. "
                    f"Either provide all individual parameters or use the 'data' parameter."
                )
            
            X = pd.DataFrame([params_dict])
        
        # Scale the input
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        predictions = self.model.predict(X_scaled, verbose=0)
        
        # Return scalar for single prediction, array for batch
        if len(predictions) == 1:
            return float(predictions[0][0])
        else:
            return predictions.flatten()
    

    def predict_from_csv(self, csv_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Make predictions on data from a CSV file.
        
        Args:
            csv_path: Path to input CSV file
            output_path: Optional path to save predictions (CSV)
            
        Returns:
            DataFrame with original data and predictions
        """
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Make predictions
        predictions = self.predict(data=df)
        
        # Add predictions to dataframe
        df['predicted_power'] = predictions
        
        # Save if output path provided
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"✅ Predictions saved to {output_path}")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Return the list of required feature names in order."""
        return self.feature_names.copy()
    
    def get_model_info(self) -> Dict:
        """Return information about the loaded model."""
        return {
            'model_path': self.model_path,
            'scaler_path': self.scaler_path,
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'total_params': self.model.count_params(),
            'feature_names': self.feature_names
        }
