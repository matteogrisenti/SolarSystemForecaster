import sys, os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from FFN_estimator import SolarPowerEstimator

def main():
    """
    Example usage of the SolarPowerEstimator.
    """
    print("="*60)
    print("Solar Power Production Estimator - Example Usage")
    print("="*60)
    
    # Initialize the estimator
    estimator = SolarPowerEstimator()
    
    print("\n" + "="*60)
    print("Model Information:")
    print("="*60)
    info = estimator.get_model_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Example 1: Single prediction with individual parameters
    print("\n" + "="*60)
    print("Example 1: Single Prediction (Individual Parameters)")

    print("="*60)
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
    print(f"Predicted Power: {power:.2f} kW")
    
    # Example 2: Single prediction with dictionary
    print("\n" + "="*60)
    print("Example 2: Single Prediction (Dictionary)")
    print("="*60)
    input_data = {
        'hour': 12,
        'day': 3,
        'month': 1,
        'air_temperature': 29.35,
        'humidity': 42.0,
        'irradiance': 522.17,
        'pressure': 986.78,
        'rain': 0.0,
        'wind_direction': 198.5,
        'wind_velocity': 3.32
    }
    power = estimator.predict(data=input_data)
    print(f"Input: {input_data}")
    print(f"Predicted Power: {power:.2f} kW")
    
    # Example 3: Batch prediction with DataFrame
    print("\n" + "="*60)
    print("Example 3: Batch Prediction (DataFrame)")
    print("="*60)
    df = pd.DataFrame({
        'hour': [12, 16, 17],
        'day': [17, 28, 9 ],
        'month': [7, 8, 9 ],
        'air_temperature': [30.12, 29.35, 28.62],
        'humidity': [39.25, 42.0, 46.75],
        'irradiance': [831.5, 522.17, 168.08],
        'pressure': [986.85, 986.78, 986.72],
        'rain': [0.0, 0.0, 0.0],
        'wind_direction': [179.83, 198.5, 256.83],
        'wind_velocity': [2.6, 3.32, 3.4]
    })
    powers = estimator.predict(data=df)
    
    df['predicted_power'] = powers
    print(df)
    
    # Example 4: Predict from CSV file
    print("\n" + "="*60)
    print("Example 4: Predict from X_test CSV File")
    print("="*60)
    
    results = estimator.predict_from_csv(
         csv_path='../dataset/train_test_split/X_test.csv',
         output_path='v2/test_predictions.csv'
    )
    print(results.head())
    
    print("\n" + "="*60)
    print("Examples completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()