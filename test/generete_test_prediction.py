import sys, os
import pandas as pd

# Add the parent directory (SolarSystemForecaster) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from FFN.FFN_estimator import SolarPowerEstimator

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
    
    # Predict for the Calibrated and PostProcessed Weather Forecast
    print("\n" + "="*60)
    print("Predict from calibrated_historical_forecast CSV File")
    print("="*60)

    output_path='test_predictions.csv'
    
    results = estimator.predict_from_csv(
         csv_path='calibrated_historical_forecast.csv',
         output_path=output_path
    )

    # Open the CSV, round 'predict_power' to 2 decimal places, and save it again
    df = pd.read_csv(output_path)
    if 'predicted_power' in df.columns:
        df['predicted_power'] = df['predicted_power'].round(1)
        df.to_csv(output_path, index=False)
    
    print("\n" + "="*60)
    print("Examples completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()