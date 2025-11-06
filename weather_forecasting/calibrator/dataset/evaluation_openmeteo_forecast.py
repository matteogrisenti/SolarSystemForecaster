import pandas as pd
import numpy as np
from datetime import datetime

def calculate_metrics(actual, forecast, feature_name):
    """Calculate comprehensive error metrics for a feature"""
    # Remove any NaN values
    mask = ~(np.isnan(actual) | np.isnan(forecast))
    actual = actual[mask]
    forecast = forecast[mask]
    
    if len(actual) == 0:
        return None
    
    # Basic error metrics
    mae = np.mean(np.abs(actual - forecast))
    rmse = np.sqrt(np.mean((actual - forecast) ** 2))
    mape = np.mean(np.abs((actual - forecast) / (actual + 1e-10))) * 100  # Add small value to avoid division by zero
    
    # Bias (systematic over/under prediction)
    bias = np.mean(forecast - actual)
    
    # Correlation
    correlation = np.corrcoef(actual, forecast)[0, 1] if len(actual) > 1 else 0
    
    # R-squared
    ss_res = np.sum((actual - forecast) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    
    # Percentage within tolerance bands
    within_5_pct = np.mean(np.abs(forecast - actual) <= np.abs(actual * 0.05)) * 100
    within_10_pct = np.mean(np.abs(forecast - actual) <= np.abs(actual * 0.10)) * 100
    
    # Max errors
    max_error = np.max(np.abs(actual - forecast))
    max_overpredict = np.max(forecast - actual)
    max_underpredict = np.min(forecast - actual)
    
    return {
        'feature': feature_name,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'bias': bias,
        'correlation': correlation,
        'r2': r2,
        'within_5_pct': within_5_pct,
        'within_10_pct': within_10_pct,
        'max_error': max_error,
        'max_overpredict': max_overpredict,
        'max_underpredict': max_underpredict,
        'n_samples': len(actual)
    }

def generate_report(actual_file, forecast_file, output_file='forecast_quality_report.txt'):
    """Generate comprehensive forecast quality report"""
    
    print("Loading datasets...")
    df_actual = pd.read_csv(actual_file)
    df_forecast = pd.read_csv(forecast_file)
    
    # Convert datetime to datetime type
    df_actual['datetime'] = pd.to_datetime(df_actual['datetime'])
    df_forecast['datetime'] = pd.to_datetime(df_forecast['datetime'])
    
    # Merge on datetime
    df_merged = pd.merge(df_actual, df_forecast, on='datetime', suffixes=('_actual', '_forecast'))
    
    print(f"Merged {len(df_merged)} records")
    
    # Features to evaluate (excluding datetime and derived features)
    features = ['air_temperature', 'humidity', 'irradiance', 'pressure', 'rain', 'wind_direction', 'wind_velocity']
    
    # Calculate metrics for each feature
    results = []
    for feature in features:
        actual_col = f"{feature}_actual"
        forecast_col = f"{feature}_forecast"
        
        if actual_col in df_merged.columns and forecast_col in df_merged.columns:
            metrics = calculate_metrics(
                df_merged[actual_col].values,
                df_merged[forecast_col].values,
                feature
            )
            if metrics:
                results.append(metrics)
    
    # Generate report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("METEOROLOGICAL FORECAST QUALITY EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Evaluation Period: {df_merged['datetime'].min()} to {df_merged['datetime'].max()}\n")
        f.write(f"Total Records Evaluated: {len(df_merged)}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall summary table
        f.write(f"{'Feature':<20} {'MAE':<12} {'RMSE':<12} {'Correlation':<12} {'R²':<12}\n")
        f.write("-" * 80 + "\n")
        for r in results:
            f.write(f"{r['feature']:<20} {r['mae']:<12.4f} {r['rmse']:<12.4f} {r['correlation']:<12.4f} {r['r2']:<12.4f}\n")
        
        f.write("\n\n")
        
        # Detailed metrics for each feature
        for r in results:
            f.write("=" * 80 + "\n")
            f.write(f"FEATURE: {r['feature'].upper()}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Error Metrics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Mean Absolute Error (MAE):           {r['mae']:.4f}\n")
            f.write(f"  Root Mean Square Error (RMSE):       {r['rmse']:.4f}\n")
            f.write(f"  Mean Absolute Percentage Error:      {r['mape']:.2f}%\n")
            f.write(f"  Bias (Systematic Error):             {r['bias']:.4f}\n")
            if r['bias'] > 0:
                f.write(f"    -> Tendency to OVER-predict\n")
            elif r['bias'] < 0:
                f.write(f"    -> Tendency to UNDER-predict\n")
            else:
                f.write(f"    -> No systematic bias\n")
            f.write("\n")
            
            f.write("Statistical Performance:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Correlation Coefficient:             {r['correlation']:.4f}\n")
            f.write(f"  R² Score:                            {r['r2']:.4f}\n")
            f.write("\n")
            
            f.write("Accuracy Analysis:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Within 5% of actual:                 {r['within_5_pct']:.2f}%\n")
            f.write(f"  Within 10% of actual:                {r['within_10_pct']:.2f}%\n")
            f.write("\n")
            
            f.write("Extreme Errors:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Maximum Absolute Error:              {r['max_error']:.4f}\n")
            f.write(f"  Maximum Over-prediction:             {r['max_overpredict']:.4f}\n")
            f.write(f"  Maximum Under-prediction:            {r['max_underpredict']:.4f}\n")
            f.write("\n\n")
        
        # Quality assessment
        f.write("=" * 80 + "\n")
        f.write("QUALITY ASSESSMENT\n")
        f.write("=" * 80 + "\n\n")
        
        for r in results:
            f.write(f"{r['feature']}:\n")
            
            # Assessment based on correlation and R²
            if r['correlation'] > 0.9 and r['r2'] > 0.8:
                quality = "EXCELLENT"
            elif r['correlation'] > 0.7 and r['r2'] > 0.6:
                quality = "GOOD"
            elif r['correlation'] > 0.5 and r['r2'] > 0.4:
                quality = "MODERATE"
            else:
                quality = "NEEDS IMPROVEMENT"
            
            f.write(f"  Overall Quality: {quality}\n")
            
            # Specific recommendations
            if abs(r['bias']) > r['mae'] * 0.5:
                f.write(f"  ! Significant bias detected - consider calibration\n")
            if r['mape'] > 20:
                f.write(f"  ! High percentage error - review forecasting model\n")
            if r['correlation'] < 0.5:
                f.write(f"  ! Low correlation - forecast may not capture actual patterns\n")
            
            f.write("\n")
    
    print(f"\nReport saved to: {output_file}")
    print("\nQuick Summary:")
    print("-" * 60)
    print(f"{'Feature':<20} {'MAE':<12} {'Correlation':<12} {'Quality':<15}")
    print("-" * 60)
    for r in results:
        if r['correlation'] > 0.9 and r['r2'] > 0.8:
            quality = "EXCELLENT"
        elif r['correlation'] > 0.7 and r['r2'] > 0.6:
            quality = "GOOD"
        elif r['correlation'] > 0.5 and r['r2'] > 0.4:
            quality = "MODERATE"
        else:
            quality = "NEEDS IMPROVE"
        print(f"{r['feature']:<20} {r['mae']:<12.4f} {r['correlation']:<12.4f} {quality:<15}")

if __name__ == "__main__":
    # Replace these with your actual file paths
    actual_file = 'historical_actual.csv'
    forecast_file = 'historical_forecast.csv'
    
    generate_report(actual_file, forecast_file)