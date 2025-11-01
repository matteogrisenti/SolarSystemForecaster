import os
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import matplotlib.pyplot as plt
from datetime import datetime

from utils.handle_missing_values import handle_solar_missing_values_smart
from utils.edit_original_dataset import add_time_and_season_columns, process_wind_data

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

def load_and_prepare_data(file_path, timestamp_column='datetime'):
    """
    Load dataset and prepare it for time series forecasting
    
    Parameters:
    -----------
    file_path : str
        Path to your CSV file
    timestamp_column : str
        Name of the timestamp column
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Ensure timestamp is datetime
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    # Sort by timestamp
    df = df.sort_values(timestamp_column)
        
    return df


# ============================================================================
# CREATE TIMESERIES DATAFRAME
# ============================================================================

def create_timeseries_dataframe(df, datetime_column='datetime', target='power'):
    """
    Convert pandas DataFrame to AutoGluon TimeSeriesDataFrame
    """
    # Create TimeSeriesDataFrame
    # AutoGluon expects 'item_id' for different series (optional if single series)

    #####
    # Here we can think about split for each solar panel 
    #####
    df['item_id'] = 'solar_panel_prediction'  # Add item_id for single series
    
    # Reorder columns: item_id, timestamp, target, features
    feature_cols = ['air_temperature', 'humidity', 'irradiance', 
                    'pressure', 'rain', 
                    'is_day', 'season',
                   #'wind_u', 'wind_v',
                    'wind_velocity',
                   'wind_direction'
                   ]
    
    cols = ['item_id', datetime_column, target] + feature_cols
    df = df[cols]
    
    # Convert to TimeSeriesDataFrame
    ts_df = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column='item_id',
        timestamp_column=datetime_column
    )
    
    return ts_df

# ============================================================================
# TRAIN-TEST SPLIT
# ============================================================================

def split_train_test(ts_df, test_size=0.2):
    """
    Split time series data into train and test sets
    """
    # Calculate split point
    total_length = len(ts_df)
    train_length = int(total_length * (1 - test_size))
    
    # Split data
    train_data = ts_df.head(train_length)
    test_data = ts_df.tail(total_length - train_length)
    
    return train_data, test_data

# ============================================================================
# TRAIN MODEL WITH CHRONOS
# ============================================================================

def train_model(train_data, prediction_length=24, time_limit=600):
    predictor = TimeSeriesPredictor(
        path="autogluon-solar-forecast",
        prediction_length=prediction_length,
        eval_metric="sMAPE",
        target="power",
        known_covariates_names=[
                'air_temperature', 'humidity', 'irradiance', 
                    'pressure', 'rain',
                    'is_day', 'season',
                   #'wind_u', 'wind_v',
                    'wind_velocity',
                   'wind_direction'
        ]
    ).fit(
        train_data,
        presets="best_quality",
        hyperparameters={
            "Chronos": [
                # Zero-shot model WITHOUT covariates
                {
                    "model_path": "bolt_small",
                    "ag_args": {"name_suffix": "ZeroShot"},
                },
                # Chronos-Bolt (Small) combined with CatBoost on covariates
                {
                    "model_path": "bolt_small",
                    "covariate_regressor": "CAT",
                    "target_scaler": "standard",
                    "ag_args": {"name_suffix": "WithRegressor"},
                },
            ],
        },
        enable_ensemble=False,
        #time_limit=120,
    )

    return predictor

def evaluate_model(train_data, test_data, models_name, predictor):
    predictions = {}
    # Get predictions
    for model_name in models_name:
        predictions[model_name] = predictor.predict(train_data, known_covariates=test_data.drop(columns=['power']), model=model_name)
    
    return predictions


# ============================================================================
# VISUALIZE PREDICTIONS
# ============================================================================

def plot_predictions(test_data, predictions):
    # ============= PLOTTING CON MATPLOTLIB =============
    plt.figure(figsize=(15, 6))

    # Serie reale
    actual = test_data.reset_index()


    #prendo solo gli ultimi 150 punti per visualizzare meglio
    actual = actual.tail(200)

    plt.plot(actual["timestamp"], actual["power"], label="Actual (Observed)", color="black", linewidth=2)

    # Colori personalizzati (uno per modello)
    colors = ["tab:orange", "tab:green", "tab:blue", "tab:red", "tab:purple"]

    for i, (model_name, preds) in enumerate(predictions.items()):
        pred_mean = preds["mean"].reset_index()
        pred_mean_tail = pred_mean.tail(200)

        plt.plot(
            pred_mean_tail["timestamp"], pred_mean_tail["mean"],
            label=f"{model_name}",
            color=colors[i % len(colors)],
            linestyle="--",
            linewidth=2
        )

        # Se il modello fornisce gli intervalli di confidenza
        if "0.1" in preds.columns and "0.9" in preds.columns:
            pred_lower = preds["0.1"].reset_index().tail(200)
            pred_upper = preds["0.9"].reset_index().tail(200)

            plt.fill_between(
                pred_mean_tail["timestamp"],
                pred_lower["0.1"],
                pred_upper["0.9"],
                color=colors[i % len(colors)],
                alpha=0.2
            )

    plt.title("Confronto previsioni Chronos (tutti i modelli)")
    plt.xlabel("Timestamp")
    plt.ylabel("Power")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    #save figure (avoid overwriting iteratively)
    for i in range(100):
        if os.path.exists(f"solar_power_forecasting_{i}.png"):
            continue
        plt.savefig(f"solar_power_forecasting_{i}.png")
        break
    
    plt.show()



def main():
    """
    Main execution function
    """
    print("="*50)
    print("SOLAR PANEL POWER FORECASTING WITH AUTOGLUON")
    print("="*50)
    
    # Configuration
    DATA_PATH = 'merged_solar_weather_data_cleaned.csv'  # UPDATE THIS PATH
    TIMESTAMP_COL = 'datetime'   # UPDATE IF DIFFERENT
    PREDICTION_LENGTH = 192        # Forecast 192 time steps ahead
    TEST_SIZE = 0.2               # 20% for testing
    TIME_LIMIT = 600              # 10 minutes training time
    
    # Step 1: Load data
    print("\n1. Loading data...")
    df = load_and_prepare_data(DATA_PATH, TIMESTAMP_COL)
    print(f"   Data shape: {df.shape}")
    print(f"   Date range: {df[TIMESTAMP_COL].min()} to {df[TIMESTAMP_COL].max()}")

    # Add is_day-hour and season columns
    df=add_time_and_season_columns(df, timestamp_column=TIMESTAMP_COL)
    
    # Process wind data → Resta uguale/peggiora (da rivedere)
    #df=process_wind_data(df)

    df_handle_missing_value=handle_solar_missing_values_smart(df)
    

    df_handle_missing_value.to_csv("handle_missing_value.csv", index=False)
    ts_df = create_timeseries_dataframe(df_handle_missing_value, TIMESTAMP_COL, target='power')

    prediction_length = PREDICTION_LENGTH
    train_data, test_data = ts_df.train_test_split(prediction_length)

    predictor=train_model(train_data=train_data, prediction_length=PREDICTION_LENGTH, time_limit=TIME_LIMIT)
    

    print("LEADERBOARD:")
    print(predictor.leaderboard(test_data))

    models=['ChronosZeroShot[bolt_small]', 'ChronosWithRegressor[bolt_small]']

    predictions=evaluate_model(train_data=train_data, test_data=test_data, models_name=models, predictor=predictor)

    plot_predictions(test_data=test_data, predictions=predictions)



if __name__ == "__main__":
    main()