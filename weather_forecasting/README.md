# 🌤 Weather Forecaster
This project provides a system to forecast weather for the next day using Open-Meteo, a free weather forecast API. Because Open-Meteo is a global system, its predictions may be less accurate for specific locations. To improve precision, a calibration system has been implemented to adjust Open-Meteo data for the target location. 

## 📁 Project Structure
```
weather_forecast/
│
├── calibrator/                        # Calibrator implementation
│   ├── dataset/                       # Dataset and code to generate it
│   ├── test/                          # Inputs and outputs for calibrator testing
│   ├── calibrator.pkl                 # Trained calibrator instance
│   ├── calibrator.py                  # Calibrator implementation, training, and testing
|
├── forecast.py                        # Forecast pipeline to generate the final forecast
├── open_meteo_fetch.py                # Fetch raw forecast data from Open-Meteo API
├── README.md                          # This file
```

## ⚡ Forecast Pipeline
The main function to get the weather forecast is weather_forecast() in forecast.py. It executes the following pipeline:

Fetch raw forecast – Use fetch_weather_forecast() in open_meteo_fetch.py to retrieve Open-Meteo data for the next day.

Calibration – Improve the raw forecast data using the trained calibrator to correct location-specific discrepancies.

Post-processing – Clean and refine the calibrated data for higher quality results.

The final output is a Pandas DataFrame with the following structure:
```
| hour | day | month | air_temperature | humidity | irradiance | pressure | rain | wind_direction | wind_velocity |
|------|-----|-------|-----------------|----------|------------|----------|------|----------------|---------------|
|0     |5    |11     |5.9              |66        |0.0         |1002.2    |0.0   |180             |0.11           |
|1     |5    |11     |5.9              |65        |0.0         |999       |0.0   |190             |0.20           |
|.     |.    |.      |.                |.         |.           |.         |.     |.               |.              |
|.     |.    |.      |.                |.         |.           |.         |.     |.               |.              |
|.     |.    |.      |.                |.         |.           |.         |.     |.               |.              |
|24    |x    |x      |x                |x         |x           |x         |x     |x               |x              |
```


