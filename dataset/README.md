# Solar System Forecasting: Dataset Creation

For the creation of the dataset we start by collecting data of the weather and the solar production. These two processes are implemented and described respectively in the subdirectory `solar_system_data` and `weather_data`. 

These two modules preprocess the raw data, which then are processed for the creation of the dataset where the input X are the weather/meteo data and the ground truth is the solar system production. 

## 📁 Project Structure
```
dataset/
│
├── README.md                          # This file
├── solar_system_data/                 # Solar production data collection
├── weather_data/                      # Weather data collection
│ 
├── dataset_preparation/               # Dataset creation and preparation
│   ├── merged_data/                   # Merged solar + weather data
│   ├── train_test_split/              # Split datasets
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── y_train.csv
│   │   ├── y_test.csv
│   │   ├── X_train_scaled.csv
│   │   └── X_test_scaled.csv
│   └── notebooks/
│       └── dataset_preparation.ipynb  # Main dataset creation notebook
│
```

## 📊 Dataset Information

### Input Features (X)
The weather/meteorological features used as input:

| Feature | Description | Unit |
|---------|-------------|------|
| `air_temperature` | Ambient air temperature | °C   |
| `humidity`        | Relative humidity       | %    |
| `irradiance`      | Solar irradiance        | W/m² |
| `pressure`        | Atmospheric pressure    | hPa  |
| `rain`            | Precipitation amount    | mm   |
| `wind_direction`  | Wind direction          | degrees |
| `wind_velocity`   | Wind speed              | m/s |

### Target Variable (y)
- `power`: Solar system power production in kW

### Data Characteristics
- **Temporal Resolution**: Hourly measurements
- **Solar Data Period**: 2019-06-18 onwards
- **Weather Data Period**: 2003-01-01 onwards
- **Merged Dataset**: Intersection of both time periods
- **Train/Test Split**: 80/20 time-based split (chronological)


## 🚀 Usage

### 1. Data Collection and Preprocessing

#### Solar Production Data
```bash
cd solar_system_data
python dataPreprocessing.py
```

#### Weather Data
```bash
cd weather_data
python dataPreprocessing.py
python dataMerge.py
```

### 2. Dataset Preparation
Run the dataset preparation file:
```bash
python datasetCreation.py
```

This will:
- Load and merge solar and weather data
- Handle missing values
- Perform time-based train/test split (80/20)
- Scale features using StandardScaler
- Save processed datasets to `train_test_split/`
