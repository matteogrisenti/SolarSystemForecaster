import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("=" * 80)
print("STEP 1: Loading Data")
print("=" * 80)

# Load power production data
power_df = pd.read_csv('solar_system_data/solar_power_data.csv')
print(f"\nPower data shape: {power_df.shape}")
print("\nPower data sample:")
print(power_df.head())

# Load weather data
weather_df = pd.read_csv('weather_data/weather_data.csv')
print(f"\nWeather data shape: {weather_df.shape}")
print("\nWeather data sample:")
print(weather_df.head())



# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: Data Preprocessing")
print("=" * 80)

# Convert datetime columns to datetime type
power_df['datetime'] = pd.to_datetime(power_df['datetime'])
weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])

# Sort by datetime
power_df = power_df.sort_values('datetime').reset_index(drop=True)
weather_df = weather_df.sort_values('datetime').reset_index(drop=True)

print("\nDate ranges:")
print(f"Power data: {power_df['datetime'].min()} to {power_df['datetime'].max()}")
print(f"Weather data: {weather_df['datetime'].min()} to {weather_df['datetime'].max()}")

# Check for missing values
print("\nMissing values in power data:")
print(power_df.isnull().sum())
print("\nMissing values in weather data:")
print(weather_df.isnull().sum())



# ============================================================================
# 3. MERGE DATASETS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Merging Datasets")
print("=" * 80)

# Merge on datetime (inner join to keep only matching timestamps)
df = pd.merge(power_df, weather_df, on='datetime', how='inner')
print(f"\nMerged dataset shape: {df.shape}")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print("\nMerged data sample:")
print(df.head(10))



# ============================================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: Exploratory Data Analysis")
print("=" * 80)

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Check for any remaining missing values
print("\nMissing values after merge:")
print(df.isnull().sum())

# Handle missing values if any (forward fill then backward fill)
if df.isnull().sum().sum() > 0:
    print("\nHandling missing values...")
    df = df.fillna(method='ffill').fillna(method='bfill')
    print("Missing values after handling:")
    print(df.isnull().sum())

# Visualize power production distribution
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.hist(df['power'], bins=50, edgecolor='black')
plt.xlabel('Power (kW)')
plt.ylabel('Frequency')
plt.title('Power Production Distribution')

plt.subplot(2, 3, 2)
plt.scatter(df['irradiance'], df['power'], alpha=0.3)
plt.xlabel('Irradiance')
plt.ylabel('Power (kW)')
plt.title('Power vs Irradiance')

plt.subplot(2, 3, 3)
plt.scatter(df['air_temperature'], df['power'], alpha=0.3)
plt.xlabel('Temperature (°C)')
plt.ylabel('Power (kW)')
plt.title('Power vs Temperature')

plt.subplot(2, 3, 4)
plt.scatter(df['humidity'], df['power'], alpha=0.3)
plt.xlabel('Humidity (%)')
plt.ylabel('Power (kW)')
plt.title('Power vs Humidity')

plt.subplot(2, 3, 5)
plt.scatter(df['wind_velocity'], df['power'], alpha=0.3)
plt.xlabel('Wind Velocity (m/s)')
plt.ylabel('Power (kW)')
plt.title('Power vs Wind Velocity')

plt.subplot(2, 3, 6)
df['power'].plot()
plt.xlabel('Time Index')
plt.ylabel('Power (kW)')
plt.title('Power Production Over Time')

plt.tight_layout()
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = df.drop('datetime', axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, linewidths=1)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

print("\nCorrelation with Power Production:")
print(correlation_matrix['power'].sort_values(ascending=False))



# ============================================================================
# 5. KEEP RAW FEATURES ONLY
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: Using Raw Features (No Feature Engineering)")
print("=" * 80)

# Keep only the raw features - let the deep learning models extract features
df_clean = df.copy()

print("\nUsing raw weather features only:")
print(df_clean.columns.tolist())
print(f"\nDataset shape: {df_clean.shape}")

# Check for any NaN values
if df_clean.isnull().sum().sum() > 0:
    print("\nRemoving rows with NaN values...")
    df_clean = df_clean.dropna().reset_index(drop=True)
    print(f"Dataset shape after removing NaN: {df_clean.shape}")



# ============================================================================
# 6. PREPARE FEATURES AND TARGET
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: Preparing Features and Target")
print("=" * 80)

# Use only raw weather features
feature_cols = [
    'power', 'hour', 'day', 'month', 'air_temperature', 'humidity', 'irradiance', 'pressure', 'rain', 'wind_direction', 'wind_velocity'
]

X = df_clean[feature_cols]
y = df_clean['power']
datetime_index = df_clean['datetime']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"\nFeatures used: {len(feature_cols)}")
print(feature_cols)



# ============================================================================
# 7. TRAIN-TEST SPLIT
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: Train-Test Split")
print("=" * 80)

# Time-based split (80-20)
# Important: For time series, we should not use random split
split_index = int(len(df_clean) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]
datetime_train = datetime_index[:split_index]
datetime_test = datetime_index[split_index:]

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")
print(f"Training period: {datetime_train.min()} to {datetime_train.max()}")
print(f"Test period: {datetime_test.min()} to {datetime_test.max()}")



# ============================================================================
# 8. FEATURE SCALING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: Feature Scaling")
print("=" * 80)

# Standardize features (fit on training data only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for easier handling
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)

print("\nFeature scaling completed!")
print(f"Scaled training set shape: {X_train_scaled.shape}")
print(f"Scaled test set shape: {X_test_scaled.shape}")



# ============================================================================
# 9. SAVE PROCESSED DATA
# ============================================================================

print("\n" + "=" * 80)
print("STEP 9: Saving Processed Data")
print("=" * 80)

# Define the output directory
output_dir = "train_test_split"

# Intelligently create the directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"📁 Created directory: '{output_dir}'")
else:
    print(f"📂 Directory already exists: '{output_dir}'")

# Define file paths
paths = {
    "X_train": f"{output_dir}/X_train.csv",
    "X_test": f"{output_dir}/X_test.csv",
    "y_train": f"{output_dir}/y_train.csv",
    "y_test": f"{output_dir}/y_test.csv",
    "X_train_scaled": f"{output_dir}/X_train_scaled.csv",
    "X_test_scaled": f"{output_dir}/X_test_scaled.csv",
}

# Save the processed datasets
X_train.to_csv(paths["X_train"], index=False)
X_test.to_csv(paths["X_test"], index=False)
y_train.to_csv(paths["y_train"], index=False)
y_test.to_csv(paths["y_test"], index=False)

X_train_scaled.to_csv(paths["X_train_scaled"], index=False)
X_test_scaled.to_csv(paths["X_test_scaled"], index=False)

print("\n✅ Processed data successfully saved to:")
for name, path in paths.items():
    print(f"   - {name}: {path}")



# ============================================================================
# 10. SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 10: Summary Statistics")
print("=" * 80)

print("\nTarget Variable Statistics:")
print(f"Training set - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}, "
      f"Min: {y_train.min():.2f}, Max: {y_train.max():.2f}")
print(f"Test set - Mean: {y_test.mean():.2f}, Std: {y_test.std():.2f}, "
      f"Min: {y_test.min():.2f}, Max: {y_test.max():.2f}")

# Visualize train-test split
plt.figure(figsize=(15, 5))
plt.plot(datetime_train.values, y_train.values, label='Training Data', alpha=0.7)
plt.plot(datetime_test.values, y_test.values, label='Test Data', alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Power Production (kW)')
plt.title('Train-Test Split Visualization')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("DATASET PREPARATION COMPLETED!")
print("=" * 80)
