from Ridge_R import MyRidgeRegression, my_train_test_split
import pandas as pd

# Load data
d = pd.read_csv("../data/weather_daily_2020-03-26_to_2025-05-24.csv")

# Prepare features and target
x = d[['Max Temperature (°C)',
       'Humidity (%)',
       'Pressure (hPa)',
       'Precipitation (mm)',
       'Wind Speed (km/h)']]

y = d['Min Temperature (°C)']  # Use single brackets for Series

# Split data
x_train, x_test, y_train, y_test = my_train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Train and evaluate model
mlr = MyRidgeRegression()
mlr.fit(x_train, y_train)
print("R² score:", mlr.score(x_test, y_test))

# Save model
mlr.save('../json/Ridge_R_Min_Temp.json')

