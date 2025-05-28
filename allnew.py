from Regression import MyLinearRegression

# Load all models
Max_Temp = MyLinearRegression.load('json/Max_Temp.json')
Min_Temp = MyLinearRegression.load('json/Min_Temp.json')
Humidity = MyLinearRegression.load('json/Humidity.json')
Precipitation_model = MyLinearRegression.load('json/Precipitation.json')
Pressure_model = MyLinearRegression.load('json/Pressure.json')
Wind_model = MyLinearRegression.load('json/Wind Speed.json')

new_data = [[38.8, 28.5, 24.0, 1014.8, 0.0, 14.6]]
original = new_data[0]
new_prediction = []

# Predict Max Temperature
max_temp_input = [[v for i, v in enumerate(original) if i != 0]]
prediction = Max_Temp.predict(max_temp_input)
new_prediction.append(prediction[0])
print(f"{prediction[0]:.1f}°C")  # Added print statement

# Predict Min Temperature
min_temp_input = [[v for i, v in enumerate(original) if i != 1]]
prediction = Min_Temp.predict(min_temp_input)
new_prediction.append(prediction[0])
print(f"{prediction[0]:.1f}°C")

# Predict Humidity
humidity_input = [[v for i, v in enumerate(original) if i != 2]]
prediction = Humidity.predict(humidity_input)
new_prediction.append(prediction[0])
print(f"{prediction[0]:.1f}%")

# Predict Precipitation (clamp negative values to 0)
precipitation_input = [[v for i, v in enumerate(original) if i != 4]]
prediction = Precipitation_model.predict(precipitation_input)
prediction_value = max(0.0, prediction[0])  # Ensure non-negative
new_prediction.append(prediction_value)
print(f"{prediction_value:.1f} mm")

# Predict Pressure
pressure_input = [[v for i, v in enumerate(original) if i != 3]]
prediction = Pressure_model.predict(pressure_input)
new_prediction.append(prediction[0])
print(f"{prediction[0]:.1f} hPa")

# Predict Wind Speed
wind_input = [[v for i, v in enumerate(original) if i != 5]]
prediction = Wind_model.predict(wind_input)
new_prediction.append(prediction[0])
print(f"{prediction[0]:.1f} km/h")

print(new_prediction)