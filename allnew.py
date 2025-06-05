import matplotlib.pyplot as plt
from Regression import MyLinearRegression

# Load all models
Max_Temp = MyLinearRegression.load('json/Max_Temp.json')
Min_Temp = MyLinearRegression.load('json/Min_Temp.json')
Humidity = MyLinearRegression.load('json/Humidity.json')
Precipitation_model = MyLinearRegression.load('json/Precipitation.json')
Pressure_model = MyLinearRegression.load('json/Pressure.json')
Wind_model = MyLinearRegression.load('json/Wind Speed.json')

# Initial weather data [Max_Temp, Min_Temp, Humidity, Pressure, Precipitation, Wind]
current = [40.4, 27.7, 20.0, 1000.5, 0.0, 14.8]
new_prediction = []

# Lists to store predictions for each day
days = range(1, 11)  # 7 days
max_temps = []
min_temps = []
humidities = []
pressures = []
precipitations = []
winds = []

# Predict for 7 days
for day in days:
    next_day = [0] * 6  # Initialize next day's features

    # Predict each feature (code remains the same as yours)
    input_max = [[current[1], current[2], current[3], current[4], current[5]]]
    next_day[0] = Max_Temp.predict(input_max)[0]

    input_min = [[current[0], current[2], current[3], current[4], current[5]]]
    next_day[1] = Min_Temp.predict(input_min)[0]

    input_hum = [[current[0], current[1], current[3], current[4], current[5]]]
    next_day[2] = Humidity.predict(input_hum)[0]

    input_pres = [[current[0], current[1], current[2], current[4], current[5]]]
    next_day[3] = Pressure_model.predict(input_pres)[0]

    input_precip = [[current[0], current[1], current[2], current[3], current[5]]]
    precipitation = Precipitation_model.predict(input_precip)[0]
    next_day[4] = max(0.0, precipitation)

    input_wind = [[current[0], current[1], current[2], current[3], current[4]]]
    next_day[5] = Wind_model.predict(input_wind)[0]

    # Store predictions
    max_temps.append(next_day[0])
    min_temps.append(next_day[1])
    humidities.append(next_day[2])
    pressures.append(next_day[3])
    precipitations.append(next_day[4])
    winds.append(next_day[5])

    # Add predictions to results and print
    print(f"\nDay {day} predictions:")
    for i, val in enumerate(next_day):
        unit = "°C" if i < 2 else ("%" if i == 2 else ("hPa" if i == 3 else ("mm" if i == 4 else "km/h")))
        print(f"{['Max Temp', 'Min Temp', 'Humidity', 'Pressure', 'Precipitation', 'Wind Speed'][i]}: {val:.1f}{unit}")

    new_prediction.extend(next_day)
    current = next_day  # Use predictions for the next day

# Historical weather data (real data)
real_max_temp = [40.4, 42.2, 40.4, 38.3, 33.3, 35.7]  
real_min_temp = [27.7, 26.9, 29.4, 28.6, 25.4, 22.7]  
real_humidity = [20.0, 21.0, 26.0, 37.0, 46.0, 44.0]  
real_pressure = [1000.5, 999.6, 1000.0, 1000.3, 1002.0, 1005.1]  
real_precipitation = [0.0, 0.0, 0.0, 0.0, 3.0, 0.0]  
real_wind = [14.8, 14.6, 26.0, 26.9, 33.1, 11.1]  

day1=len(real_pressure)
day=days[:day1]
# Create plots
plt.figure(figsize=(15, 10))

# Temperature plot
plt.subplot(2, 3, 1)
plt.plot(days, max_temps, 'r-', label='Predicted Max')
plt.plot(days, min_temps, 'b-', label='Predicted Min')
plt.plot(day, real_max_temp, 'r--', label='Actual Max')
plt.plot(day, real_min_temp, 'b--', label='Actual Min')
plt.xlabel('Day')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Forecast vs Actual')
plt.legend()
plt.grid(True)

# Humidity plot
plt.subplot(2, 3, 2)
plt.plot(days, humidities, 'g-', label='Predicted')
plt.plot(day, real_humidity, 'g--', label='Actual')
plt.xlabel('Day')
plt.ylabel('Humidity (%)')
plt.title('Humidity Forecast vs Actual')
plt.legend()
plt.grid(True)

# Pressure plot
plt.subplot(2, 3, 3)
plt.plot(days, pressures, 'm-', label='Predicted')
plt.plot(day, real_pressure, 'm--', label='Actual')
plt.xlabel('Day')
plt.ylabel('Pressure (hPa)')
plt.title('Pressure Forecast vs Actual')
plt.legend()
plt.grid(True)

# Precipitation plot
plt.subplot(2, 3, 4)
plt.bar(days, precipitations, color='c', alpha=0.5, label='Predicted')
plt.bar(day, real_precipitation, color='c', alpha=0.2, label='Actual')
plt.xlabel('Day')
plt.ylabel('Precipitation (mm)')
plt.title('Precipitation Forecast vs Actual')
plt.legend()
plt.grid(True)

# Wind plot
plt.subplot(2, 3, 5)
plt.plot(days, winds, 'y-', label='Predicted')
plt.plot(day, real_wind, 'y--', label='Actual')
plt.xlabel('Day')
plt.ylabel('Wind Speed (km/h)')
plt.title('Wind Speed Forecast vs Actual')
plt.legend()
plt.grid(True)

# Combined plot of predictions only
plt.subplot(2, 3, 6)
for data, label, color in zip([max_temps, min_temps, humidities, winds],
                            ['Max Temp', 'Min Temp', 'Humidity', 'Wind Speed'],
                            ['r', 'b', 'g', 'y']):
    plt.plot(days, data, color, label=label)
plt.xlabel('Day')
plt.title('Combined Predictions')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()