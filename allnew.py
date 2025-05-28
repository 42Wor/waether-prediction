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
current = [40.4,27.7,20.0,1000.5,0.0,14.8]
new_prediction = []

# Lists to store predictions for each day
days = range(1, 8)
max_temps = []
min_temps = []
humidities = []
pressures = []
precipitations = []
winds = []

# Predict for 7 days
for day in days:
    next_day = [0] * 6  # Initialize next day's features

    # Predict each feature using current day's data (excluding the target feature)
    # Max_Temp (index0): exclude current[0]
    input_max = [[current[1], current[2], current[3], current[4], current[5]]]
    next_day[0] = Max_Temp.predict(input_max)[0]

    # Min_Temp (index1): exclude current[1]
    input_min = [[current[0], current[2], current[3], current[4], current[5]]]
    next_day[1] = Min_Temp.predict(input_min)[0]

    # Humidity (index2): exclude current[2]
    input_hum = [[current[0], current[1], current[3], current[4], current[5]]]
    next_day[2] = Humidity.predict(input_hum)[0]

    # Pressure (index3): exclude current[3]
    input_pres = [[current[0], current[1], current[2], current[4], current[5]]]
    next_day[3] = Pressure_model.predict(input_pres)[0]

    # Precipitation (index4): exclude current[4], clamp negative values
    input_precip = [[current[0], current[1], current[2], current[3], current[5]]]
    precipitation = Precipitation_model.predict(input_precip)[0]
    next_day[4] = max(0.0, precipitation)  # Ensure non-negative

    # Wind (index5): exclude current[5]
    input_wind = [[current[0], current[1], current[2], current[3], current[4]]]
    next_day[5] = Wind_model.predict(input_wind)[0]

    # Store predictions for plotting
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

# Create plots
plt.figure(figsize=(15, 10))

# Temperature plot
plt.subplot(2, 3, 1)
plt.plot(days, max_temps, 'r-', label='Max Temp')
plt.plot(days, min_temps, 'b-', label='Min Temp')
plt.xlabel('Day')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Forecast')
plt.legend()
plt.grid(True)

# Humidity plot
plt.subplot(2, 3, 2)
plt.plot(days, humidities, 'g-')
plt.xlabel('Day')
plt.ylabel('Humidity (%)')
plt.title('Humidity Forecast')
plt.grid(True)

# Pressure plot
plt.subplot(2, 3, 3)
plt.plot(days, pressures, 'm-')
plt.xlabel('Day')
plt.ylabel('Pressure (hPa)')
plt.title('Pressure Forecast')
plt.grid(True)

# Precipitation plot
plt.subplot(2, 3, 4)
plt.bar(days, precipitations, color='c')
plt.xlabel('Day')
plt.ylabel('Precipitation (mm)')
plt.title('Precipitation Forecast')
plt.grid(True)

# Wind plot
plt.subplot(2, 3, 5)
plt.plot(days, winds, 'y-')
plt.xlabel('Day')
plt.ylabel('Wind Speed (km/h)')
plt.title('Wind Speed Forecast')
plt.grid(True)

# Combined plot
plt.subplot(2, 3, 6)
for data, label, color in zip([max_temps, min_temps, humidities, winds],
                              ['Max Temp', 'Min Temp', 'Humidity', 'Wind Speed'],
                              ['r', 'b', 'g', 'y']):
    plt.plot(days, data, color, label=label)
plt.xlabel('Day')
plt.title('Combined Forecast')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()