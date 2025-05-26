
from Regression import MyLinearRegression,my_train_test_split

import pandas

d = pandas.read_csv("weather_daily_2020-03-26_to_2025-05-24.csv")
print(d.head())
print(d.columns)



x = d[['Min Temperature (°C)',
       'Humidity (%)',
       'Pressure (hPa)',
       'Precipitation (mm)',
       'Wind Speed (km/h)']]
y = d[['Max Temperature (°C)']]

print("--"*25,"X data","--"*25)
print(x.info())
print("--"*25,"y data","--"*25)
print(y.info())
x_train, x_test, y_train, y_test = my_train_test_split(x, y, test_size=0.2, random_state=42)


mlr = MyLinearRegression()
mlr.fit(x_train, y_train)

print("Linear Regression R^2 score:", mlr.score(x_test, y_test))

new_data = [[12.1,74.0,1014.8,0.0,18.7]]
predicted_T_Max = mlr.predict(new_data)
print("Predicted Max Temperature (°C):", predicted_T_Max[0][0])

new_data1 = [[15.0,65.0,1012.0,2.0,20.0]]
predicted_T_Max = mlr.predict(new_data1)
print("T1 Predicted Max Temperature (°C):", predicted_T_Max[0][0])

new_data2 = [[20.0,60.0,1010.0,5.0,25.0]]
predicted_T_Max = mlr.predict(new_data2)
print("T2 Predicted Max Temperature (°C):", predicted_T_Max[0][0])

new_data3 = [[25.0,55.0,1008.0,10.0,30.0]]
predicted_T_Max = mlr.predict(new_data3)
print("T3 Predicted Max Temperature (°C):", predicted_T_Max[0][0])

new_data4 = [[18.0,70.0,1011.0,0.0,12.0]]
predicted_T_Max = mlr.predict(new_data4)
print("T4 Predicted Max Temperature (°C):", predicted_T_Max[0][0])