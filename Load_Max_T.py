from Regression import MyLinearRegression, my_train_test_split

loaded_model = MyLinearRegression.load('weather_model.json')

# Example prediction using loaded model
new_data = [[15.0, 65.0, 1012.0, 2.0, 20.0]]
prediction = loaded_model.predict(new_data)
print(f"Predicted temperature: {prediction[0]:.1f}°C")

# Example predictions using trained model
test_samples = [
    ([15.0, 65.0, 1012.0, 2.0, 20.0], "T1"),
    ([20.0, 60.0, 1010.0, 5.0, 25.0], "T2"),
    ([25.0, 55.0, 1008.0, 10.0, 30.0], "T3"),
    ([18.0, 70.0, 1011.0, 0.0, 12.0], "T4"),
]

for features, label in test_samples:
    predicted_T_Max = loaded_model.predict([features])
    print(f"{label} Predicted Max Temperature (°C): {predicted_T_Max[0]:.1f}")