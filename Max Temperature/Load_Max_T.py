from Regression import MyLinearRegression, my_train_test_split

loaded_model = MyLinearRegression.load('Max_Temp.json')

# Example prediction using loaded model
new_data = [[28.5,24.0,1014.8,0.0,14.6]]
prediction = loaded_model.predict(new_data)
print(f"Predicted temperature: {prediction[0]:.1f}°C")

# Example predictions using trained model
test_samples = [
    ([27.7, 20.0, 1000.5, 0.0, 14.8], "T1"),
    ([26.9, 21.0, 999.6, 0.0, 14.6], "T2"),
    ([29.4, 26.0, 1000.0, 0.0, 26.0], "T3"),
    ([28.6, 37.0, 1000.3, 0.0, 26.9], "T4"),
    ([25.4, 46.0, 1002.0, 3.0, 33.1], "T5"),
    ([22.7, 44.0, 1005.1, 0.0, 11.1], "T6")
]

for features, label in test_samples:
    predicted_T_Max = loaded_model.predict([features])
    print(f"{label} Predicted Max Temperature (°C): {predicted_T_Max[0]:.1f}")