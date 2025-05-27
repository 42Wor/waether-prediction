from Regression import MyLinearRegression, my_train_test_split

loaded_model = MyLinearRegression.load('Pressure.json')

# Example prediction using loaded model
new_data = [[38.8,28.5,24.0,0.0,14.6]]
prediction = loaded_model.predict(new_data)
print(f"Predicted : {prediction[0]:.1f} hPa")

# Example predictions using trained model
test_samples = [
    ([40.4, 27.7, 20.0, 0.0, 14.8], "T1"),
    ([42.2, 26.9, 21.0,  0.0, 14.6], "T2"),
    ([40.4, 29.4, 26.0,  0.0, 26.0], "T3"),
    ([38.3, 28.6, 37.0,  0.0, 26.9], "T4"),
    ([33.3, 25.4, 46.0,  3.0, 33.1], "T5"),
    ([35.7, 22.7, 44.0, 0.0, 11.1], "T6")
]


for features, label in test_samples:
    predicted = loaded_model.predict([features])
    print(f"{label} Predicted  {predicted[0]:.1f} hPa")