from Polynomial_R import MyPolynomial_R

loaded_model = MyPolynomial_R.load('../json/Polynomial_R_Min_Temp.json')

# Example prediction using loaded model
new_data = [[38.8 ,24.0 ,1014.8,0.0  ,14.6]]
prediction = loaded_model.predict(new_data)
print(f"Predicted temperature: {prediction[0]:.1f}°C")

# Example predictions using trained model
test_samples = [
    ([40.4, 20.0, 1000.5, 0.0, 14.8], "T1"),
    ([42.2, 21.0, 999.6, 0.0, 14.6], "T2"),
    ([40.4,  26.0, 1000.0, 0.0, 26.0], "T3"),
    ([38.3, 37.0, 1000.3, 0.0, 26.9], "T4"),
    ([33.3, 46.0, 1002.0, 3.0, 33.1], "T5"),
    ([35.7,44.0, 1005.1, 0.0, 11.1], "T6")
]

for features, label in test_samples:
    predicted_T_Mix = loaded_model.predict([features])
    print(f"{label} Predicted Min Temperature (°C): {predicted_T_Mix[0]:.1f}")