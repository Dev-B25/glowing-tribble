import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load The Dataset
data = pd.read_csv(r'D:\OneDrive\Desktop\INTERNSHIP\Project 2\dataset vehicle.csv')

# Data Preprocessing
data = data[['year', 'price', 'cylinders', 'fuel', 'mileage', 'transmission', 'doors']]
data.dropna(inplace=True)

# Encode categorical features
label_encoders = {}
categorical_features = ['fuel', 'transmission']

for col in categorical_features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Features and target
X = data[['year', 'cylinders', 'fuel', 'mileage', 'transmission', 'doors']]
y = data['price']

# Spliting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save the model
with open('vehicle_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Streamlit Web App Interface
st.title("Vehicle Price Prediction")
year = st.number_input("Enter Vehicle Year:", min_value=1900, max_value=2025, step=1)
cylinders = st.number_input("Enter Number of Cylinders:", min_value=1, max_value=16, step=1)
fuel = st.selectbox("Select Fuel Type:", label_encoders['fuel'].classes_)
mileage = st.number_input("Enter Mileage (miles):", min_value=0.0, step=0.1)
transmission = st.selectbox("Select Transmission Type:", label_encoders['transmission'].classes_)
doors = st.number_input("Enter Number of Doors:", min_value=1, max_value=6, step=1)

# Encoding Inputs
fuel_encoded = label_encoders['fuel'].transform([fuel])[0]
transmission_encoded = label_encoders['transmission'].transform([transmission])[0]

if st.button("Predict Price"):
    input_data = pd.DataFrame([[year, cylinders, fuel_encoded, mileage, transmission_encoded, doors]],
                              columns=['year', 'cylinders', 'fuel', 'mileage', 'transmission', 'doors'])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Vehicle Price: ${prediction:.2f}")


