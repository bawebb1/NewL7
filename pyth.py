import streamlit as st
import pandas as pd
import openpyxl
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import subprocess
import sys

# Packages did not work :(
def install_if_missing(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        __import__(package)

install_if_missing("openpyxl")
install_if_missing("joblib")


# Load 
df = pd.read_excel('AmesHousing.xlsx', engine='openpyxl')


features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
target = 'SalePrice'

# Data 
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Save 
joblib.dump(model, 'housing_model.pkl')

# Streamlit
st.title('Ames Housing Price Predictor')
st.write('Enter details to predict house price')

# User input
inputs = {}
for feature in features:
    inputs[feature] = st.number_input(f'{feature}', min_value=float(X[feature].min()), max_value=float(X[feature].max()), value=float(X[feature].median()))

# Predict button
if st.button('Predict Price'):
    model = joblib.load('housing_model.pkl')
    input_data = pd.DataFrame([inputs])
    prediction = model.predict(input_data)
    st.success(f'Estimated House Price: ${prediction[0]:,.2f}')
