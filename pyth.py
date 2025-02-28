import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_excel('AmesHousing.xlsx')

# Select relevant features (Modify based on dataset columns)
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
target = 'SalePrice'

# Data preprocessing
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'housing_model.pkl')

# Streamlit app
st.title('Ames Housing Price Predictor')
st.write('Enter details to predict house price')

# User input fields
inputs = {}
for feature in features:
    inputs[feature] = st.number_input(f'{feature}', min_value=float(X[feature].min()), max_value=float(X[feature].max()), value=float(X[feature].median()))

# Predict button
if st.button('Predict Price'):
    model = joblib.load('housing_model.pkl')
    input_data = pd.DataFrame([inputs])
    prediction = model.predict(input_data)
    st.success(f'Estimated House Price: ${prediction[0]:,.2f}')
