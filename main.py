import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import streamlit as st

# Load dataset
data = pd.read_csv('consolidated_coin_data.csv')

# Data preprocessing steps
data['Date'] = pd.to_datetime(data['Date'])

# Replace commas in numeric columns and convert to float
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']

for col in numeric_columns:
    data[col] = data[col].str.replace(',', '').astype(float)

# Update the selected features list with the correct column names
selected_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']
X = data[selected_features]
y = data['Close']  # Assuming 'Close' price is the target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Streamlit UI
st.title('Cryptocurrency Price Predictor')

# User inputs through sliders, text inputs, etc.
st.sidebar.title('Enter Data for Prediction')

open_price = st.sidebar.number_input('Enter Open Price', min_value=0.0)
high_price = st.sidebar.number_input('Enter High Price', min_value=0.0)
low_price = st.sidebar.number_input('Enter Low Price', min_value=0.0)
close_price = st.sidebar.number_input('Enter Close Price', min_value=0.0)
volume = st.sidebar.number_input('Enter Volume', min_value=0)
market_cap = st.sidebar.number_input('Enter Market Cap', min_value=0)

# Preprocess user input for prediction
user_input = pd.DataFrame({
    'Open': [open_price],
    'High': [high_price],
    'Low': [low_price],
    'Close': [close_price],
    'Volume': [volume],
    'Market Cap': [market_cap]
})

# Make predictions
prediction = model.predict(user_input)

# Display prediction to the user
st.write('Predicted Close Price:', prediction[0])
