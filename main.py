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



st.set_page_config(page_title='Cryptocurrency Price Predictor', layout='wide')


st.title('Cryptocurrency Price Predictor')
st.markdown("---")



# Sidebar layout
st.sidebar.title('Enter Data for Prediction')

# Currency name dropdown
currency_names = [
    'tezos', 'binance-coin', 'eos', 'bitcoin', 'tether',
    'xrp', 'bitcoin-cash', 'stellar', 'litecoin', 'ethereum',
    'cardano', 'bitcoin-sv'
]

selected_currency = st.sidebar.selectbox('Select Currency Name', currency_names)

if selected_currency:
    filtered_data = data[data['Currency'] == selected_currency]

    # Predefined options for inputs
    open_prices = [1.28, 0.909666, 44.53, 90.17, 238.02, 8320.83, None]
    high_prices = [1.32, 0.965669, 7.09, 24.58, 8410.71, 395.5, None]
    low_prices = [367.83, 4377.46, 23.16, 3.79, 0.606857, None]
    close_prices = [0.850526, 1.23, 5285.14, 382.3, None]
    volumes = [0, 3284328, 792592, 62818.3, None]
    market_caps = [951600, 1451600, 9048082, None]

    col1, col2 = st.sidebar.columns(2)

    with col1:
        open_price = st.selectbox('Enter Open Price', open_prices)
        high_price = st.selectbox('Enter High Price', high_prices)
        low_price = st.selectbox('Enter Low Price', low_prices)
        close_price = st.selectbox('Enter Close Price', close_prices)

    with col2:
        volume = st.selectbox('Enter Volume', volumes)
        market_cap = st.selectbox('Enter Market Cap', market_caps)

    if None in [open_price, high_price, low_price, close_price, volume, market_cap]:
        st.warning('Please enter custom values')
    else:
        selected_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']
        X = filtered_data[selected_features]
        y = filtered_data['Close']  # Assuming 'Close' price is the target variable

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize the model (Random Forest Regressor)
        model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Train the model
        model.fit(X_train, y_train)

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
        st.markdown('---')
        st.success('Predicted Close Price:')
        st.title(prediction[0])
else:
    st.warning('Please select a Currency Name')
