import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('consolidated_coin_data.csv')

# Data preprocessing steps
data['Date'] = pd.to_datetime(data['Date'])

# Replace commas in numeric columns and convert to float
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']

for col in numeric_columns:
    data[col] = data[col].str.replace(',', '').astype(float)

# Streamlit UI
st.set_page_config(page_title='Cryptocurrency Price Predictor', layout='wide')
st.title('Cryptocurrency Price Predictor')
st.markdown("---")

# Sidebar layout
st.sidebar.title('Enter Data for Prediction')

# Currency name dropdown
currency_names = data['Currency'].unique()
selected_currency = st.sidebar.selectbox('Select Currency Name', currency_names)

if selected_currency:
    filtered_data = data[data['Currency'] == selected_currency]

    # Display the first two lines of data for the selected currency
    st.subheader('First Two Lines of Data:')
    st.table(filtered_data.head(2))

    # Model selection
    st.sidebar.subheader('Select Model:')
    model_options = ['Random Forest', 'Gradient Boosting', 'Support Vector Machine']
    selected_model = st.sidebar.selectbox('Choose a Model', model_options)

    if selected_model == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif selected_model == 'Gradient Boosting':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif selected_model == 'Support Vector Machine':
        model = SVR()

    # User input for historical prices
    st.sidebar.subheader('Enter Custom Historical Prices:')
    open_price = st.sidebar.number_input('Open Price', min_value=0.0)
    high_price = st.sidebar.number_input('High Price', min_value=0.0)
    low_price = st.sidebar.number_input('Low Price', min_value=0.0)
    close_price = st.sidebar.number_input('Close Price', min_value=0.0)
    volume = st.sidebar.number_input('Volume', min_value=0)
    market_cap = st.sidebar.number_input('Market Cap', min_value=0)

    # User input for model selection
    st.sidebar.subheader('Model Parameters:')
    # Add parameters for the selected model if needed

    user_input = pd.DataFrame({
        'Open': [open_price],
        'High': [high_price],
        'Low': [low_price],
        'Close': [close_price],
        'Volume': [volume],
        'Market Cap': [market_cap]
    })

    if st.sidebar.button('Predict'):
        # Split data into training and testing sets
        X = filtered_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']]
        y = filtered_data['Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        prediction = model.predict(user_input)

        # Display prediction to the user
        st.markdown('---')
        st.success('Predicted Close Price:')
        st.title(f"${prediction[0]:,.2f}")

        # Visualize predictions vs. actual values
        st.subheader('Predictions vs. Actual Values:')
        plt.figure(figsize=(10, 5))
        plt.plot(filtered_data['Date'], filtered_data['Close'], label='Actual Close Price', color='blue')
        plt.scatter(filtered_data['Date'].iloc[-1], prediction[0], label='Predicted Close Price', color='red')
        plt.title('Predicted vs. Actual Close Price Over Time')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        st.pyplot(plt)

        # Model evaluation
        st.subheader("Model Performance Metrics")
        st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, model.predict(X_test)):.2f}")
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, model.predict(X_test)):.2f}")
        st.write(f"R-squared: {r2_score(y_test, model.predict(X_test)):.2f}")

    # Plot historical prices
    st.subheader('Historical Prices:')
    plt.figure(figsize=(10, 5))
    plt.plot(filtered_data['Date'], filtered_data['Close'], label='Close Price', color='green')
    plt.title('Historical Close Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    st.pyplot(plt)

else:
    st.warning('Please select a Currency Name')
