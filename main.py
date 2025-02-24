import streamlit as st
from datetime import date, timedelta
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px

# Get current Date and up to 10 years ago
TODAY = date.today().strftime("%Y-%m-%d")
START = (date.today() - timedelta(days=365 * 10)).strftime("%Y-%m-%d")

st.title('Stock Forecast App')

# Initialize session state for user-added tickers
if "user_tickers" not in st.session_state:
    st.session_state.user_tickers = []  # Stores tickers added during session

# Predefined stock tickers
default_stocks = ['AAPL', 'GOOG', 'MSFT', 'NVDA']

# User input for custom ticker
new_ticker = st.text_input("Enter a stock ticker (e.g., TSLA, NFLX)")

# Function to validate if a ticker is valid
def is_valid_ticker(ticker):
    try:
        stock_data = yf.Ticker(ticker).history(period="1d")
        return not stock_data.empty  # Returns True if valid, False if invalid
    except:
        return False

# Add new ticker to dropdown list if valid
if st.button("Add Ticker"):
    new_ticker = new_ticker.upper().strip()
    
    if new_ticker and new_ticker not in default_stocks + st.session_state.user_tickers:
        if is_valid_ticker(new_ticker):
            st.session_state.user_tickers.append(new_ticker)  # Add to session state
            st.success(f"Added {new_ticker} to the list!")
        else:
            st.error("Invalid ticker! Please enter a valid stock symbol.")
    elif new_ticker in default_stocks + st.session_state.user_tickers:
        st.warning("Ticker already in the list!")

# Combine default stocks with user-added stocks
all_tickers = default_stocks + st.session_state.user_tickers

# Dropdown menu with updated list, defaulting to GOOG
selected_stock = st.selectbox('Select dataset for prediction', all_tickers, index=all_tickers.index("AAPL"))

# Slider for years of prediction/raw data display
n_years = st.slider('Years of prediction:', 1, 10)
period = n_years * 365  # used for forecasting horizon

# Caching to avoid re-downloading
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    
    # Flatten column names if multi-indexed
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

# Check date format
data['Date'] = pd.to_datetime(data['Date'])

# Filter raw data to only show the last n_years
selected_start_date = pd.Timestamp(date.today() - timedelta(days=n_years * 365))
filtered_data = data[data['Date'] >= selected_start_date]

st.subheader(f'{selected_stock} Raw data')
st.write(filtered_data.tail())

# Raw data plot using Plotly Express
def plot_raw_data():
    fig = px.line(filtered_data, x='Date', y=['Open', 'Close'], title='Time Series Data with Rangeslider')
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    st.plotly_chart(fig)

# Call the function
plot_raw_data()

# Prepare data for Prophet forecasting using the filtered data
df_train = filtered_data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

# Forecast with Prophet
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display forecast data
st.subheader(f"{selected_stock} Forecast over the next {n_years} year(s)")
st.write(forecast.tail())

# Forecast Plot
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.subheader(f"{selected_stock} Forecast Trends in {n_years} year(s)")
fig2 = m.plot_components(forecast)
st.write(fig2)
