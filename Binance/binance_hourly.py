from binance.client import Client
import pandas as pd
import datetime
from keys import api_key, api_secret

# Initialize Binance client with API credentials
client = Client(api_key, api_secret)

# Get hourly historical data
historical = client.get_historical_klines('BTCUSDT', Client.KLINE_INTERVAL_1DAY, "1 Jan 2022")
historical_pd = pd.DataFrame(historical)

# Set column names
columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time',
          'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume',
          'Taker Buy Quote Asset Volume', 'Ignore']
historical_pd.columns = columns

# Convert timestamps to datetime
historical_pd['Open Time'] = pd.to_datetime(historical_pd['Open Time'], unit='ms')
historical_pd['Close Time'] = pd.to_datetime(historical_pd['Close Time'], unit='ms')

# Convert price and volume columns to float
for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume',
            'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume']:
    historical_pd[col] = historical_pd[col].astype(float)

# Save to CSV file
output_file = 'btcusdt_hourly_data.csv'
historical_pd.to_csv(output_file, index=False)
print(f'Hourly data has been saved to {output_file}')

# Display first few rows of the data
print("\nFirst few rows of the hourly data:")
print(historical_pd.head())