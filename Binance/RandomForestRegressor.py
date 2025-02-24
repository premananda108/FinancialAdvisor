import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set Pandas option to avoid silent downcasting
pd.set_option('future.no_silent_downcasting', True)

class BitcoinPricePredictor:
    def __init__(self, csv_file, forecast_window=30):
        """
        Initializes the Bitcoin price predictor with data from a CSV file and sets the forecast window.
        """
        self.csv_file = csv_file
        self.data = self.load_and_preprocess_data(csv_file)
        self.forecast_window = forecast_window
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)  # Use RandomForestRegressor
        self.features = ['Open', 'High', 'Low', 'Volume', 'Price_Lag1', 'Price_Lag2', 'Volume_Lag1', 'Volatility', 'RSI', 'MACD']  # Feature list
        self.prepare_training_data()

    def load_and_preprocess_data(self, csv_file):
        """
        Loads data from the CSV file, preprocesses it, and sets the timestamp as index.
        """
        try:
            df = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_file}")
            return pd.DataFrame()  # Return an empty DataFrame
        except pd.errors.EmptyDataError:
            print(f"Error: CSV file is empty: {csv_file}")
            return pd.DataFrame()  # Return an empty DataFrame
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return pd.DataFrame()

        if df.empty:
            print("DataFrame is empty after loading.")
            return df

        df['Open Time'] = pd.to_datetime(df['Open Time'], errors='coerce') # errors='coerce' to handle unparseable dates
        df.dropna(subset=['Open Time'], inplace=True) # Drop rows where 'Open Time' could not be parsed

        df.set_index('Open Time', inplace=True)
        df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        return df

    def create_features(self, df):
        """
        Creates lagged features for time series prediction.
        """
        df['Price_Lag1'] = df['Close'].shift(1)
        df['Price_Lag2'] = df['Close'].shift(2)
        df['Volume_Lag1'] = df['Volume'].shift(1)
        df['Volatility'] = (df['High'] - df['Low']).rolling(window=7).mean()
        df['RSI'] = self.calculate_rsi(df)
        macd_df = self.calculate_macd(df)
        df['MACD'] = macd_df['MACD']
        df['Signal_Line'] = macd_df['Signal']
        df.fillna(0, inplace=True)  # Fill NaN with 0
        return df

    def calculate_rsi(self, df, period=14):
        """
        Calculates the Relative Strength Index (RSI).
        """
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_macd(self, df, span1=12, span2=26, signal=9):
        """
        Calculates the Moving Average Convergence Divergence (MACD).
        """
        exp12 = df['Close'].ewm(span=span1, adjust=False).mean()
        exp26 = df['Close'].ewm(span=span2, adjust=False).mean()
        macd = exp12 - exp26
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return pd.DataFrame({'MACD': macd, 'Signal': signal_line}, index=df.index)

    def prepare_training_data(self):
        """
        Prepares the training data for the machine learning model.
        """
        self.data = self.create_features(self.data)

        # Define features and target variable
        features = ['Open', 'High', 'Low', 'Volume', 'Price_Lag1', 'Price_Lag2', 'Volume_Lag1', 'Volatility', 'RSI', 'MACD']
        self.features = features
        target = 'Close'

        # Remove rows with NaN values resulting from lag
        self.data.dropna(inplace=True)

        # Prepare data for the model
        X = self.data[features]
        y = self.data[target]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

        # Train the model
        self.model.fit(X_train, y_train)
        self.X_test = X_test

        # Evaluate the model
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")

    def predict_future_prices(self):
        """
        Predicts future Bitcoin prices for the next month.
        """
        # Get the last row of data as DataFrame
        last_data_df = self.data.iloc[[-1]].copy()
        future_dates = [self.data.index[-1] + timedelta(hours=i) for i in range(1, self.forecast_window*24)]
        future_prices = []
        feature_names = self.features  # Use self.features

        for i in range(self.forecast_window * 24 - 1):

            predicted_price = self.model.predict(last_data_df[feature_names].values)[0]
            future_prices.append(predicted_price)

            new_date = future_dates[i]
            new_data = pd.DataFrame(index=[new_date], columns=self.data.columns)  # Create DataFrame with all columns

            new_data['Open'] = predicted_price
            new_data['High'] = predicted_price * 1.01  # Estimate
            new_data['Low'] = predicted_price * 0.99  # Estimate
            new_data['Close'] = predicted_price
            new_data['Volume'] = self.data['Volume'].mean()

            #Lag features from PREVIOUS data
            new_data['Price_Lag1'] = last_data_df['Close'].iloc[0]
            new_data['Price_Lag2'] = last_data_df['Price_Lag1'].iloc[0]
            new_data['Volume_Lag1'] = last_data_df['Volume'].iloc[0]
            new_data['Volatility'] = last_data_df['Volatility'].iloc[0]
            new_data['RSI'] = last_data_df['RSI'].iloc[0]
            new_data['MACD'] = last_data_df['MACD'].iloc[0]

            new_df = self.create_features(new_data) # Create features
            new_df[feature_names] = new_df[feature_names].astype(float)
            # Update last_data_df for the next iteration
            last_data_df = new_df.iloc[[-1]]


        # Create a DataFrame for the predictions
        predictions_df = pd.DataFrame({'Open Time': future_dates[:720], 'Predicted Close': future_prices[:720]})
        predictions_df['Open Time'] = pd.to_datetime(predictions_df['Open Time'])
        predictions_df.set_index('Open Time', inplace=False) # Keep 'Open Time' as a Column, not index

        return predictions_df

    def save_predictions_to_csv(self, predictions_df, output_file='btcusdt_hourly_data_new.csv'):
        """
        Saves the predicted prices to a new CSV file.
        """
        # Create dummy columns to match original data structure
        predictions_df['Open'] = predictions_df['Predicted Close']
        predictions_df['High'] = predictions_df['Predicted Close'] * 1.01
        predictions_df['Low'] = predictions_df['Predicted Close'] * 0.99
        predictions_df['Close'] = predictions_df['Predicted Close']
        predictions_df['Volume'] = self.data['Volume'].mean()
        predictions_df['Close Time'] = predictions_df['Open Time']
        predictions_df['Quote Asset Volume'] = 0
        predictions_df['Number of Trades'] = 0
        predictions_df['Taker Buy Base Asset Volume'] = 0
        predictions_df['Taker Buy Quote Asset Volume'] = 0
        predictions_df['Ignore'] = 0

        predictions_df = predictions_df[['Open Time','Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore' ]] # Reorder columns

        predictions_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")


# Example usage (assuming you have a CSV file named 'btcusdt_hourly_data.csv' in the same directory)
predictor = BitcoinPricePredictor('btcusdt_hourly_data.csv')
future_prices = predictor.predict_future_prices()
predictor.save_predictions_to_csv(future_prices)