import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from datetime import timedelta
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set Pandas option to avoid silent downcasting
pd.set_option('future.no_silent_downcasting', True)


class BitcoinPricePredictor:
    def __init__(self, csv_file):
        """
        Initializes the Bitcoin price predictor.
        """
        self.csv_file = csv_file
        self.data = self.load_and_preprocess_data(csv_file)
        self.model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
        self.features = ['Open', 'High', 'Low', 'Volume', 'Price_Lag1', 'Price_Lag2', 'Price_Lag3',
                         'Volume_Lag1', 'Volume_Lag2', 'Volatility', 'RSI', 'MACD', 'MA_7', 'MA_30']
        self.scaler = StandardScaler()
        self.train_model()  # Train the model during initialization

    def load_and_preprocess_data(self, csv_file):
        """
        Loads and preprocesses data.
        """
        try:
            df = pd.read_csv(csv_file)
            print(f"Successfully loaded data with {len(df)} rows.")
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_file}")
            return pd.DataFrame()
        except pd.errors.EmptyDataError:
            print(f"Error: CSV file is empty: {csv_file}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return pd.DataFrame()

        if df.empty:
            print("DataFrame is empty after loading.")
            return df

        df['Open Time'] = pd.to_datetime(df['Open Time'], errors='coerce')
        df.dropna(subset=['Open Time'], inplace=True)
        df.set_index('Open Time', inplace=True)

        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'Close Time' in df.columns:
            df['Close Time'] = pd.to_datetime(df['Close Time'], errors='coerce')

        df.sort_index(inplace=True)
        return df

    def create_features(self, df):
        """
        Creates features, using ewm for moving averages.
        """
        result_df = df.copy()

        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in result_df.columns:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce')

        result_df['Price_Lag1'] = result_df['Close'].shift(1)
        result_df['Price_Lag2'] = result_df['Close'].shift(2)
        result_df['Price_Lag3'] = result_df['Close'].shift(3)
        result_df['Volume_Lag1'] = result_df['Volume'].shift(1)
        result_df['Volume_Lag2'] = result_df['Volume'].shift(2)

        # Use ewm (Exponentially Weighted Moving Average)
        result_df['MA_7'] = result_df['Close'].ewm(span=7, adjust=False).mean()
        result_df['MA_30'] = result_df['Close'].ewm(span=30, adjust=False).mean()

        if len(result_df) >= 7:
             result_df['Volatility'] = result_df['Close'].rolling(window=7).std()
        else:
             result_df['Volatility'] = np.nan

        result_df['RSI'] = self.calculate_rsi(result_df)
        macd_df = self.calculate_macd(result_df)
        result_df['MACD'] = macd_df['MACD']
        result_df['Signal_Line'] = macd_df['Signal']

        tech_indicator_cols = ['MA_7', 'MA_30', 'RSI', 'MACD', 'Signal_Line', 'Volatility']
        for col in tech_indicator_cols:
            if col in result_df.columns:
                result_df[col] = result_df[col].ffill()

        lag_cols = [col for col in result_df.columns if 'Lag' in col]
        for col in lag_cols:
            result_df[col] = result_df[col].bfill()

        numeric_columns = result_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            median_value = result_df[col].median()
            result_df[col] = result_df[col].fillna(median_value)

        remaining_na_cols = result_df.columns[result_df.isna().any()]
        for col in remaining_na_cols:
            result_df[col] = result_df[col].ffill().bfill()

        return result_df


    def calculate_rsi(self, df, period=14):
        """
        Calculates RSI with the most robust NaN handling.
        """
        close_series = pd.to_numeric(df['Close'], errors='coerce')
        delta = close_series.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        if len(df) >= period:
            # Check for valid (non-NaN) data *before* rolling
            if gain.notna().any() and loss.notna().any():
                avg_gain = gain.rolling(window=period, min_periods=1).mean()
                avg_loss = loss.rolling(window=period, min_periods=1).mean()
                avg_loss = avg_loss.replace(0, 0.001)  # Keep this!
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                return rsi
            else:
                return pd.Series(np.nan, index=df.index) # if not enough non-NaN values
        else:
            return pd.Series(np.nan, index=df.index)


    def calculate_macd(self, df, span1=12, span2=26, signal=9):
        """
        Calculates MACD.
        """
        close_series = pd.to_numeric(df['Close'], errors='coerce')
        # No need to check lengths here, ewm handles short series gracefully
        exp12 = close_series.ewm(span=span1, adjust=False).mean()
        exp26 = close_series.ewm(span=span2, adjust=False).mean()
        macd = exp12 - exp26
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return pd.DataFrame({'MACD': macd, 'Signal': signal_line}, index=df.index)

    def train_model(self):
        """
        Prepares data and trains the model.
        """
        df = self.create_features(self.data)
        df.dropna(inplace=True)

        if len(df) < 50:
            raise ValueError(f"Insufficient data after preprocessing: only {len(df)} rows remain")

        X = df[self.features]
        y = df['Close']

        for col in X.columns:
            X.loc[:, col] = pd.to_numeric(X[col], errors='coerce')
        y = pd.to_numeric(y, errors='coerce')

        valid_indices = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_indices]
        y = y[valid_indices]

        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.features, index=X.index)

        self.model.fit(X_scaled, y)
        print(f"Model trained on {X_scaled.shape[0]} samples.")

        # Feature importance
        feature_importance = pd.DataFrame(
            self.model.feature_importances_,
            index=self.features,
            columns=['importance']
        ).sort_values('importance', ascending=False)
        print("\nFeature Importance:")
        print(feature_importance.head(5))

    def predict_next_price(self):
        """
        Predicts the next 15-minute price.
        """
        last_data = self.data.iloc[-5:].copy()

        # Robust temporary NaN filling
        last_data = last_data.ffill().bfill()
        numeric_cols = last_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
             if last_data[col].isnull().any():
                last_data[col] = last_data[col].fillna(last_data[col].median())

        temp_df = self.create_features(last_data)
        latest_features = temp_df.iloc[[-1]][self.features]
        scaled_features = self.scaler.transform(latest_features)
        predicted_price = self.model.predict(scaled_features)[0]

        lower_bound = predicted_price * 0.95
        upper_bound = predicted_price * 1.05
        last_timestamp = self.data.index[-1]
        next_timestamp = last_timestamp + timedelta(minutes=15)

        return next_timestamp, predicted_price, lower_bound, upper_bound
    def save_prediction_to_csv(self, next_timestamp, predicted_price, lower_bound, upper_bound,
                               output_file='btcusdt_next_prediction.csv'):
        """Saves the single prediction."""
        full_prediction = pd.DataFrame(index=[0])  # Single-row DataFrame
        full_prediction['Open Time'] = next_timestamp
        full_prediction['Open'] = predicted_price * 0.995  # Example, adjust as needed
        full_prediction['High'] = upper_bound
        full_prediction['Low'] = lower_bound
        full_prediction['Close'] = predicted_price
        full_prediction['Volume'] = self.data['Volume'].mean()  # Use historical average
        full_prediction['Close Time'] = next_timestamp + timedelta(minutes=15) - timedelta(milliseconds=1)

        # Add other required columns (simplified, as it's a single prediction)
        full_prediction['Quote Asset Volume'] = full_prediction['Volume'] * full_prediction['Close']
        full_prediction['Number of Trades'] = int(self.data['Number of Trades'].mean())
        full_prediction['Taker Buy Base Asset Volume'] = full_prediction['Volume'] * 0.5
        full_prediction['Taker Buy Quote Asset Volume'] = full_prediction['Taker Buy Base Asset Volume'] * full_prediction['Close']
        full_prediction['Ignore'] = 0

        required_columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time',
                            'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume',
                            'Taker Buy Quote Asset Volume', 'Ignore']

        # Ensure all required columns
        for col in required_columns:
            if col not in full_prediction.columns:
                full_prediction[col] = 0  # Or appropriate default

        # Save
        full_prediction.to_csv(output_file, index=False)
        print(f"Next 15-minute prediction saved to {output_file}")
        return full_prediction

    def visualize_prediction(self, next_timestamp, predicted_price, lower_bound, upper_bound,
                             output_file='img/bitcoin_next_price_forecast.png'):
        """
        Visualizes the historical data (last 24 hours) and the single next prediction.
        """
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})

            # --- Plot 1: Historical (Last 24 Hours) and Predicted ---
            # Calculate the start time for the historical data (24 hours before the last timestamp)
            start_time = self.data.index[-1] - timedelta(hours=24)
            historical_data = self.data[self.data.index >= start_time]

            ax1.plot(historical_data.index, historical_data['Close'], label='Historical Price', color='blue')

            # Plot the single prediction point
            ax1.plot([next_timestamp], [predicted_price], 'ro', label='Predicted Price')
            ax1.vlines(x=next_timestamp, ymin=lower_bound, ymax=upper_bound, color='red', alpha=0.5, label='Confidence Interval')

            ax1.set_title('Bitcoin Price Forecast (Next 15 Minutes)', fontsize=16)
            ax1.set_ylabel('Price (USD)', fontsize=12)
            ax1.legend(loc='best')
            ax1.grid(True)
            # Format x-axis to show more readable dates and times
            ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
            fig.autofmt_xdate()  # Rotate date labels for better readability


            # --- Plot 2: Feature importance (Same as before) ---
            feature_importance = pd.DataFrame(
                self.model.feature_importances_,
                index=self.features,
                columns=['importance']
            ).sort_values('importance', ascending=False)

            top_features = feature_importance.head(8)
            bars = ax2.bar(top_features.index, top_features['importance'])

            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.3f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),
                             textcoords="offset points",
                             ha='center', va='bottom', rotation=0)

            ax2.set_title('Top Feature Importance', fontsize=14)
            ax2.set_ylabel('Importance', fontsize=12)
            ax2.grid(axis='y')


            plt.tight_layout()
            plt.savefig(output_file)
            print(f"Visualization saved to {output_file}")
            plt.close()
        except Exception as e:
            print(f"Error in visualization: {e}")


# Example Usage
if __name__ == "__main__":
    try:
        predictor = BitcoinPricePredictor('btcusdt_15minute_data.csv')
        next_timestamp, predicted_price, lower_bound, upper_bound = predictor.predict_next_price()
        predictor.save_prediction_to_csv(next_timestamp, predicted_price, lower_bound, upper_bound)
        predictor.visualize_prediction(next_timestamp, predicted_price, lower_bound, upper_bound)

        print(f"\nNext 15-minute Prediction (at {next_timestamp}):")
        print(f"  Predicted Close: {predicted_price:.2f}")
        print(f"  Lower Bound: {lower_bound:.2f}")
        print(f"  Upper Bound: {upper_bound:.2f}")

    except Exception as e:
        print(f"An error occurred: {e}")