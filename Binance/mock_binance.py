import pandas as pd
import datetime

class MockClient:
    def __init__(self, api_key=None, api_secret=None):
        """Mock initialization that ignores credentials"""
        self.KLINE_INTERVAL_1DAY = '1d'
        self.KLINE_INTERVAL_1HOUR = '1h'
        self._load_data_from_csv()
    
    def _load_data_from_csv(self):
        """Load historical data from CSV file"""
        try:
            # Read the CSV file
            df = pd.read_csv('btcusdt_day_data.csv')
            
            # Convert timestamps to milliseconds
            df['Open Time'] = pd.to_datetime(df['Open Time']).astype('int64') // 10**6
            df['Close Time'] = pd.to_datetime(df['Close Time']).astype('int64') // 10**6
            
            # Convert DataFrame to list of lists format matching Binance API response
            self.mock_data = df[['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
                                'Close Time', 'Quote Asset Volume', 'Number of Trades',
                                'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume',
                                'Ignore']].values.tolist()
            
            # Convert numeric values to strings to match Binance API format
            self.mock_data = [
                [row[0]] + [str(x) for x in row[1:]] for row in self.mock_data
            ]
            
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            self.mock_data = []
    
    def get_historical_klines(self, symbol, interval, start_str, end_str=None):
        """Mock implementation of get_historical_klines"""
        if interval not in [self.KLINE_INTERVAL_1DAY, self.KLINE_INTERVAL_1HOUR]:
            raise ValueError("Only daily and hourly intervals are supported in mock")
            
        # Parse start date
        try:
            start_date = pd.to_datetime(start_str)
            start_timestamp = int(start_date.timestamp() * 1000)
        except:
            start_timestamp = 0
            
        # Filter data based on start date
        filtered_data = [kline for kline in self.mock_data 
                        if kline[0] >= start_timestamp]
        
        # If daily interval is requested, resample hourly data to daily
        if interval == self.KLINE_INTERVAL_1DAY:
            daily_data = []
            current_day = None
            day_data = None
            
            for kline in filtered_data:
                date = datetime.datetime.fromtimestamp(kline[0] / 1000).date()
                
                if date != current_day:
                    if day_data:
                        daily_data.append(day_data)
                    current_day = date
                    day_data = kline.copy()
                else:
                    # Update high and low
                    day_data[2] = str(max(float(day_data[2]), float(kline[2])))
                    day_data[3] = str(min(float(day_data[3]), float(kline[3])))
                    # Update close price and time
                    day_data[4] = kline[4]
                    day_data[6] = kline[6]
                    # Sum volumes and trades
                    day_data[5] = str(float(day_data[5]) + float(kline[5]))
                    day_data[7] = str(float(day_data[7]) + float(kline[7]))
                    day_data[8] = str(int(float(day_data[8]) + float(kline[8])))
                    day_data[9] = str(float(day_data[9]) + float(kline[9]))
                    day_data[10] = str(float(day_data[10]) + float(kline[10]))
            
            if day_data:
                daily_data.append(day_data)
            return daily_data
        
        return filtered_data