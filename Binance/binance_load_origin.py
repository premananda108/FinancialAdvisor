from mock_binance import MockClient
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # для форматирования дат на оси x

client = MockClient()

historical = client.get_historical_klines('BTCUSDT', Client.KLINE_INTERVAL_1DAY, "1 Jan 2022")
historical_pd = pd.DataFrame(historical)

# Назначение имен столбцов для DataFrame (чтобы было понятнее)
historical_pd.columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore']

# Преобразование столбца 'Open Time' в datetime и установка его в качестве индекса
historical_pd['Open Time'] = pd.to_datetime(historical_pd['Open Time'], unit='ms') # unit='ms' потому что Binance возвращает timestamp в миллисекундах
historical_pd = historical_pd.set_index('Open Time')

# Преобразование столбца 'Close' в числовой тип (float)
historical_pd['Close'] = historical_pd['Close'].astype(float)

# Построение графика
plt.figure(figsize=(12, 6)) # Установка размера графика (опционально)
plt.plot(historical_pd.index, historical_pd['Close']) # Строим график: ось x - даты (индекс), ось y - цена закрытия
plt.xlabel('Дата') # Подпись оси x
plt.ylabel('Цена закрытия (USDT)') # Подпись оси y
plt.title('График цены BTCUSDT (с 1 января 2022)') # Заголовок графика
plt.grid(True) # Добавление сетки для лучшей читаемости (опционально)

# Форматирование оси x для дат (опционально, но полезно для читаемости)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # Формат даты: Год-Месяц-День
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator()) # Автоматическое определение расположения меток на оси x
plt.gcf().autofmt_xdate() # Автоматический поворот меток на оси x для предотвращения перекрытия

plt.tight_layout() # Автоматическая корректировка отступов для предотвращения обрезания меток
plt.show() # Отображение графика