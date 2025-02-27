from mock_binance import MockClient
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class TradingAlgorithm:
    def __init__(self, start_balance, symbol='BTCUSDT'):
        self.client = MockClient()
        self.symbol = symbol
        self.initial_balance = start_balance
        self.balance = start_balance
        self.in_position = False
        self.entry_price = 0
        self.trade_history = []

    def get_historical_data(self, interval, start_date):
        """Получение исторических данных"""
        historical = self.client.get_historical_klines(self.symbol, interval, start_date)
        df = pd.DataFrame(historical)

        # Назначение имен столбцов
        df.columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                      'Close Time', 'Quote Asset Volume', 'Number of Trades',
                      'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore']

        # Преобразование временной метки в datetime
        df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
        df = df.set_index('Open Time')

        # Преобразование строковых значений в числовые
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)

        return df

    def execute_trade(self, df, take_profit_pct=1.0, stop_loss_pct=1.0, start_date='01.01.2022', end_date='31.12.2022'):
        """
        Выполнение торговой логики

        Args:
            df: DataFrame с историческими данными
            take_profit_pct: процент тейк-профита
            stop_loss_pct: процент стоп-лосса
            start_date: начальная дата в формате 'DD.MM.YYYY'
            end_date: конечная дата в формате 'DD.MM.YYYY'
        """
        # Преобразуем строковые даты в datetime
        start_dt = pd.to_datetime(start_date, format='%d.%m.%Y')
        end_dt = pd.to_datetime(end_date, format='%d.%m.%Y')

        # Фильтруем DataFrame по датам
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]

        # Пропускаем первую строку, так как для неё нет предыдущей цены
        for i in range(1, len(df)):
            current_row = df.iloc[i]
            current_price = float(current_row['Close'])

            # Получаем предыдущую цену закрытия
            previous_price = float(df.iloc[i - 1]['Close'])

            # Условие входа в позицию - цена растёт
            #entry_conditions_met = current_price > previous_price
            entry_conditions_met = True

            if not self.in_position and entry_conditions_met:
                # Открываем лонг позицию
                self.entry_price = current_price
                self.in_position = True
                self.trade_history.append({
                    'date': df.index[i],
                    'action': 'BUY',
                    'price': current_price
                })

            elif self.in_position:
                # Проверяем условия выхода
                take_profit = self.entry_price * (1 + take_profit_pct / 100)
                stop_loss = self.entry_price * (1 - stop_loss_pct / 100)

                if current_price >= take_profit:
                    profit = ((current_price - self.entry_price) / self.entry_price) * 100
                    self.balance *= (1 + profit / 100)
                    self.in_position = False
                    self.trade_history.append({
                        'date': df.index[i],
                        'action': 'SELL',
                        'price': current_price,
                        'result': 'TAKE PROFIT'
                    })

                elif current_price <= stop_loss:
                    loss = ((self.entry_price - current_price) / self.entry_price) * 100
                    self.balance *= (1 - loss / 100)
                    self.in_position = False
                    self.trade_history.append({
                        'date': df.index[i],
                        'action': 'SELL',
                        'price': current_price,
                        'result': 'STOP LOSS'
                    })

    def plot_results(self, df):
        """Построение графика с результатами торговли.
        Зелёный кружочек - прибыльная сделка, красный кружочек - убыточная сделка"""
        plt.figure(figsize=(12, 8))  # Увеличим размер графика
        plt.plot(df.index, df['Close'])

        # Отображение кружочков посередине сделки
        for i in range(0, len(self.trade_history), 2):
            # Проверяем, что есть и вход, и выход
            if i + 1 >= len(self.trade_history):
                break

            buy_trade = self.trade_history[i]
            sell_trade = self.trade_history[i + 1]

            # Вычисляем среднюю дату и цену для позиции кружочка
            mid_date = buy_trade['date'] + (sell_trade['date'] - buy_trade['date']) / 2
            mid_price = (buy_trade['price'] + sell_trade['price']) / 2

            # Определяем прибыльность сделки
            if buy_trade['action'] == 'BUY':  # Long позиция
                is_profitable = sell_trade['price'] > buy_trade['price']
            else:  # Short позиция (если бы она была реализована)
                is_profitable = sell_trade['price'] < buy_trade['price']

            # Выбираем цвет кружочка
            circle_color = 'green' if is_profitable else 'red'

            # Отображаем кружочек
            plt.scatter(mid_date, mid_price,
                        color=circle_color,
                        marker='o',  # Кружочек
                        s=100)

        plt.title(f'Результаты торговли {self.symbol}')
        plt.xlabel('Дата')
        plt.ylabel('Цена (USDT)')
        plt.grid(True)
        #plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()

        # Увеличим отступы
        plt.subplots_adjust(bottom=0.2, top=0.9)

        plt.show()

    def print_trade_results(self, df, start_date, end_date):
        """Вывод текстового отчета о сделках и анализа стоимости BTC"""
        print("\n=== ОТЧЕТ О ТОРГОВЛЕ ===")
        print(f"Инструмент: {self.symbol}")
        print(f"Начальный баланс: {self.initial_balance:.2f} USDT")
        print(f"Конечный баланс: {self.balance:.2f} USDT")

        total_profit = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        print(f"Общий результат: {total_profit:.2f}%")

        # Преобразуем строковые даты в datetime для фильтрации DataFrame
        start_dt = pd.to_datetime(start_date, format='%d.%m.%Y')
        end_dt = pd.to_datetime(end_date, format='%d.%m.%Y')

        # Фильтруем DataFrame для получения данных только за указанный период
        period_df = df[(df.index >= start_dt) & (df.index <= end_dt)]

        # Получаем первую и последнюю цену за указанный период
        first_price = period_df.iloc[0]['Close']
        last_price = period_df.iloc[-1]['Close']

        # Рассчитываем количество BTC, которое можно купить на начальный баланс
        possible_btc_start = self.initial_balance / first_price

        # Рассчитываем количество BTC, которое можно купить на конечный баланс
        possible_btc_end = self.balance / last_price

        first_date = period_df.index[0].strftime('%d.%m.%Y')
        last_date = period_df.index[-1].strftime('%d.%m.%Y')

        print(f"\nВозможное количество BTC:")
        print(
            f"На начальную сумму {self.initial_balance:.2f} USDT по цене на {first_date} ({first_price:.2f} USDT): {possible_btc_start:.8f} BTC")
        print(
            f"На конечную сумму {self.balance:.2f} USDT по цене на {last_date} ({last_price:.2f} USDT): {possible_btc_end:.8f} BTC")
        print(f"Изменение стоимости BTC за период: {((last_price - first_price) / first_price * 100):.2f}%")

        # Сравниваем стратегию с простой покупкой и удержанием
        hodl_value = possible_btc_start * last_price
        hodl_profit = ((hodl_value - self.initial_balance) / self.initial_balance) * 100
        print(f"Стратегия HODL (купить и держать): {hodl_profit:.2f}%")

        if total_profit > hodl_profit:
            print(f"Ваша стратегия превосходит HODL на {(total_profit - hodl_profit):.2f}%")
        else:
            print(f"HODL превосходит вашу стратегию на {(hodl_profit - total_profit):.2f}%")

        print("-" * 80)

        print("\nСписок сделок:")
        print("-" * 80)
        print(f"{'Дата':25} {'Действие':10} {'Цена':15} {'Результат':15} {'П/У %':10}")
        print("-" * 80)

        entry_price = None
        for trade in self.trade_history:
            date_str = trade['date'].strftime('%Y-%m-%d %H:%M:%S')
            action = trade['action']
            price = trade['price']

            if action == 'BUY':
                entry_price = price
                print(f"{date_str:25} {'ВХОД':10} {price:15.2f}")
            else:  # SELL
                profit_loss = ((price - entry_price) / entry_price) * 100
                result = trade.get('result', '')
                print(f"{date_str:25} {'ВЫХОД':10} {price:15.2f} {result:15} {profit_loss:10.2f}%")

        print("-" * 80)

        # Статистика
        profitable_trades = sum(1 for trade in self.trade_history if
                                trade['action'] == 'SELL' and
                                ((trade['price'] - self.trade_history[self.trade_history.index(trade) - 1][
                                    'price']) > 0))
        total_trades = len([trade for trade in self.trade_history if trade['action'] == 'SELL'])

        if total_trades > 0:
            win_rate = (profitable_trades / total_trades) * 100
            print(f"\nСтатистика:")
            print(f"Всего сделок: {total_trades}")
            print(f"Прибыльных сделок: {profitable_trades}")
            print(f"Процент успешных сделок: {win_rate:.2f}%")

# Пример использования
if __name__ == "__main__":
    start_date = '01.10.2022'
    end_date = '01.04.2023'
    start_balance = 100

    bot = TradingAlgorithm(start_balance)
    historical_data = bot.get_historical_data(bot.client.KLINE_INTERVAL_1DAY, "1 Jan 2022")
    bot.execute_trade(
        historical_data,
        take_profit_pct=1.0,
        stop_loss_pct=1.0,
        start_date=start_date,
        end_date=end_date
    )
    bot.plot_results(historical_data)
    bot.print_trade_results(historical_data, start_date, end_date)
