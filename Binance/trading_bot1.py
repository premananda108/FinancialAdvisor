import requests
import time


def send_telegram_message(bot_token, chat_id, message):
    """
    Отправляет сообщение в Telegram с использованием HTTP API.

    Args:
        bot_token (str): Токен вашего бота, полученный от BotFather
        chat_id (str): ID чата/пользователя для отправки сообщения
        message (str): Текст сообщения для отправки

    Returns:
        dict: Ответ API Telegram в виде словаря
    """
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }

    response = requests.post(url, data=payload)
    return response.json()


# Замените на ваш токен, полученный от BotFather
BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"

# Ваш ID пользователя
CHAT_ID = "7934283421"


def main():
    # Пример отправки простого сообщения
    message = "Привет! Это тестовое сообщение от вашего бота."

    try:
        result = send_telegram_message(BOT_TOKEN, CHAT_ID, message)
        if result.get("ok"):
            print(f"Сообщение успешно отправлено пользователю {CHAT_ID}")
        else:
            print(f"Ошибка при отправке сообщения: {result.get('description')}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    main()