import os
import time
import random
import requests
import numpy as np
import pandas as pd
from flask import Flask, jsonify
from datetime import datetime
from threading import Thread
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

app = Flask(__name__)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
MODEL_PATH = "model/lstm_model.h5"
DATA_PATH = "data/past_results.csv"

def fetch_real_data():
    return [random.randint(0, 9) for _ in range(100)]

def send_telegram(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": text})

def save_data(new_results):
    if not os.path.exists("data"):
        os.makedirs("data")
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.DataFrame(columns=["timestamp", "result"])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for val in new_results:
        df = pd.concat([df, pd.DataFrame([[now, val]], columns=["timestamp", "result"])], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)

def prepare_data():
    df = pd.read_csv(DATA_PATH)
    sequence_length = 10
    X, y = [], []
    data = df["result"].values
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    X = np.array(X).reshape(-1, 10, 1) / 9.0
    y = np.array(y)
    return X, y

def train_model():
    X, y = prepare_data()
    model = Sequential()
    model.add(LSTM(64, input_shape=(10, 1)))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if not os.path.exists("model"):
        os.makedirs("model")
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)
    model.save(MODEL_PATH)
    return model

def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    else:
        return train_model()

def predict_next(model):
    df = pd.read_csv(DATA_PATH)
    last_seq = df["result"].values[-10:]
    input_seq = np.array(last_seq).reshape(1, 10, 1) / 9.0
    prediction = model.predict(input_seq)[0]
    confidence = np.max(prediction)
    pred_val = np.argmax(prediction)
    return pred_val, confidence

def color_map(n):
    return "Green" if n % 2 == 0 else "Red"

def big_small(n):
    return "Big" if n >= 5 else "Small"

def prediction_loop():
    while True:
        try:
            new_data = fetch_real_data()
            save_data(new_data)
            model = load_or_train_model()
            pred, conf = predict_next(model)
            color = color_map(pred)
            size = big_small(pred)
            msg = f"🎯 Prediction: {pred} ({size}, {color})\nConfidence: {round(conf*100,2)}%\nTime: {datetime.now().strftime('%H:%M:%S')}"
            send_telegram(msg)
            if conf > 0.90:
                send_telegram(f"🚨 High Confidence Alert: {pred} is likely!\nConfidence: {round(conf*100,2)}%")
            time.sleep(60)
        except Exception as e:
            print(f"Prediction Error: {e}")
            time.sleep(30)

@app.route('/')
def home():
    return "✅ Wingo Predictor Bot Running!"

@app.route('/summary')
def weekly_summary():
    df = pd.read_csv(DATA_PATH)
    last_7_days = df.tail(7*24*60)
    msg = f"📅 Weekly Total Predictions: {len(last_7_days)}"
    send_telegram(msg)
    return "✅ Weekly Summary Sent!"

@app.route('/pattern')
def pattern_analyzer():
    df = pd.read_csv(DATA_PATH)
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    common_by_hour = df.groupby('hour')['result'].agg(lambda x: x.value_counts().index[0])
    msg = "📈 Pattern Analyzer (Top Result Per Hour):\n"
    for hour, res in common_by_hour.items():
        msg += f"{hour}:00 - {res}\n"
    send_telegram(msg)
    return "✅ Pattern Analyzed"

@app.route('/accuracy')
def live_accuracy():
    df = pd.read_csv(DATA_PATH)
    total = len(df)
    big = len(df[df['result'] >= 5])
    even = len(df[df['result'] % 2 == 0])
    msg = f"📊 Live Stats:\nTotal: {total}\nBig: {round((big/total)*100,2)}%\nGreen: {round((even/total)*100,2)}%"
    send_telegram(msg)
    return "✅ Accuracy Report Sent"

if __name__ == "__main__":
    Thread(target=prediction_loop).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
    import requests

TELEGRAM_TOKEN = "7751314755:AAFXmYJ2lW7xZhU7Txl1JuqCxG8LfbKmNZM"
CHAT_ID = "6848807471"

def send_prediction_to_telegram(period, number, size, color, confidence=None):
    message = f"""
🎯 *Wingo Prediction*

🆔 Period: `{period}`
🎲 Result: `{number}` ({size})
🎨 Color: `{color}`
{f'✅ Confidence: {confidence}%' if confidence else ''}
"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, data=data)
        print("Telegram response:", response.text)
    except Exception as e:
        print("Telegram error:", e)

