# main.py

import requests
import pandas as pd
import time
import json
import os
import random
import numpy as np
import threading
from datetime import datetime
from flask import Flask
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import telegram

# Telegram Config
BOT_TOKEN = "7751314755:AAFXmYJ2lW7xZhU7Txl1JuqCxG8LfbKmNZM"
CHAT_ID = "6848807471"
bot = telegram.Bot(token=BOT_TOKEN)

# Model Path
MODEL_PATH = "wingo_model.h5"
DATA_PATH = "wingo_data.csv"

# App
app = Flask(__name__)

# ------------------------- Helper Functions -------------------------

def fetch_latest_result():
    try:
        url = "https://api.wingo.one/minuteWin/getMinuteWinLatestResult"
        response = requests.get(url)
        data = response.json()['data']
        return {
            "period": data["periodNumber"],
            "number": int(data["number"]),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except:
        return None

def get_color(number):
    color_map = {
        0: "Green", 1: "Red", 2: "Violet", 3: "Green", 4: "Green",
        5: "Red", 6: "Green", 7: "Red", 8: "Green", 9: "Red"
    }
    return color_map.get(number, "Unknown")

def get_size(number):
    if number in [0, 1, 2, 3, 4]:
        return "Small"
    else:
        return "Big"

# ------------------------- Data + Training -------------------------

def save_result(data):
    new_row = {
        "period": data["period"],
        "number": data["number"],
        "color": get_color(data["number"]),
        "size": get_size(data["number"]),
        "time": data["time"]
    }
    df = pd.DataFrame([new_row])
    if os.path.exists(DATA_PATH):
        df.to_csv(DATA_PATH, mode='a', header=False, index=False)
    else:
        df.to_csv(DATA_PATH, index=False)

def load_data():
    if not os.path.exists(DATA_PATH):
        return None, None
    df = pd.read_csv(DATA_PATH)
    df.dropna(inplace=True)
    sequence_length = 10
    if len(df) < sequence_length + 1:
        return None, None
    X, y = [], []
    for i in range(sequence_length, len(df)):
        X.append(df['number'].iloc[i-sequence_length:i].values)
        y.append(df['number'].iloc[i])
    return np.array(X), np.array(y)

def train_and_save_model():
    X, y = load_data()
    if X is None:
        return
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = np.array(y)
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(64))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)
    model.save(MODEL_PATH)

def load_model_if_exists():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    return None

# ------------------------- Prediction + Report -------------------------

def predict_next(model):
    df = pd.read_csv(DATA_PATH)
    recent = df['number'].values[-10:]
    input_seq = np.array(recent).reshape((1, 10, 1))
    pred = model.predict(input_seq)[0]
    number = np.argmax(pred)
    confidence = pred[number]
    return number, confidence

def send_prediction(number, confidence):
    size = get_size(number)
    color = get_color(number)
    msg = f"""📢 *Wingo Prediction*
🕒 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🎯 Result: {number} ({size}, {color})"""
    bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
    if confidence >= 0.9:
        bot.send_message(chat_id=CHAT_ID, text=f"🔔 *High Confidence Alert*\nPrediction: {number} with {round(confidence*100,2)}%", parse_mode="Markdown")

def send_accuracy_report():
    if not os.path.exists(DATA_PATH):
        return
    df = pd.read_csv(DATA_PATH)
    total = len(df)
    big = len(df[df['size'] == 'Big'])
    small = len(df[df['size'] == 'Small'])
    green = len(df[df['color'] == 'Green'])
    red = len(df[df['color'] == 'Red'])
    msg = f"""📊 *Wingo Accuracy Summary*
Total Results: {total}
🟢 Green: {green} ({green*100//total}%)
🔴 Red: {red} ({red*100//total}%)
🔵 Big: {big} ({big*100//total}%)
🟡 Small: {small} ({small*100//total}%)"""
    bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")

def send_pattern_analysis():
    if not os.path.exists(DATA_PATH):
        return
    df = pd.read_csv(DATA_PATH)
    hourly = df.copy()
    hourly['hour'] = pd.to_datetime(hourly['time']).dt.hour
    pattern = hourly.groupby('hour')['number'].mean().reset_index()
    msg = "*📈 Pattern Analyzer*\n"
    for _, row in pattern.iterrows():
        msg += f"⏰ Hour {int(row['hour'])}: Avg = {round(row['number'], 2)}\n"
    bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")

# ------------------------- Background Loop -------------------------

def background_predict_loop():
    last_period = None
    while True:
        latest = fetch_latest_result()
        if latest and latest["period"] != last_period:
            save_result(latest)
            train_and_save_model()
            model = load_model_if_exists()
            if model:
                number, confidence = predict_next(model)
                send_prediction(number, confidence)
            last_period = latest["period"]
        time.sleep(10)

# ------------------------- Flask API -------------------------

@app.route("/")
def home():
    return "✅ Wingo Predictor Bot is Running"

@app.route("/summary")
def summary():
    send_accuracy_report()
    return "📊 Summary sent to Telegram!"

@app.route("/pattern")
def pattern():
    send_pattern_analysis()
    return "📈 Pattern sent to Telegram!"

@app.route("/accuracy")
def accuracy():
    send_accuracy_report()
    return "✅ Accuracy report sent."

# ------------------------- Run App -------------------------

if __name__ == "__main__":
    threading.Thread(target=background_predict_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=10000)
