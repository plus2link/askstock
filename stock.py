import tkinter as tk
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, firestore
import pytz
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import yfinance as yf
import json


# Replace 'path/to/your/firebase-credentials.json' with the path to your Firebase Admin SDK JSON file
cred = credentials.Certificate("path/to/stock-e0e77-firebase.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

API_KEY = '13DP0TAQWPY8LY5H'


def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)

    # If data is empty, return an empty DataFrame
    if data.empty:
        return pd.DataFrame(columns=["Open", "Close", "High", "Low", "Volume"])

    data = data.reset_index()
    data = data[["Date", "Open", "Close", "High", "Low", "Volume"]]
    data.columns = ["Date", "Open", "Close", "High", "Low", "Volume"]

    return data


def get_stock_name(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    if info and 'shortName' in info:
        return info['shortName']
    return None



def preprocess_data(stock_data):
    stock_data["Open-Close"] = stock_data["Open"] - stock_data["Close"]
    stock_data["High-Low"] = stock_data["High"] - stock_data["Low"]
    stock_data["Target"] = np.where(stock_data["Close"].shift(-1) > stock_data["Close"], 1, -1)

    return stock_data.dropna()


def split_data(preprocessed_data):
    X = preprocessed_data[["Open-Close", "High-Low"]]
    y = preprocessed_data["Target"]

    return train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)


def train_model(X_train, y_train, max_depth, min_samples_split, random_state):
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

def save_to_firebase(date, ticker, stock_name, prediction, accuracy, result, max_depth, min_samples_split, random_state):
    # Convert the date object to a datetime object with timezone information
    date = datetime.combine(date, datetime.min.time()).replace(tzinfo=pytz.UTC)

    # Create a unique document ID based on the date and ticker
    doc_id = f"{date.strftime('%Y%m%d')}-{ticker}"

    doc_ref = db.collection("stock_history").document(doc_id)
    doc_ref.set({
        "Date": date,
        "Ticker": ticker,
        "Stock Name": stock_name,
        "Prediction": prediction,
        "Accuracy": accuracy,
        "Result": result,
        "Depth" : max_depth, 
        "Split" : min_samples_split, 
        "State" : random_state
    })


def predict_stock(ticker, max_depth, min_samples_split, random_state):
    now = datetime.now()

    # Convert to US/Eastern timezone
    tz = pytz.timezone('US/Eastern')
    end_date = now.astimezone(tz)
    start_date = end_date - timedelta(days=365 * 2)

    stock_data = get_stock_data(ticker, start_date, end_date)
    stock_name = get_stock_name(ticker)


    if stock_data.empty:
        return "Error: Unable to fetch stock data. Please check the ticker and try again."

    preprocessed_data = preprocess_data(stock_data)
    X_train, X_test, y_train, y_test = split_data(preprocessed_data)
    model = train_model(X_train, y_train, max_depth, min_samples_split, random_state)
    accuracy = evaluate_model(model, X_test, y_test)

    accuracy_str = f"Accuracy: {accuracy * 100:.2f}%\n"

    last_row = preprocessed_data.iloc[-1][['Open-Close', 'High-Low']].values.reshape(1, -1)
    prediction = model.predict(last_row)

    success = None
    if prediction == 1:
        prediction_str = "The model predicts the stock will go up tomorrow.\n"
        result = "up"
    else:
        prediction_str = "The model predicts the stock will go down tomorrow.\n"
        result = "down"

    # Save the current prediction to Firebase
    today_date = end_date
    tomorrow_date = today_date + timedelta(days=1)
    save_to_firebase(tomorrow_date, ticker, stock_name, result, accuracy, success, max_depth, min_samples_split, random_state)

    return prediction_str, result, accuracy



def get_stock_history(ticker):
    stock_history = []
    docs = db.collection("stock_history").where("Ticker", "==", ticker).stream()
    for doc in docs:
        data = doc.to_dict()
        actual_data = get_stock_data(ticker, data["Date"], data["Date"] + pd.Timedelta(hours=23, minutes=59, seconds=59))
        if not actual_data.empty:
            actual_data_1 = actual_data.iloc[0]['Close']
        else:
            actual_data_1 = 0

        actual_data_y = get_stock_data(ticker, data["Date"]-pd.Timedelta(hours=23, minutes=59, seconds=59), data["Date"]-pd.Timedelta(hours=0, minutes=1, seconds=1))
        if not actual_data_y.empty:
            actual_data_2 = actual_data_y.iloc[0]['Close']
        else:
            actual_data_2 = 0
            
        data["Date"] = data["Date"].date()
        if len(actual_data) == 0:
            data["Result"] = "N/A"
            data["Close"] = "N/A"
        else:
            actual_close = actual_data_1 - actual_data_2
            data["Close"] = actual_data_1
            if actual_close > 0:
                actual_success = "up"
            elif actual_close == 0:
                actual_success = "="
            else:
                actual_success = "down"
            print(actual_close)

            data["Result"] = actual_success
        stock_history.append(data)
    return stock_history


def get_stock_history_all():
    stock_history = []
    docs = db.collection("stock_history").stream()
    for doc in docs:
        data = doc.to_dict()
        data["Date"] = data["Date"].date()
        stock_history.append(data)
    return stock_history



def main():
    ticker = entry_ticker.get().upper()
    if ticker:
        result_text.delete(1.0, tk.END)  # Clear the text widget
        try:
            result = predict_stock(ticker)
        except Exception as e:
            result = f"Error: {str(e)}"

        result_text.insert(tk.END, result)

def on_enter_key(event):
    main()

app = tk.Tk()
app.title("Stock Predictor")

frame = tk.Frame(app)
frame.pack(padx=10, pady=10)

label_ticker = tk.Label(frame, text="Enter the stock ticker:")
label_ticker.pack()


entry_ticker = tk.Entry(frame, font=("Arial", 14))
entry_ticker.pack(pady=(0, 20))
entry_ticker.bind("<Return>", on_enter_key)

result_text = tk.Text(frame, wrap="word", width=80, height=15)
result_text.pack()