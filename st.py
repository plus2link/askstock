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


# Replace 'path/to/your/firebase-credentials.json' with the path to your Firebase Admin SDK JSON file
cred = credentials.Certificate("path/to/stock-e0e77-firebase.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

API_KEY = '13DP0TAQWPY8LY5H'


def get_stock_data(ticker, start_date, end_date):
    api_key = "13DP0TAQWPY8LY5H"
    base_url = "https://www.alphavantage.co/query?"
    function = "TIME_SERIES_DAILY_ADJUSTED"

    url = f"{base_url}function={function}&symbol={ticker}&apikey={api_key}&outputsize=full&datatype=csv"
    stock_data = pd.read_csv(url, index_col=0)

    # Parse dates and set the index
    stock_data.index = pd.to_datetime(stock_data.index)

    # Filter the data based on the date range
    stock_data = stock_data.loc[end_date:start_date]

    return stock_data


def get_stock_name(ticker):
    url = f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={ticker}&apikey={API_KEY}'
    response = requests.get(url)
    data = response.json()
    if data and 'bestMatches' in data and len(data['bestMatches']) > 0:
        return data['bestMatches'][0]['2. name']
    return None


def preprocess_data(stock_data):
    stock_data["Open-Close"] = stock_data["open"] - stock_data["close"]
    stock_data["High-Low"] = stock_data["high"] - stock_data["low"]
    stock_data["Target"] = np.where(stock_data["close"].shift(-1) > stock_data["close"], 1, -1)

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

def save_to_firebase(date, ticker, stock_name, prediction, accuracy, result):
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
        "Result": result
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

    # Retrieve the last available prediction for the stock ticker
    last_prediction_doc = db.collection("stock_history").where("Ticker", "==", ticker).order_by("Date", direction=firestore.Query.DESCENDING).limit(1).get()

    success = None
    if last_prediction_doc:
        last_prediction = last_prediction_doc[0].to_dict()
        last_prediction_result = last_prediction["Prediction"]
        last_prediction_date = last_prediction["Date"].date()

        # Check the success of the previous day's prediction
        if last_prediction_date == end_date.date():
            success = "N/A"
        elif last_prediction_result == "up" and prediction == 1:
            success = True
        elif last_prediction_result == "down" and prediction == -1:
            success = True
        else:
            success = False



    if prediction == 1:
        prediction_str = "The model predicts the stock will go up tomorrow.\n"
        result = "up"
    else:
        prediction_str = "The model predicts the stock will go down tomorrow.\n"
        result = "down"

    # Save the current prediction to Firebase
    today_date = datetime.now().date()
    save_to_firebase(today_date, ticker, stock_name, result, accuracy, success)

    return prediction_str



def get_stock_history(ticker):
    stock_history = []
    docs = db.collection("stock_history").where("Ticker", "==", ticker).stream()
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