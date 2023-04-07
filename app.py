from flask import Flask, render_template, request, jsonify, redirect, url_for
import firebase_admin
from firebase_admin import credentials, auth
from stock import predict_stock, get_stock_history, get_stock_history_all
import os


app = Flask(__name__)

@app.route("/predict", methods=["GET", "POST"])
def predict_stock_route():
    if request.method == "POST":
        ticker = request.form["ticker"].upper()
        max_depth = int(request.form.get("max_depth", 10))
        min_samples_split = int(request.form.get("min_samples_split", 5))
        random_state = int(request.form.get("random_state", 60))
        if ticker:
            try:
                result = predict_stock(ticker, max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)
            except Exception as e:
                result = f"Error: {str(e)}"

            history = get_stock_history(ticker)
            return render_template("index.html", ticker=ticker, result=result, history=history)

    return render_template("index.html")

@app.route("/history")
def history():
    history = get_stock_history_all()
    return render_template("history.html", history=history)


@app.route("/")
def home():
    return render_template("index.html")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, port=port, host='0.0.0.0')