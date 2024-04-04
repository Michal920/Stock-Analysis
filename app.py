from flask import Flask, render_template, request
import datetime as dt
import pandas_datareader as web
import yfinance as yf
import plotly.graph_objs as go
import os
import sqlite3

app = Flask(__name__)

# Tworzenie katalogu static, jeśli nie istnieje
static_folder = os.path.join(app.root_path, 'static')
if not os.path.isdir(static_folder):
    os.mkdir(static_folder)

# Funkcja do pobierania tickers i nazw firm z bazy danych
def get_tickers_and_names_from_db():
    conn = sqlite3.connect('tickers.db')
    c = conn.cursor()
    c.execute("SELECT symbol, name FROM tickers")
    data = c.fetchall()
    conn.close()
    return data    

@app.route('/')
def index():
    tickers_and_names = get_tickers_and_names_from_db()
    return render_template('index.html', tickers_and_names=tickers_and_names)

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        start = request.form['start']
        end = request.form['end']
        ticker = request.form['ticker']
        investment_amt = float(request.form['investment'])

        start_date = dt.datetime.strptime(start, "%Y-%m-%d")
        end_date = dt.datetime.strptime(end, "%Y-%m-%d")

        ticker = yf.Ticker(ticker)
        df = ticker.history(start=start, end=end)
        df = df[['Open', 'High', 'Low', 'Close']]

        open_price_day1 = df.iloc[0]['Open']
        close_price_last_day = df.iloc[-1]['Close']

        percentage_change = ((close_price_last_day - open_price_day1) / open_price_day1) * 100
        profit_loss = investment_amt +  ((percentage_change*investment_amt)/100)

        # Tworzenie interaktywnego wykresu świecowego przy użyciu Plotly
        fig = go.Figure(data=[go.Candlestick(x=df.index,
                                             open=df['Open'],
                                             high=df['High'],
                                             low=df['Low'],
                                             close=df['Close'])])

        plot_path = 'static/plot.html'
        fig.write_html(plot_path)

        return render_template('result.html', 
                               open_price=open_price_day1,
                               close_price=close_price_last_day,
                               percentage_change=percentage_change,
                               profit_loss=profit_loss,
                               plot_url=plot_path)

if __name__ == '__main__':
    app.run(debug=True)
