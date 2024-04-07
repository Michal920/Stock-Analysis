from flask import Flask, render_template, request
import datetime as dt
import pandas_datareader as web
import yfinance as yf
import plotly.graph_objs as go
import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt


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
      # Pobierz dane o wybranych spółkach z ostatniego tygodnia
    symbols = ['MSFT', 'NVDA', 'GOOGL', 'META', 'AMZN', 'AAPL']
    stocks_data = {}
    for symbol in symbols:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1mo")
        stocks_data[symbol] = data

    # Wygeneruj interaktywne wykresy za pomocą Plotly
    plot_urls = {}
    for symbol, data in stocks_data.items():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
        fig.update_layout(title=f'{symbol} Stock Price (Last Week)', xaxis_title='Date', yaxis_title='Price (USD)')
        fig.update_layout(font=dict(size=10))  # Zmniejsz rozmiar czcionki
        fig.update_layout(height=250, width=300)  # Zmniejsz rozmiar wykresu
        plot_urls[symbol] = fig.to_html(full_html=False)

    # Pobierz dane o indeksie S&P 500
    sp500 = yf.Ticker("^GSPC")
    sp500_data = sp500.history(period="1y")
    fig_sp500 = go.Figure()
    fig_sp500.add_trace(go.Scatter(x=sp500_data.index, y=sp500_data['Close'], mode='lines', name='Close Price'))
    fig_sp500.update_layout(title='S&P 500 Index', xaxis_title='Date', yaxis_title='Price (USD)')
    fig_sp500.update_layout(font=dict(size=12))  # Zmniejsz rozmiar czcionki
    fig_sp500.update_layout(height=400, width=600)  # Zmniejsz rozmiar wykresu
    plot_url_sp500 = fig_sp500.to_html(full_html=False)

    # Wyświetl interaktywne wykresy spółek na stronie głównej
    return render_template('index.html', plot_url_sp500=plot_url_sp500, plot_urls=plot_urls)

def home():
    return render_template('index.html')

@app.route('/stock_analysis')
def stock_analysis():
    tickers_and_names = get_tickers_and_names_from_db()
    return render_template('stock_analysis.html', tickers_and_names=tickers_and_names)

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
