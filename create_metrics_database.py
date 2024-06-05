import pandas as pd
import numpy as np
import sqlite3
import yfinance as yf
from datetime import datetime, timedelta

def get_tickers_and_names_from_db():
    conn = sqlite3.connect('tickers.db')
    c = conn.cursor()
    c.execute("SELECT symbol, name FROM tickers")
    data = c.fetchall()
    conn.close()
    return data

def get_historical_data(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10*365)
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def calculate_metrics(data):
    returns = data['Adj Close'].pct_change().dropna()
    avg_return = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = avg_return / volatility
    return avg_return, volatility, sharpe_ratio

def create_new_table():
    conn = sqlite3.connect('tickers.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS stock_metrics
                 (id INTEGER PRIMARY KEY, 
                  symbol TEXT, 
                  avg_return REAL, 
                  volatility REAL, 
                  sharpe_ratio REAL)''')
    conn.commit()
    conn.close()

def insert_data_to_db(data):
    conn = sqlite3.connect('tickers.db')
    c = conn.cursor()
    for row in data:
        c.execute("INSERT INTO stock_metrics (symbol, avg_return, volatility, sharpe_ratio) VALUES (?, ?, ?, ?)",
                  (row['symbol'], row['avg_return'], row['volatility'], row['sharpe_ratio']))
    conn.commit()
    conn.close()

def main():
    tickers_and_names = get_tickers_and_names_from_db()
    metrics_data = []
    
    for symbol, name in tickers_and_names:
        try:
            historical_data = get_historical_data(symbol)
            avg_return, volatility, sharpe_ratio = calculate_metrics(historical_data)
            metrics_data.append({
                'symbol': symbol,
                'avg_return': avg_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio
            })
        except Exception as e:
            print(f"Could not process {symbol}: {e}")
    
    create_new_table()
    insert_data_to_db(metrics_data)
    print("Dane zostały zapisane w bazie danych.")

if __name__ == "__main__":
    main()
