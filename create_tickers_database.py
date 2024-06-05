import pandas as pd
import sqlite3
import requests

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

tables = pd.read_html(url)

sp500_table = tables[0]

tickers_and_names = sp500_table[['Symbol', 'Security']]

tickers_and_names.columns = ['Symbol', 'Name']

db_file_path = "tickers.db"

conn = sqlite3.connect(db_file_path)
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS tickers
             (id INTEGER PRIMARY KEY, symbol TEXT, name TEXT)''')

for index, row in tickers_and_names.iterrows():
    symbol = row['Symbol']
    name = row['Name']
    c.execute("INSERT INTO tickers (symbol, name) VALUES (?, ?)", (symbol, name))

conn.commit()

conn.close()

print("Baza danych została zaktualizowana.")
