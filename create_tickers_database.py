import pandas as pd
import sqlite3
import requests

# URL strony Wikipedia z listą firm S&P 500
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Wczytanie tabeli z danej strony internetowej
tables = pd.read_html(url)

# Pierwsza tabela zawiera listę firm S&P 500
sp500_table = tables[0]

# Wybieramy kolumny z tickerami i nazwami firm
tickers_and_names = sp500_table[['Symbol', 'Security']]

# Renaming columns for better understanding
tickers_and_names.columns = ['Symbol', 'Name']

# Określenie ścieżki do pliku baz danych
db_file_path = "tickers.db"

# Tworzenie lub łączenie się z bazą danych SQLite
conn = sqlite3.connect(db_file_path)
c = conn.cursor()

# Tworzenie tabeli tickerów w bazie danych, jeśli nie istnieje
c.execute('''CREATE TABLE IF NOT EXISTS tickers
             (id INTEGER PRIMARY KEY, symbol TEXT, name TEXT)''')

# Wstawianie danych do tabeli
for index, row in tickers_and_names.iterrows():
    symbol = row['Symbol']
    name = row['Name']
    c.execute("INSERT INTO tickers (symbol, name) VALUES (?, ?)", (symbol, name))

# Zapisywanie zmian w bazie danych
conn.commit()

# Zamykanie połączenia z bazą danych
conn.close()

print("Baza danych została zaktualizowana.")
