import pandas as pd
import sqlite3

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

tables = pd.read_html(url)

sp500_table = tables[0]

stock_info = sp500_table[['Symbol', 'GICS Sector', 'GICS Sub-Industry', 'Date added']]

db_file_path = "stock_info.db"

conn = sqlite3.connect(db_file_path)
c = conn.cursor()

c.execute('DROP TABLE IF EXISTS stock_info')

c.execute('''CREATE TABLE stock_info
             (id INTEGER PRIMARY KEY, 
              symbol TEXT, 
              gics_sector TEXT, 
              sub_industry TEXT, 
              date DATE)''')

for index, row in stock_info.iterrows():
    symbol = row['Symbol']
    gics_sector = row['GICS Sector']
    sub_industry = row['GICS Sub-Industry']
    date = row['Date added']
    c.execute("INSERT INTO stock_info (symbol, gics_sector, sub_industry, date) VALUES (?, ?, ?, ?)", 
              (symbol, gics_sector, sub_industry, date))

conn.commit()

conn.close()

print("Baza danych została zaktualizowana.")
