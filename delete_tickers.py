import sqlite3

def delete_tickers_table():
    try:
        conn = sqlite3.connect('metrics.db')
        c = conn.cursor()

        c.execute("DROP TABLE IF EXISTS tickers")
        print("Tabela 'tickers' została usunięta.")

        c.execute("ALTER TABLE stock_metrics RENAME TO metrics")
        print("Tabela 'stock_metrics' została zmieniona na 'metrics'.")

        conn.commit()
        conn.close()
    except Exception as e:
        print("Wystąpił błąd podczas usuwania tabeli 'tickers':", e)

if __name__ == "__main__":
    delete_tickers_table()
