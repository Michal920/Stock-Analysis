import yfinance as yf
from flask import Flask, render_template, request, jsonify, make_response
import datetime as dt
import pandas_datareader as web
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import pdfkit
from jinja2 import Template
import base64
from io import BytesIO

app = Flask(__name__)

def get_tickers_and_names_from_db():
    conn = sqlite3.connect('tickers.db')
    c = conn.cursor()
    c.execute("SELECT symbol, name FROM tickers")
    data = c.fetchall()
    conn.close()
    return data    

@app.route('/')
def index():
    period = request.args.get('period', '1y')  
    sp500 = yf.Ticker("^GSPC")
    sp500_data = sp500.history(period=period)
    fig_sp500 = go.Figure()
    fig_sp500.add_trace(go.Scatter(x=sp500_data.index, y=sp500_data['Close'], mode='lines', name='Close Price'))
    fig_sp500.update_layout(title='<b>S&P 500 Index</b>', xaxis_title='Date', yaxis_title='Price (USD)', title_x=0.5, title_font=dict(size=18, family="Arial", color="black"))
    plot_url_sp500 = fig_sp500.to_html(full_html=False)

    return render_template('index.html', plot_url_sp500=plot_url_sp500)

@app.route('/home')
def home():
    return index()

@app.route('/stock_info')
def stock_info():
    tickers_and_names = get_tickers_and_names_from_db()
    return render_template('stock_info.html', tickers_and_names=tickers_and_names)

@app.route('/get_stock_info', methods=['POST'])
def get_stock_info():
    data = request.json
    stock_symbol = data['stock_symbol']
    
    try:
        ticker = yf.Ticker(stock_symbol)
        info = ticker.info
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        dividends = ticker.dividends

        financials_dict = financials.to_dict()
        financials_dict = {str(key): {str(inner_key): inner_value.date().strftime('%Y-%m-%d') if isinstance(inner_value, pd.Timestamp) else inner_value if not pd.isna(inner_value) else None for inner_key, inner_value in value.items()} for key, value in financials_dict.items()}

        balance_sheet_dict = balance_sheet.to_dict()
        balance_sheet_dict = {str(key): {str(inner_key): inner_value.date().strftime('%Y-%m-%d') if isinstance(inner_value, pd.Timestamp) else inner_value if not pd.isna(inner_value) else None for inner_key, inner_value in value.items()} for key, value in balance_sheet_dict.items()}

        dividends_dict = dividends.to_dict()
        dividends_dict = {str(date): dividend for date, dividend in dividends_dict.items()}

        info_list = [(key, info[key]) for key in info]
        
        return jsonify({'info': info_list, 'financials': financials_dict, 'balance_sheet': balance_sheet_dict, 'dividends': dividends_dict})
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/data')
def data():
    return render_template('data.html')    

def get_combined_data():
    conn_tickers = sqlite3.connect('tickers.db')
    c_tickers = conn_tickers.cursor()
    c_tickers.execute("SELECT symbol, name FROM tickers")
    tickers_data = c_tickers.fetchall()
    conn_tickers.close()

    conn_stock_info = sqlite3.connect('stock_info.db')
    c_stock_info = conn_stock_info.cursor()
    c_stock_info.execute("SELECT symbol, gics_sector, sub_industry, date FROM stock_info")
    stock_info_data = c_stock_info.fetchall()
    conn_stock_info.close()

    conn_metrics = sqlite3.connect('metrics.db')
    c_metrics = conn_metrics.cursor()
    c_metrics.execute("SELECT symbol, avg_return, volatility, sharpe_ratio FROM metrics")
    metrics_data = c_metrics.fetchall()
    conn_metrics.close()

    tickers_df = pd.DataFrame(tickers_data, columns=['symbol', 'name'])
    stock_info_df = pd.DataFrame(stock_info_data, columns=['symbol', 'gics_sector', 'sub_industry', 'date'])
    metrics_df = pd.DataFrame(metrics_data, columns=['symbol', 'avg_return', 'volatility', 'sharpe_ratio'])

    combined_data = tickers_df.merge(stock_info_df, on='symbol', how='left')
    combined_data = combined_data.merge(metrics_df, on='symbol', how='left')

    return combined_data

@app.route('/get_combined_data', methods=['GET'])
def get_combined_data_endpoint():
    try:
        data = get_combined_data()
        return data.to_json(orient='records')
    except Exception as e:
        return jsonify({'error': str(e)})    
    
@app.route('/portfolio_simulator')
def portfolio_simulator():
    tickers_and_names = get_tickers_and_names_from_db()
    return render_template('portfolio_simulator.html', tickers_and_names=tickers_and_names)

def fig_to_base64(fig):
    img_bytes = fig.to_image(format="png")
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    return base64_str

@app.route('/generate_report', methods=['POST'])
def generate_report():
    data = request.json
    selected_stocks = data['selected_stocks']
    start_date = data['start_date']
    end_date = data['end_date']
    quantities = data['quantities']

    print(f"generate_report: selected_stocks: {selected_stocks}, start_date: {start_date}, end_date: {end_date}, quantities: {quantities}")  # Debugowanie

    stock_prices = {}
    for stock_symbol in selected_stocks:
        try:
            data = yf.download(stock_symbol, start=start_date, end=end_date, auto_adjust=True)
            print(f"Dane dla {stock_symbol}: {data}")  # Debugowanie
            if not data.empty:
                if 'Adj Close' in data.columns:
                    stock_prices[stock_symbol] = data['Adj Close']
                elif 'Close' in data.columns:
                    stock_prices[stock_symbol] = data['Close']
                    print(f"Użyto kolumny 'Close' dla {stock_symbol}, brak 'Adj Close'")
                else:
                    stock_prices[stock_symbol] = None
                    print(f"Brak kolumn 'Adj Close' lub 'Close' dla {stock_symbol}")
                print(f"Type of stock_prices[{stock_symbol}]: {type(stock_prices[stock_symbol])}")  # Debugowanie typu
            else:
                stock_prices[stock_symbol] = None
                print(f"Pusty DataFrame dla {stock_symbol}")
        except Exception as e:
            print(f"An error occurred while fetching data for {stock_symbol}: {str(e)}")
            stock_prices[stock_symbol] = None

    weighted_stock_data = {}
    for i, stock_symbol in enumerate(selected_stocks):
        if stock_prices[stock_symbol] is not None and not stock_prices[stock_symbol].empty:
            try:
                quantity = float(quantities[i])
                # Upewnij się, że wynik jest Series
                if isinstance(stock_prices[stock_symbol], pd.Series):
                    weighted_stock_data[stock_symbol] = stock_prices[stock_symbol] * quantity
                else:
                    # Jeśli stock_prices[stock_symbol] jest DataFrame, weź pierwszą kolumnę jako Series
                    weighted_stock_data[stock_symbol] = stock_prices[stock_symbol].iloc[:, 0] * quantity
                print(f"weighted_stock_data[{stock_symbol}]: {weighted_stock_data[stock_symbol]}")  # Debugowanie
                print(f"Type of weighted_stock_data[{stock_symbol}]: {type(weighted_stock_data[stock_symbol])}")  # Debugowanie typu
            except (ValueError, TypeError) as e:
                print(f"Błąd przy mnożeniu dla {stock_symbol}: {str(e)}")
                weighted_stock_data[stock_symbol] = None

    # Sprawdzenie, czy mamy jakiekolwiek dane
    if not weighted_stock_data:
        return jsonify({'error': 'Brak prawidłowych danych dla wybranych akcji'})

    # Debugowanie zawartości weighted_stock_data
    print(f"weighted_stock_data keys: {list(weighted_stock_data.keys())}")
    for key, value in weighted_stock_data.items():
        print(f"weighted_stock_data[{key}] type: {type(value)}, is_series: {isinstance(value, pd.Series)}, is_dataframe: {isinstance(value, pd.DataFrame)}")

    # Tworzenie DataFrame
    try:
        valid_data = []
        valid_keys = []
        for key, value in weighted_stock_data.items():
            if value is not None:
                if isinstance(value, pd.Series):
                    valid_data.append(value)
                    valid_keys.append(key)
                elif isinstance(value, pd.DataFrame) and not value.empty:
                    valid_data.append(value.iloc[:, 0])
                    valid_keys.append(key)
        print(f"valid_data: {len(valid_data)} elements")  # Debugowanie
        if not valid_data:
            return jsonify({'error': 'Brak prawidłowych danych do utworzenia DataFrame (pusta lista valid_data)'})
        portfolio_value = pd.concat(valid_data, axis=1, keys=valid_keys)
        portfolio_value.columns = valid_keys
        print(f"portfolio_value: {portfolio_value}")  # Debugowanie
    except Exception as e:
        print(f"Błąd przy tworzeniu DataFrame: {str(e)}")
        return jsonify({'error': f'Błąd podczas tworzenia DataFrame: {str(e)}'})

    # Sprawdzenie, czy DataFrame jest pusty lub zawiera tylko NaN
    if portfolio_value.empty or portfolio_value.isnull().all().all():
        return jsonify({'error': 'Brak prawidłowych danych dla wybranych akcji'})

    # Sum across columns to get the total portfolio value
    portfolio_value_sum = portfolio_value.sum(axis=1)
    print(f"portfolio_value_sum: {portfolio_value_sum}")  # Debugowanie

    portfolio_value_initial = portfolio_value_sum.iloc[0]
    print(f"portfolio_value_initial: {portfolio_value_initial}, type: {type(portfolio_value_initial)}")  # Debugowanie
    if portfolio_value_initial != 0:
        portfolio_return = (portfolio_value_sum.iloc[-1] - portfolio_value_initial) / portfolio_value_initial * 100
    else:
        portfolio_return = 0
    portfolio_return = f"{portfolio_return:.2f}%"
    print(f"portfolio_return: {portfolio_return}")  # Debugowanie

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_value_sum.index, y=portfolio_value_sum, mode='lines', name='Portfolio Value'))
    fig.update_layout(title='Portfolio Performance', xaxis_title='Date', yaxis_title='Portfolio Value (USD)')
    performance_chart = fig_to_base64(fig)

    initial_values, final_values, percentage_changes = [], [], []
    for i, stock_symbol in enumerate(selected_stocks):
        if stock_prices[stock_symbol] is not None and not stock_prices[stock_symbol].empty:
            # Użyj skalarnej wartości z DataFrame lub Series
            if isinstance(stock_prices[stock_symbol], pd.Series):
                initial_value = stock_prices[stock_symbol].iloc[0] * float(quantities[i])
                final_value = stock_prices[stock_symbol].iloc[-1] * float(quantities[i])
            else:
                initial_value = stock_prices[stock_symbol].iloc[0, 0] * float(quantities[i])
                final_value = stock_prices[stock_symbol].iloc[-1, 0] * float(quantities[i])
        else:
            initial_value = 0
            final_value = 0
        initial_values.append(initial_value)
        final_values.append(final_value)
        print(f"initial_value for {stock_symbol}: {initial_value}, type: {type(initial_value)}")  # Debugowanie
        print(f"final_value for {stock_symbol}: {final_value}, type: {type(final_value)}")  # Debugowanie
        if initial_value != 0:
            percentage_change = ((final_value - initial_value) / initial_value) * 100
        else:
            percentage_change = 0
        sign = '+' if percentage_change > 0 else ''
        percentage_changes.append(f"{sign}{percentage_change:.2f}%")

    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(x=selected_stocks, y=initial_values, name='Initial Value'))
    bar_fig.add_trace(go.Bar(x=selected_stocks, y=final_values, name='Final Value', text=percentage_changes, textposition='auto'))
    bar_fig.update_layout(title='Initial and Final Values with Percentage Change', xaxis_title='Stocks', yaxis_title='Value (USD)')
    bar_chart = fig_to_base64(bar_fig)

    pie_fig_initial = go.Figure(data=[go.Pie(labels=selected_stocks, values=initial_values)])
    pie_fig_initial.update_layout(title='Initial Portfolio Distribution')
    pie_chart_initial = fig_to_base64(pie_fig_initial)

    pie_fig_final = go.Figure(data=[go.Pie(labels=selected_stocks, values=final_values)])
    pie_fig_final.update_layout(title='Final Portfolio Distribution')
    pie_chart_final = fig_to_base64(pie_fig_final)

    total_investment = sum(initial_values)

    rendered = render_template('report_template.html',
                              selected_stocks=selected_stocks,
                              quantities=quantities,
                              initial_values=initial_values,
                              final_values=final_values,
                              start_date=start_date,
                              end_date=end_date,
                              total_investment=total_investment,
                              portfolio_return=portfolio_return,
                              performance_chart=performance_chart,
                              bar_chart=bar_chart,
                              pie_chart_initial=pie_chart_initial,
                              pie_chart_final=pie_chart_final,
                              zip=zip)

    pdf = pdfkit.from_string(rendered, False)
    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'inline; filename=report.pdf'

    return response

@app.route('/get_performance_data', methods=['POST'])
def get_performance_data():
    data = request.json
    selected_stocks = data['selected_stocks']
    start_date = data['start_date']
    end_date = data['end_date']
    share_values = data['share_values']
    quantities = data['quantities']

    print(f"start_date: {start_date}, end_date: {end_date}, selected_stocks: {selected_stocks}, quantities: {quantities}")  # Debugowanie
    print("Używam wersji z pd.concat")  # Potwierdzenie wersji kodu

    if len(selected_stocks) != len(share_values) or len(selected_stocks) != len(quantities):
        return jsonify({'error': 'Liczba wybranych akcji nie zgadza się z liczbą udziałów lub ilości'})

    # Walidacja dat
    try:
        start_date_dt = dt.datetime.strptime(start_date, "%Y-%m-%d")
        end_date_dt = dt.datetime.strptime(end_date, "%Y-%m-%d")
        if start_date_dt >= end_date_dt:
            return jsonify({'error': 'Data początkowa musi być wcześniejsza niż data końcowa'})
        # Ogranicz end_date do bieżącej daty
        current_date = dt.datetime.now()
        if end_date_dt > current_date:
            end_date_dt = current_date
            end_date = current_date.strftime("%Y-%m-%d")
            print(f"end_date ograniczono do bieżącej daty: {end_date}")
    except ValueError:
        return jsonify({'error': 'Nieprawidłowy format daty'})

    stock_prices = {}
    for i, stock_symbol in enumerate(selected_stocks):
        try:
            stock_data = yf.download(stock_symbol, start=start_date, end=end_date, auto_adjust=True)
            print(f"Dane dla {stock_symbol}: {stock_data}")  # Debugowanie
            if not stock_data.empty:
                if 'Adj Close' in stock_data.columns:
                    stock_prices[stock_symbol] = stock_data['Adj Close']
                elif 'Close' in stock_data.columns:
                    stock_prices[stock_symbol] = stock_data['Close']
                    print(f"Użyto kolumny 'Close' dla {stock_symbol}, brak 'Adj Close'")
                else:
                    stock_prices[stock_symbol] = None
                    print(f"Brak kolumn 'Adj Close' lub 'Close' dla {stock_symbol}")
                print(f"Type of stock_prices[{stock_symbol}]: {type(stock_prices[stock_symbol])}")  # Debugowanie typu
            else:
                stock_prices[stock_symbol] = None
                print(f"Pusty DataFrame dla {stock_symbol}")
        except Exception as e:
            print(f"Błąd podczas pobierania danych dla {stock_symbol}: {str(e)}")
            stock_prices[stock_symbol] = None

    # Konstrukcja weighted_stock_data
    weighted_stock_data = {}
    for i, stock_symbol in enumerate(selected_stocks):
        if stock_prices[stock_symbol] is not None and not stock_prices[stock_symbol].empty:
            try:
                quantity = float(quantities[i])
                # Upewnij się, że wynik jest Series
                if isinstance(stock_prices[stock_symbol], pd.Series):
                    weighted_stock_data[stock_symbol] = stock_prices[stock_symbol] * quantity
                else:
                    # Jeśli stock_prices[stock_symbol] jest DataFrame, weź pierwszą kolumnę jako Series
                    weighted_stock_data[stock_symbol] = stock_prices[stock_symbol].iloc[:, 0] * quantity
                print(f"weighted_stock_data[{stock_symbol}]: {weighted_stock_data[stock_symbol]}")  # Debugowanie
                print(f"Type of weighted_stock_data[{stock_symbol}]: {type(weighted_stock_data[stock_symbol])}")  # Debugowanie typu
            except (ValueError, TypeError) as e:
                print(f"Błąd przy mnożeniu dla {stock_symbol}: {str(e)}")
                weighted_stock_data[stock_symbol] = None

    # Sprawdzenie, czy mamy jakiekolwiek dane
    if not weighted_stock_data:
        return jsonify({'error': 'Brak prawidłowych danych dla wybranych akcji'})

    # Debugowanie zawartości weighted_stock_data
    print(f"weighted_stock_data keys: {list(weighted_stock_data.keys())}")
    for key, value in weighted_stock_data.items():
        print(f"weighted_stock_data[{key}] type: {type(value)}, is_series: {isinstance(value, pd.Series)}, is_dataframe: {isinstance(value, pd.DataFrame)}")

    # Tworzenie DataFrame
    try:
        valid_data = []
        valid_keys = []
        for key, value in weighted_stock_data.items():
            if value is not None:
                if isinstance(value, pd.Series):
                    valid_data.append(value)
                    valid_keys.append(key)
                elif isinstance(value, pd.DataFrame) and not value.empty:
                    # Jeśli DataFrame, weź pierwszą kolumnę jako Series
                    valid_data.append(value.iloc[:, 0])
                    valid_keys.append(key)
        print(f"valid_data: {len(valid_data)} elements")  # Debugowanie
        if not valid_data:
            return jsonify({'error': 'Brak prawidłowych danych do utworzenia DataFrame (pusta lista valid_data)'})
        portfolio_value = pd.concat(valid_data, axis=1, keys=valid_keys)
        portfolio_value.columns = valid_keys
        print(f"portfolio_value: {portfolio_value}")  # Debugowanie
    except Exception as e:
        print(f"Błąd przy tworzeniu DataFrame: {str(e)}")
        return jsonify({'error': f'Błąd podczas tworzenia DataFrame: {str(e)}'})

    # Sprawdzenie, czy DataFrame jest pusty lub zawiera tylko NaN
    if portfolio_value.empty or portfolio_value.isnull().all().all():
        return jsonify({'error': 'Brak prawidłowych danych dla wybranych akcji'})

    portfolio_value_sum = portfolio_value.sum(axis=1)
    print(f"portfolio_value_sum: {portfolio_value_sum}")  # Debugowanie

    portfolio_value_initial = portfolio_value_sum.iloc[0]
    print(f"portfolio_value_initial: {portfolio_value_initial}, type: {type(portfolio_value_initial)}")  # Debugowanie
    if portfolio_value_initial != 0:
        portfolio_return = (portfolio_value_sum.iloc[-1] - portfolio_value_initial) / portfolio_value_initial * 100
    else:
        portfolio_return = 0

    portfolio_return = f"{portfolio_return:.2f}%"
    print(f"portfolio_return: {portfolio_return}")  # Debugowanie

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_value_sum.index, y=portfolio_value_sum, mode='lines', name='Wartość portfela'))
    fig.update_layout(title='Wyniki portfela', xaxis_title='Data', yaxis_title='Wartość portfela (USD)')
    performance_chart = fig.to_html(full_html=False)

    initial_values = []
    final_values = []
    percentage_changes = []
    for i, stock_symbol in enumerate(selected_stocks):
        if stock_prices[stock_symbol] is not None and not stock_prices[stock_symbol].empty:
            # Użyj skalarnej wartości z DataFrame lub Series
            if isinstance(stock_prices[stock_symbol], pd.Series):
                initial_value = stock_prices[stock_symbol].iloc[0] * float(quantities[i])
                final_value = stock_prices[stock_symbol].iloc[-1] * float(quantities[i])
            else:
                initial_value = stock_prices[stock_symbol].iloc[0, 0] * float(quantities[i])
                final_value = stock_prices[stock_symbol].iloc[-1, 0] * float(quantities[i])
        else:
            initial_value = 0
            final_value = 0
        initial_values.append(initial_value)
        final_values.append(final_value)
        print(f"initial_value for {stock_symbol}: {initial_value}, type: {type(initial_value)}")  # Debugowanie
        print(f"final_value for {stock_symbol}: {final_value}, type: {type(final_value)}")  # Debugowanie
        if initial_value != 0:
            percentage_change = ((final_value - initial_value) / initial_value) * 100
        else:
            percentage_change = 0
        sign = '+' if percentage_change > 0 else ''
        percentage_changes.append(f"{sign}{percentage_change:.2f}%")

    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(x=selected_stocks, y=initial_values, name='Wartość początkowa'))
    bar_fig.add_trace(go.Bar(x=selected_stocks, y=final_values, name='Wartość końcowa', text=percentage_changes, textposition='auto'))
    bar_fig.update_layout(title='Wartości początkowe i końcowe z procentową zmianą', xaxis_title='Akcje', yaxis_title='Wartość (USD)')
    bar_chart = bar_fig.to_html(full_html=False)

    colors = px.colors.qualitative.Plotly
    while len(colors) < len(selected_stocks):
        colors = colors + colors
    colors = colors[:len(selected_stocks)]

    pie_fig_initial = go.Figure(data=[go.Pie(labels=selected_stocks, values=initial_values, name='Początkowy rozkład portfela', marker=dict(colors=colors))])
    pie_fig_initial.update_layout(title='Początkowy rozkład portfela')
    pie_chart_initial = pie_fig_initial.to_html(full_html=False)

    pie_fig_final = go.Figure(data=[go.Pie(labels=selected_stocks, values=final_values, name='Końcowy rozkład portfela', marker=dict(colors=colors))])
    pie_fig_final.update_layout(title='Końcowy rozkład portfela')
    pie_chart_final = pie_fig_final.to_html(full_html=False)

    response = {
        'portfolio_return': portfolio_return,
        'performance_chart': performance_chart,
        'bar_chart': bar_chart,
        'pie_chart_initial': pie_chart_initial,
        'pie_chart_final': pie_chart_final
    }
    print(f"Response JSON: {response}")  # Debugowanie
    return jsonify(response)
@app.route('/get_start_date_prices', methods=['POST'])
def get_start_date_prices():
    data = request.json
    selected_stocks = data['selected_stocks']
    start_date = data['start_date']
    start_date = dt.datetime.strptime(start_date, "%Y-%m-%d")
    
    stock_prices = {}
    for stock_symbol in selected_stocks:
        try:
            stock_data = yf.download(
                tickers=stock_symbol,
                start=start_date,
                end=start_date + dt.timedelta(days=1),
                auto_adjust=True
            )
            print(f"Dane dla {stock_symbol} w get_start_date_prices: {stock_data}")  # Debugowanie
            if not stock_data.empty:
                if 'Adj Close' in stock_data.columns:
                    close_price = stock_data['Adj Close'].iloc[0].item()  # Użycie .item() zamiast float()
                    stock_prices[stock_symbol] = close_price
                elif 'Close' in stock_data.columns:
                    close_price = stock_data['Close'].iloc[0].item()  # Użycie .item() zamiast float()
                    stock_prices[stock_symbol] = close_price
                    print(f"Użyto kolumny 'Close' dla {stock_symbol} w get_start_date_prices")
                else:
                    stock_prices[stock_symbol] = None
                    print(f"Brak kolumn 'Adj Close' lub 'Close' dla {stock_symbol} w get_start_date_prices")
            else:
                stock_prices[stock_symbol] = None
                print(f"Pusty DataFrame dla {stock_symbol} w get_start_date_prices")
        except Exception as e:
            print(f"Błąd podczas pobierania danych dla {stock_symbol} na {start_date}: {str(e)}")
            stock_prices[stock_symbol] = None

    return jsonify(stock_prices)
def get_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    if 'Adj Close' in data:
        return data['Adj Close']
    elif 'Close' in data:
        return data['Close']
    else:
        return pd.DataFrame()

def get_stock_data2(tickers, start_date2, end_date2):
    data = yf.download(tickers, start=start_date2, end=end_date2)
    if 'Adj Close' in data:
        return data['Adj Close']
    elif 'Close' in data:
        return data['Close']
    else:
        return pd.DataFrame()

def calculate_returns(weights, data):
    returns = np.dot(data.pct_change().mean(), weights) * 252
    return returns

def calculate_volatility(weights, data):
    volatility = np.sqrt(np.dot(weights.T, np.dot(data.pct_change().cov() * 252, weights)))
    return volatility

def negative_sharpe_ratio(weights, data, risk_free_rate):
    returns = calculate_returns(weights, data)
    volatility = calculate_volatility(weights, data)
    sharpe_ratio = (returns - risk_free_rate) / volatility
    return -sharpe_ratio

def get_max_units(budget, price):
    max_units = int(budget / price)
    return max_units

def optimize_portfolio(tickers, budget, start_date, end_date, risk_aversion):
    data = get_stock_data(tickers, start_date, end_date)
    if data.empty:
        return []

    num_assets = len(tickers)
    initial_weights = np.array([1/num_assets] * num_assets)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))

    optimal_result = minimize(negative_sharpe_ratio, initial_weights, args=(data, risk_aversion), method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimal_weights = optimal_result.x

    mean_returns = data.pct_change().mean() * 252
    volatilities = data.pct_change().std() * np.sqrt(252)
    cov_matrix = data.pct_change().cov() * 252
    corr_matrix = data.pct_change().corr()

    allocation = []
    total_cost = 0
    for ticker, weight in zip(tickers, optimal_weights):
        price = data[ticker][-1]
        allocated_units = weight * budget / price
        total_cost += allocated_units * price
        allocation.append((ticker, allocated_units))

    if total_cost > budget * 1.1:
        scaling_factor = (budget * 1.1) / total_cost
        allocation = [(ticker, units * scaling_factor) for ticker, units in allocation]

    rounded_allocation = [(ticker, round(units)) for ticker, units in allocation]    

    return {
        'allocation': rounded_allocation,
        'mean_returns': mean_returns.to_dict(),
        'volatilities': volatilities.to_dict(),
        'cov_matrix': cov_matrix.to_dict(),
        'corr_matrix': corr_matrix.to_dict(),
        'tickers': tickers,
    }

@app.route('/optimize_portfolio', methods=['POST'])
def optimize_portfolio_endpoint():
    data = request.json

    tickers = data['selected_stocks']
    budget = data['budget']
    end_date = data['start_date']
    start_date = (dt.datetime.strptime(end_date, "%Y-%m-%d") - dt.timedelta(days=3*365)).strftime("%Y-%m-%d")
    start_date2 = data['start_date']
    end_date2 = data['end_date']
    risk_aversion = data['risk_aversion']

    allocation_result = optimize_portfolio(tickers, budget, start_date, end_date, risk_aversion)

    if not allocation_result:
        print("Error")
    else:
        print("Optimal allocation:")
        for ticker, units in allocation_result['allocation']:
            print(f"{ticker}: {units:.2f} units")

    allocation = allocation_result['allocation']
    mean_returns = allocation_result['mean_returns']
    volatilities = allocation_result['volatilities']
    cov_matrix = allocation_result['cov_matrix']
    corr_matrix = allocation_result['corr_matrix']   
    tickers = allocation_result['tickers']  

    data2 = get_stock_data2(tickers, start_date2, end_date2)
    
    portfolio_values = []
    dates = data2.index
    for date in dates:
        daily_value = 0
        for ticker, units in allocation:
            daily_value += data2.loc[date, ticker] * units
        portfolio_values.append(daily_value)

    portfolio_value_initial = portfolio_values[0]
    portfolio_value_final = portfolio_values[-1]

    if portfolio_value_initial != 0:
        portfolio_return = (portfolio_value_final - portfolio_value_initial) / portfolio_value_initial * 100
    else:
        portfolio_return = 0    

    portfolio_return = f"{portfolio_return:.2f}%"    

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=portfolio_values, mode='lines', name='Portfolio Value'))
    fig.update_layout(title='Portfolio Performance After Optimization', xaxis_title='Date', yaxis_title='Portfolio Value (USD)')
    performance_chart = fig.to_html(full_html=False)  

    covariance_df = pd.DataFrame(cov_matrix)
    correlation_df = pd.DataFrame(corr_matrix)

    covariance_dict = covariance_df.to_dict()
    correlation_dict = correlation_df.to_dict()

    return jsonify({
        'allocation': allocation,
        'performance_chart': performance_chart,
        'mean_returns': mean_returns,
        'volatilities': volatilities,
        'cov_matrix': covariance_dict,
        'corr_matrix': correlation_dict,
        'portfolio_return': portfolio_return,
        'tickers': tickers
    })

if __name__ == '__main__':
    app.run()