import pandas as pd
import requests
import json
import time
import os
import numpy as np
import sqlite3

def loadHistoricStockPrices():
    df_multiline_report_index = pd.read_csv('./data/multiline_report_index.csv', sep='\t')
    api_key_file = open('./data/Alphavantage_API_Key.txt', 'r')
    api_key = api_key_file.read()
    api_key_file.close()
    api_endpoint = 'https://www.alphavantage.co/query'

    connection = _initializeDatabase('./data/historic_stock_prices.db')
    db_cursor = connection.cursor()
    already_loaded = _getAlreadyLoadedTickers(db_cursor)

    if not os.path.exists('./data/error_codes.csv'):
        error_codes = open('./data/error_codes.csv', 'w')
        error_codes.write('Ticker|Error_Code|Error_Message\n')
        error_codes.close()

    ticker_symbols = df_multiline_report_index["Ticker"].unique()

    for ticker_symbol in ticker_symbols[390:]:
        if not ticker_symbol in already_loaded:
            print(f"Requesting historic stock prices for {ticker_symbol} at {api_endpoint}.")

            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': ticker_symbol,
                'apikey': api_key,
                'outputsize': 'full',
                'datatype': 'json'
            }

            response = requests.get(api_endpoint, params=params)
            print(f'Sent request to: {response.request.url}')
            if 'Note' in response.json():
                print(f'Note: {response.json()["Note"]}')

            if (response.status_code == 200) and (not 'Error Message' in response.json()) and ('Time Series (Daily)' in response.json()):
                print('Request successful, processing data...')


                cik = df_multiline_report_index.loc[df_multiline_report_index['Ticker'] == ticker_symbol, 'CIK'].iloc[0]
                company_name = df_multiline_report_index.loc[df_multiline_report_index['Ticker'] == ticker_symbol, 'Company'].iloc[0]

                _insertIntoCompany(db_cursor, ticker_symbol, cik, company_name)

                daily_time_series = response.json()["Time Series (Daily)"]

                for date in daily_time_series.keys():
                    trading_day = daily_time_series[f"{date}"]
                    _insertIntoTradingDay(db_cursor, date, ticker_symbol, trading_day["1. open"], trading_day["4. close"])
            
                print('Finished.')

            else:
                print(f'Error during request, Status Code: {response.status_code}')
                if 'Error Message' in response.json():
                    print(f'Error Message: {response.json()["Error Message"]}')
                    error_codes = open('./data/error_codes.csv', 'a')
                    error_message = response.json()["Error Message"]
                    error_codes.write(f'{ticker_symbol}|{response.status_code}|{error_message}\n')
                    error_codes.close()
                else:
                    error_codes = open('./data/error_codes.csv', 'a')
                    error_codes.write(f'{ticker_symbol}|{response.status_code}|\n')
                    error_codes.close()

            connection.commit()
            time.sleep(12.1) # Depending on the API key you use you can change the timeout. I am using the free API key, so I have to stay below five requests per minute

    connection.close()

def _initializeDatabase(path):
    print(f'Initializing database at {path}')
    connection = sqlite3.connect(path)

    cursor = connection.cursor()

    sql_command = "SELECT name FROM sqlite_master WHERE type='table' AND name='company';"
    cursor.execute(sql_command)
    if not cursor.fetchone():
        print("Creating company table.")
        sql_command = """
        CREATE TABLE company ( 
        ticker VARCHAR(20) PRIMARY KEY, 
        cik VARCHAR(10), 
        name VARCHAR(50));"""
        
        cursor.execute(sql_command)

    sql_command = "SELECT name FROM sqlite_master WHERE type='table' AND name='trading_day';"
    cursor.execute(sql_command)
    if not cursor.fetchone():
        print("Creating trading_day table.")
        sql_command = """
        CREATE TABLE trading_day ( 
        date DATE, 
        ticker VARCHAR(20), 
        open DOUBLE(10, 4), 
        close DOUBLE(10, 4),
        PRIMARY KEY(date, ticker)
        FOREIGN KEY(ticker) REFERENCES company(ticker));"""

        cursor.execute(sql_command)

    return connection

def _insertIntoCompany(cursor, ticker, cik, name):
    sql_command = f"""
    INSERT INTO company (ticker, cik, name)
    VALUES ('{ticker}', '{cik}', '{name}');"""
    cursor.execute(sql_command)

def _insertIntoTradingDay(cursor, date, ticker, open_price, close_price):
    sql_command = f"""
    INSERT INTO trading_day (date, ticker, open, close)
    VALUES ('{date}', '{ticker}', '{open_price}', '{close_price}');"""
    cursor.execute(sql_command)


def _getAlreadyLoadedTickers(cursor):
    sql_command = "SELECT ticker FROM company;"
    cursor.execute(sql_command)
    already_loaded = cursor.fetchall()
    already_loaded = list(map(lambda item: item[0], already_loaded)) # Map to first value of tuple
    return already_loaded

def _convert(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError

if __name__ == "__main__":
    loadHistoricStockPrices()

