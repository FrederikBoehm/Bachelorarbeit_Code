import pandas as pd
import sqlite3

# Labels the reports as positive and negative

def getPriceChanges():
    df_multiline_reports = pd.read_csv('./data/multiline_report_index.csv', sep='\t')

    connection = sqlite3.connect('./data/historic_stock_prices.db')
    db_cursor = connection.cursor()

    output_file = open('./data/multiline_report_index_with_price_changes.csv', 'w')
    output_file.write('CIK\tTicker\tCompany\tFiling_Date\tForm_Type\tChange_Ratio\tChange_Nominal\tFile_Path\n')
    output_file.close()

    cik_column = list(df_multiline_reports['CIK'])
    ticker_column = list(df_multiline_reports['Ticker'])
    company_column = list(df_multiline_reports['Company'])
    date_column = list(df_multiline_reports['Filing_Date'])
    type_column = list(df_multiline_reports['Form_Type'])
    file_column = list(df_multiline_reports['File_Path'])

    for index, date in enumerate(date_column):

        cik = cik_column[index]
        ticker = ticker_column[index]
        company = company_column[index]
        filing_date = date_column[index]
        form_type = type_column[index]
        file_path = file_column[index]

        # Converting date for sql
        date_string = str(date)
        year = date_string[:4]
        month = date_string[4:6]
        day = date_string[6:]

        sql_date = f'{year}-{month}-{day}'

        sql_command = f"""
        SELECT open, close FROM trading_day
        WHERE date >= '{sql_date}'
        AND ticker = '{ticker}'
        ORDER BY date ASC
        LIMIT 3;"""

        db_cursor.execute(sql_command)
        prices = db_cursor.fetchall()

        if len(prices) == 3:
            open_on_disclosure = prices[0][0]
            close_two_days_later = prices[len(prices)-1][1]

            change_ratio = (close_two_days_later - open_on_disclosure) / open_on_disclosure
            change_nominal = 'positive' if change_ratio >= 0 else 'negative'

            print(f'Adding entry with CIK {cik}, Ticker {ticker}, Company {company}, Filing Date {filing_date}, Form Type {form_type}, Change Ratio {change_ratio}, Nominal Change {change_nominal} Output File Path {file_path} .')
            output_row = f'{cik}\t{ticker}\t{company}\t{filing_date}\t{form_type}\t{change_ratio}\t{change_nominal}\t{file_path}\n'
            output_file = open('./data/multiline_report_index_with_price_changes.csv', 'a')
            output_file.write(output_row)
            output_file.close()

        else:
            print(f'No historic prices found for {company} on {sql_date}')

    connection.close()

if __name__ == '__main__':
    getPriceChanges()






    