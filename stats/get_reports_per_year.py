import pandas as pd
import numpy as np

def getReportsPerYear():

    print('Loading file summaries...')
    df_file_summaries = pd.read_csv('./data/Edgar/LM_10X_Summaries_2018.csv')
    df_historic_composition = pd.read_csv('./data/S_and_P_historical.csv')

    # Filter for reports after 31.12.2003 and only 10-K's and 10-Q's
    df_file_summaries = df_file_summaries[df_file_summaries['FILING_DATE'].astype(str).str.contains(r'20(?:04|05|06|07|08|09|10|11|12|13|14|15|16|17|18)\S{4}')]
    df_file_summaries = df_file_summaries[(df_file_summaries['FORM_TYPE'] == '10-K') | (df_file_summaries['FORM_TYPE'] == '10-Q')]
    df_cik_ticker_mapping = pd.read_csv('./data/cik_ticker_modified.csv', delimiter='|')

    historic_composition_date =  20031231
    init_index =  0

    _countReports(df_file_summaries, df_cik_ticker_mapping, df_historic_composition, init_index, historic_composition_date)

def _countReports(df_file_summaries, df_cik_ticker_mapping, df_historic_composition, init_index, historic_composition_date):
    historic_composition_at_date = df_historic_composition[df_historic_composition['Date'] == historic_composition_date].values

    number_10K = 0
    number_10Q = 0
    companys = set()

    for index, cik in _enumerate(df_file_summaries['CIK'], start=init_index, step=1):
        if cik in historic_composition_at_date:

            form_type = df_file_summaries['FORM_TYPE'].iloc[index]
            if form_type == '10-K':
                number_10K += 1
            else:
                number_10Q += 1

            companys.add(str(cik))
            
        if (df_file_summaries['FILING_DATE'].iloc[index] != historic_composition_date) & (df_historic_composition['Date'].eq(df_file_summaries['FILING_DATE'].iloc[index]).any()):
            historic_composition_date = df_file_summaries['FILING_DATE'].iloc[index]
            historic_composition_at_date = df_historic_composition[df_historic_composition['Date'] == historic_composition_date].values
            print('Changed composition timestamp to ' + str(historic_composition_date))

    print(f'Number of Form 10-K: {number_10K}')
    print(f'Number of Form 10-Q: {number_10Q}')
    print(f'Number of Companys: {len(list(companys))}')

    output_file = open('./data/reports_per_year.txt', 'w+')
    output_file.write(f'Number of Form 10-K: {number_10K}\nNumber of Form 10-Q: {number_10Q}\nNumber of Companys: {len(list(companys))}')
    output_file.close()
            

def _enumerate(iterable, start=0, step=1):

    index = start
    iterable_list = list(iterable)
    iterable_length = len(iterable_list)
    while index < iterable_length:
        yield (index, iterable_list[index])
        index = index + step

if __name__ == "__main__":
    getReportsPerYear()