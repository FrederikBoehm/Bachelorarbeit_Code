import pandas as pd
import re

def createHistoricSandPComposition():
    print("Starting composition...")

    df_cik_ticker = pd.read_csv('./data/cik_ticker_modified.csv', delimiter='|')
    df_composition = pd.read_csv('./data/S_and_P_500_Composition.csv', delimiter='\t')

    unavailable_tickers = set()
    historical_component_list = []
    for label, column in df_composition.iteritems():
        if (label == '30/09/2003'):
            break
        
        if ((label != 'Ticker') and (label != 'ISIN Code') and (label != 'Company Name')):
            counter = 0
            #historical_component_list.append(label)
            components_at_date = set()
            for index, line_content in enumerate(column):
                if line_content == 'X':
                    ticker = df_composition['Ticker'][index]
                    #cik = df_cik_ticker.loc[df_cik_ticker['Ticker'] == ticker, 'CIK'].values[0]
                    ticker = re.sub(r'\s\(\S*\)', '', ticker)
                    cik = ''
                    for index2, line_content2 in enumerate(df_cik_ticker['Ticker']):
                        if line_content2 == ticker:
                            cik = df_cik_ticker['CIK'][index2]
                            break
                    
                    if cik == '':
                        unavailable_tickers.add(ticker)
                    else:
                        components_at_date.add(cik)
                        counter = counter + 1
            
            splitted_date = label.split('/')
            date = splitted_date[2] + splitted_date[1] + splitted_date[0]
            historical_component_list_row = []
            historical_component_list_row.append(date)
            components_at_date_list = list(components_at_date)

            while len(components_at_date_list) < 500:
                components_at_date_list.append('')
                
            if len(components_at_date_list) > 500:
                sObject = slice(500)
                components_at_date_list = components_at_date_list[sObject]
            
            historical_component_list_row.extend(components_at_date_list)
            historical_component_list.append(historical_component_list_row)
            print("Finished composition for " + date)

    cik_labels = []

    for index in range(500):
        cik_labels.append('CIK_' + str(index))
        
    dataframe_column_names = []
    dataframe_column_names.append('Date')
    dataframe_column_names.extend(cik_labels)

    dataframe = pd.DataFrame(historical_component_list, columns = dataframe_column_names)

    print("Writing dataframe to file...")
    print(dataframe.head())
    dataframe.to_csv('./data/S_and_P_historical.csv')
    print("Finished.")

if __name__ == "__main__":
    createHistoricSandPComposition()