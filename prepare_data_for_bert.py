import pandas as pd
import numpy as np
import re
from spacy.lang.en import English
import os
import json

def prepareDataForBert():
    print('Loading file summaries...')
    df_file_summaries = pd.read_csv('./data/Edgar/LM_10X_Summaries_2018.csv')
    df_historic_composition = pd.read_csv('./data/S_and_P_historical.csv')

    # Filter for reports after 31.12.2003 and only 10-K's and 10-Q's
    df_file_summaries = df_file_summaries[df_file_summaries['FILING_DATE'].astype(str).str.contains(r'20(?:04|05|06|07|08|09|10|11|12|13|14|15|16|17|18)\S{4}')]
    df_file_summaries = df_file_summaries[(df_file_summaries['FORM_TYPE'] == '10-K') | (df_file_summaries['FORM_TYPE'] == '10-Q')]


    checkpoint = {}
    checkpoint_file = open('./data-preparation.checkpoint.json', 'r') 
    checkpoint = json.load(checkpoint_file)
    checkpoint_file.close()

    # historic_composition_date = 20031231
    historic_composition_date = checkpoint["historic_composition_date"] if checkpoint["historic_composition_date"] else 20031231
    historic_composition_at_date = df_historic_composition[df_historic_composition['Date'] == historic_composition_date].values
    # count = 0
    count = checkpoint["count"] if checkpoint["count"] else 0
    init_index = checkpoint["index"] if checkpoint["index"] else 0
    for index, cik in enumerate(df_file_summaries['CIK'][init_index:], start=init_index):
        # if historic_composition_at_date.contains(cik):
        if cik in historic_composition_at_date:
            print('Progressing for CIK ' + str(cik))
            report_file_path = df_file_summaries['FILE_NAME'].iloc[index]
            report_file_path = report_file_path.replace('D:', './data').replace('\\', '/')
            
            f = open(report_file_path, 'r')
            if f.mode == 'r':
                report = f.read()

                print('Starting preprocessing for ' + report_file_path)
                report = _remove_header(report)
                report = _remove_exhibits(report)
                report = _make_single_line(report)

                sentences = _splitReportToSentences(report)                

                seperator = '\n'
                multiline_report = seperator.join(sentences)

                if not os.path.exists('./data/multiline_reports'):
                    os.makedirs('./data/multiline_reports')

                output_file_path = './data/multiline_reports/multiline_report' + str(count)
                output_file = open(output_file_path, 'w+')
                output_file.write(multiline_report)
                output_file.close()
                count = count + 1
                print('Wrote processed report to ' + output_file_path)

            else:
                print('Unable to read file ' + report_file_path)

        if (df_file_summaries['FILING_DATE'].iloc[index] != historic_composition_date) & (df_historic_composition['Date'].eq(df_file_summaries['FILING_DATE'].iloc[index]).any()):
            historic_composition_date = df_file_summaries['FILING_DATE'].iloc[index]
            historic_composition_at_date = df_historic_composition[df_historic_composition['Date'] == historic_composition_date].values
            print('Changed composition timestamp to ' + str(historic_composition_date))

        checkpoint_file = open('./data-preparation.checkpoint.json', 'w', encoding='utf-8')
        checkpoint = {
            "historic_composition_date": int(historic_composition_date),
            "index": index,
            "count": count
        }
        checkpoint_file.write(json.dumps(checkpoint))
        checkpoint_file.close()

def _remove_header(input):
    regex = r'<\/Header>'
    match = re.search(regex, input)
    if match:
        sObject = slice(match.span()[1], len(input))
        return input[sObject]
    else:
        return input

def _remove_exhibits(input):
    regex = r'<EX-.*>'
    match = re.search(regex, input)
    if match:
        sObject = slice(0, match.span()[0])
        return input[sObject]
    else:
        return input

def _make_single_line(input):
    regex = r'\s+'
    return re.sub(regex, ' ', input)

def _splitReportToSentences(report):
    nlp = English()
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)

    sentences = []
    if len(report) < nlp.max_length:
        doc = nlp(report)
        for sentence in doc.sents:
            sentences.extend(sentence.text)
    else:
        print(f'Document too long, processing in two parts...')
        # If the report is longer than what spacy can process we first split the report in half and process the first part...
        sObject = slice(int(len(report) / 2))
        sliced_report = report[sObject]
        sentences.extend(_splitReportToSentences(sliced_report))
        # ... and now the second part
        sObject = slice(int(len(report) / 2), len(report))
        sliced_report = report[sObject]
        sentences.extend(_splitReportToSentences(sliced_report))

    return sentences

if __name__ == "__main__":
    prepareDataForBert()