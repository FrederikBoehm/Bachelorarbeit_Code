import pandas as pd
import numpy as np
import re
from spacy.lang.en import English
import os
import json
import sys
import logging
import subprocess
from multiprocessing import Process, current_process, cpu_count, Queue
import time


# Parses business reports from companys in the S&P 500 between the 31/12/2003 and the 31/12/2018:
# Removes the headers and exhibits
# Applies spaCy sentencizer
# Removes short and long sentences, sentences that include only capitals, sentences that contain only ... or ---

def prepareDataForBert():
    start_time = time.time()

    print('Loading file summaries...')
    df_file_summaries = pd.read_csv('./data/Edgar/LM_10X_Summaries_2018.csv')
    df_historic_composition = pd.read_csv('./data/S_and_P_historical.csv')

    # Filter for reports after 31.12.2003 and only 10-K's and 10-Q's
    df_file_summaries = df_file_summaries[df_file_summaries['FILING_DATE'].astype(str).str.contains(r'20(?:04|05|06|07|08|09|10|11|12|13|14|15|16|17|18)\S{4}')]
    df_file_summaries = df_file_summaries[(df_file_summaries['FORM_TYPE'] == '10-K') | (df_file_summaries['FORM_TYPE'] == '10-Q')]
    df_cik_ticker_mapping = pd.read_csv('./data/cik_ticker_modified.csv', delimiter='|')

    checkpoint = {}
    checkpoint_file = open('./data-preparation-test.checkpoint.json', 'r') 
    checkpoint = json.load(checkpoint_file)
    checkpoint_file.close()

    historic_composition_date = checkpoint["historic_composition_date"] if checkpoint["historic_composition_date"] else 20031231
    init_index = checkpoint["index"] if checkpoint["index"] else 0

    if init_index == 0:
        multiline_report_index = open('./data/multiline_report_index.csv', 'w')
        multiline_report_index.write('CIK\tTicker\tCompany\tFiling_Date\tForm_Type\tFile_Path\n')
        multiline_report_index.close()

    cpu_cores = cpu_count()
    print(f'Detected {cpu_cores} cores, splitting dataset...')
    start_indexes = []
    for index in range(cpu_cores):
        start_index = init_index + index
        if (start_index < len(df_file_summaries.index)):
            start_indexes.append(start_index)

    print(start_indexes)
    if not os.path.exists('./data/multiline_reports'):
        os.makedirs('./data/multiline_reports')

    queue = Queue()
        
    document_parsing_processes = []
    for start_index in start_indexes:
        
        process = Process(target=_handleDocumentParsing, args=(df_file_summaries,
                                                               df_cik_ticker_mapping,
                                                               df_historic_composition,
                                                               start_index,
                                                               historic_composition_date,
                                                               len(start_indexes),
                                                               queue))
        process.start()
        document_parsing_processes.append(process)

    index_building_process = Process(target=_buildIndexFile, args=(queue, len(start_indexes)))
    index_building_process.daemon = True
    index_building_process.start()

    for process in document_parsing_processes:
        process.join()

    index_building_process.join()

    print('Finished work.')
    end_time = time.time()

    duration = end_time - start_time
    print(f"Duration: {duration}")


def _handleDocumentParsing(df_file_summaries, df_cik_ticker_mapping, df_historic_composition, init_index, historic_composition_date, spawned_processes, output_queue):
    process_id = os.getpid()
    historic_composition_at_date = df_historic_composition[df_historic_composition['Date'] == historic_composition_date].values

    for index, cik in _enumerate(df_file_summaries['CIK'], start=init_index, step=spawned_processes):
        if cik in historic_composition_at_date:
            print('Progressing for CIK ' + str(cik))
            report_file_path = df_file_summaries['FILE_NAME'].iloc[index]
            report_file_path = report_file_path.replace('D:', './data').replace('\\', '/')
            
            print('Starting preprocessing for ' + report_file_path)
            f = open(report_file_path, 'r')
            report = f.read()

            report = _removeHeader(report)
            report = _removeExhibits(report)
            report = _makeSingleLine(report)

            sentences = _splitReportToSentences(report)      

            sentences = _getMeaningfulSequences(sentences)          

            seperator = '\n'
            multiline_report = seperator.join(sentences)

            output_file_path = './data/multiline_reports/multiline_report' + str(index)
            output_file = open(output_file_path, 'w+')
            output_file.write(multiline_report)
            output_file.close()
            print('Wrote processed report to ' + output_file_path)

            output_data = {}
            output_data['filing_date'] = df_file_summaries['FILING_DATE'].iloc[index]
            output_data['form_type'] = df_file_summaries['FORM_TYPE'].iloc[index]
            output_data['ticker'] = df_cik_ticker_mapping.loc[df_cik_ticker_mapping['CIK'] == cik, 'Ticker'].iloc[0]
            output_data['cik'] = cik
            output_data['company'] = df_cik_ticker_mapping.loc[df_cik_ticker_mapping['CIK'] == cik, 'Name'].iloc[0]
            output_data['output_file_path'] = output_file_path
            output_data['index'] = index
            output_data['historic_composition_date'] = historic_composition_date
            
            process_output = {}
            process_output[f'{process_id}'] = output_data

            output_queue.put(process_output)

        if (df_file_summaries['FILING_DATE'].iloc[index] != historic_composition_date) & (df_historic_composition['Date'].eq(df_file_summaries['FILING_DATE'].iloc[index]).any()):
            historic_composition_date = df_file_summaries['FILING_DATE'].iloc[index]
            historic_composition_at_date = df_historic_composition[df_historic_composition['Date'] == historic_composition_date].values
            print('Changed composition timestamp to ' + str(historic_composition_date))
            

    print(f'Process {process_id} finished data preparation.')
    process_output = {}
    process_output[f'{process_id}'] = 'FINISHED'

    output_queue.put(process_output)

def _enumerate(iterable, start=0, step=1):

    index = start
    iterable_list = list(iterable)
    iterable_length = len(iterable_list)
    while index < iterable_length:
        yield (index, iterable_list[index])
        index = index + step

def _buildIndexFile(input_queue, spawned_processes):
    finished_processes = []

    previous_index = 0
    while len(finished_processes) < spawned_processes:
        output_from_working_processes = []
        while not input_queue.empty():
            output_from_working_process = input_queue.get()
            pid = list(output_from_working_process.keys())[0]

            if output_from_working_process[f'{pid}'] == 'FINISHED':
                finished_processes.append(pid)
            else:
                output_from_working_processes.append(output_from_working_process[f'{pid}'])

        output_from_working_processes.sort(key=lambda process_output: process_output['index'])

        for process_output in output_from_working_processes:
            print('Building Index file...')
            filing_date = process_output['filing_date']
            form_type = process_output['form_type']
            ticker = process_output['ticker']
            cik = process_output['cik']
            company = process_output['company']
            output_file_path = process_output['output_file_path']
            
            print(f'Adding entry with CIK {cik}, Ticker {ticker}, Company {company}, Filing Date {filing_date}, Form Type {form_type} Output File Path {output_file_path} .')
            multiline_report_index = open('./data/multiline_report_index.csv', 'a')
            multiline_report_index.write(f'{cik}\t{ticker}\t{company}\t{filing_date}\t{form_type}\t{output_file_path}\n')
            multiline_report_index.close()

        if (len(output_from_working_processes) > 0):
            last_output = output_from_working_processes[0]
            index = last_output['index']
            historic_composition_date = last_output['historic_composition_date']
            if (previous_index < index):
                print(f'Updating checkpoint. Index: {index}, Historic Composition date: {historic_composition_date}')
                checkpoint_file = open('./data-preparation-test.checkpoint.json', 'w', encoding='utf-8')
                checkpoint = {
                    "historic_composition_date": int(historic_composition_date),
                    "index": index
                }
                checkpoint_file.write(json.dumps(checkpoint))
                checkpoint_file.close()
                previous_index = index

    print('Finished building index file.')


def _removeHeader(input):
    regex = r'<\/Header>'
    match = re.search(regex, input)
    if match:
        sObject = slice(match.span()[1], len(input))
        return input[sObject]
    else:
        return input

def _removeExhibits(input):
    regex = r'<EX-.*>'
    match = re.search(regex, input)
    if match:
        sObject = slice(0, match.span()[0])
        return input[sObject]
    else:
        return input

def _makeSingleLine(input):
    regex = r'\s+'
    return re.sub(regex, ' ', input)

def _splitReportToSentences(report):
    nlp = English()
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)

    sentences = []
    if len(report) < nlp.max_length:
        doc = nlp(report)
        sentences = [sent.string.strip() for sent in doc.sents]
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

def _getMeaningfulSequences(sequences):
    output_sequences = []
    for sequence in sequences:
        number_of_words = len(re.findall(r'\w+', sequence))
        capital_sequence = lambda string: re.search(r'[A-Z]+\s+[A-Z]+', string) # We match for sequences like "NOTES TO CONDENSED CONSOLIDATED FINANCIAL STATEMENTS"
        toc_match = lambda string: re.search(r'table of contents', string, re.IGNORECASE) # We match for sequences which were links to the toc
        repeated_characters = lambda string: re.search(r'(.)\1{2,}', string) # We match for sequences like ... or --- which indicate a headline or table
        if number_of_words >= 8 and number_of_words <= 95 and not capital_sequence(sequence) and not toc_match(sequence) and not repeated_characters(sequence):
            output_sequences.append(sequence)

    return output_sequences

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    prepareDataForBert()