import codecs
import collections
import json
import re
import numpy as np
from bert.modeling import BertConfig
from bert.tokenization import FullTokenizer
from bert.extract_features import read_examples, convert_examples_to_features, model_fn_builder, input_fn_builder
import tensorflow as tf
import glob
import pandas as pd
import os
from multiprocessing import Process, current_process, cpu_count, Queue
import time


def extractFeatures():

    start_time = time.time()
    print('Loading multiline_report_index_feature_extraction.csv')
    df_index_file = pd.read_csv('./data/multiline_report_index_feature_extraction.csv', sep='\t')

    available_cuda_processors = [0, 2]

    output_directory = './data/multiline_report_features'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    queue = Queue()
    feature_extraction_processes = []

    for index in range(len(available_cuda_processors)):
        start_index = int(len(df_index_file.index) / len(available_cuda_processors)) * index
        end_index = int(len(df_index_file.index) / len(available_cuda_processors)) * (index + 1)
        splitted_df = df_index_file[start_index:end_index]

        process = Process(target=_processSplittedDataset, args=(splitted_df,
                                                                available_cuda_processors[index],
                                                                output_directory,
                                                                queue))
        process.start()
        feature_extraction_processes.append(process)

    index_building_process = Process(target=_createOutputFile, args=(queue, len(feature_extraction_processes)))
    index_building_process.daemon = True
    index_building_process.start()

    for process in feature_extraction_processes:
        process.join()

    index_building_process.join()

    end_time = time.time()
    duration = end_time - start_time
    duration_file = open('./data/feature_extraction_duration.txt', 'w+')
    duration_file.write(f'{duration}\n')
    duration_file.close()

    print('Finished work.')


def _createOutputFile(input_queue, spawned_processes):
    finished_processes = 0

    index_df = pd.DataFrame()

    while finished_processes < spawned_processes:
        while not input_queue.empty():
            output_from_working_process = input_queue.get()

            if output_from_working_process == 'FINISHED':
                finished_processes += 1
            else:
                index_df = index_df.append(output_from_working_process, ignore_index=True)
                index_df.to_csv('./data/multiline_report_features_index.csv', sep='\t', index=False)

    print('Finished building feature file.')
    

def _processSplittedDataset(splitted_index_df, cuda_target, output_directory, output_queue):
    process_id = os.getpid()
    multiline_report_paths = list(splitted_index_df['File_Path'])

    cik_column = list(splitted_index_df['CIK'])
    ticker_column = list(splitted_index_df['Ticker'])
    company_column = list(splitted_index_df['Company'])
    date_column = list(splitted_index_df['Filing_Date'])
    type_column = list(splitted_index_df['Form_Type'])
    change_ratio_column = list(splitted_index_df['Change_Ratio'])
    change_nominal_column = list(splitted_index_df['Change_Nominal'])

    bert_config_file = './data/BERT/uncased_L-12_H-768_A-12/bert_config.json'
    vocab_file = './data/BERT/uncased_L-12_H-768_A-12/vocab.txt'
    do_lower_case = True

    os.environ['CUDA_VISIBLE_DEVICES'] = f'{cuda_target}'


    tf.logging.set_verbosity(tf.logging.INFO)

    layer_indexes = [-1] # We are only interested in the last layer to obtain our embedding

    bert_config = BertConfig.from_json_file(bert_config_file)

    tokenizer = FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        master=None,
        tpu_config=tf.contrib.tpu.TPUConfig(
            num_shards=8,
            per_host_input_for_training=is_per_host))

    for index, file_path in enumerate(multiline_report_paths):
        print(f'Getting feature vectors for {file_path}')
        feature_vector = _getFeaturesForFile(file_path, layer_indexes, bert_config, tokenizer, run_config)

        feature_columns = list(range(768))
        feature_columns = list(map(lambda column_index: f'F_{column_index}', feature_columns))

        output_df = pd.DataFrame(data=feature_vector, columns=feature_columns)

        splitted_file_path = file_path.split('/')
        output_file_path = f'{output_directory}/{splitted_file_path[len(splitted_file_path)-1]}.csv'

        output_df.to_csv(output_file_path, sep='\t', index=False)

        row_entries = {
            'CIK': str(cik_column[index]),
            'Ticker': str(ticker_column[index]),
            'Company': str(company_column[index]),
            'Filing_Date': str(date_column[index]),
            'Form_Type': str(type_column[index]),
            'Change_Ratio': str(change_ratio_column[index]),
            'Change_Nominal': str(change_nominal_column[index]),
            'File_Path': output_file_path
        }

        output_queue.put(row_entries)

    output_queue.put('FINISHED')
    print(f'Process {process_id} finished feature extraction.')


def _getFeaturesForFile(input_file, layer_indexes, bert_config, tokenizer, run_config):
    max_seq_length = 512
    init_checkpoint = './data/finetuning_512_4/model.ckpt-403308'
    batch_size = 6


    examples = read_examples(input_file)

    features = convert_examples_to_features(
        examples=examples, seq_length=max_seq_length, tokenizer=tokenizer)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=init_checkpoint,
        layer_indexes=layer_indexes,
        use_tpu=False,
        use_one_hot_embeddings=False)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        predict_batch_size=batch_size)

    input_fn = input_fn_builder(
        features=features, seq_length=max_seq_length)

    vectorized_text_segments = []
    for result in estimator.predict(input_fn, yield_single_examples=True):
        layer_output = result["layer_output_0"]
        feature_vec = [
            round(float(x), 6) for x in layer_output[0:(0 + 1)].flat
        ]
        vectorized_text_segments.append(feature_vec)

    return vectorized_text_segments


if __name__ == "__main__":
    extractFeatures()
