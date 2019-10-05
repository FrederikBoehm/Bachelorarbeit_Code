import codecs
import collections
import json
import re
import numpy as np
from bert.modeling import BertConfig
from bert.tokenization import FullTokenizer
import tensorflow as tf
import glob
from bert.extract_features import read_examples, convert_examples_to_features, model_fn_builder, input_fn_builder
import pandas as pd

def extractFeatures():

    # input_file_path = './data/multiline_reports/multiline_report*'

    # input_files = glob.glob(input_file_path)

    # for input_file in input_files:
    #     _getFeaturesForFile(input_file)
    print('Loading multiline_report_index.csv')
    df_index_file = pd.read_csv('./data/multiline_report_index.csv', sep='\t')

    column_names_list = ['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type']
    feature_columns = list(range(768))
    feature_columns = list(map(lambda column_index: f'F_{column_index}', feature_columns))
    column_names_list.extend(feature_columns)
    seperator = '\t'
    column_names = seperator.join(column_names_list)
    column_names = f'{column_names}\n'

    report_features_file = open('./data/report_features.csv', 'w')
    report_features_file.write(column_names)
    report_features_file.close()

    multiline_report_paths = list(df_index_file['File_Path'])

    cik_column = list(df_index_file['CIK'])
    ticker_column = list(df_index_file['Ticker'])
    company_column = list(df_index_file['Company'])
    date_column = list(df_index_file['Filing_Date'])
    type_column = list(df_index_file['Form_Type'])

    FLAGS = {}
    FLAGS["bert_config_file"] = './data/BERT/uncased_L-12_H-768_A-12/bert_config.json'
    FLAGS["vocab_file"] = './data/BERT/uncased_L-12_H-768_A-12/vocab.txt'
    FLAGS["do_lower_case"] = True
    FLAGS["master"] = None
    FLAGS["num_tpu_cores"] = 8


    tf.logging.set_verbosity(tf.logging.INFO)

    # layer_indexes = [int(x) for x in FLAGS.layers.split(",")]
    layer_indexes = [-1] # We are only interested in the last layer to obtain our embedding

    bert_config = BertConfig.from_json_file(FLAGS["bert_config_file"])

    tokenizer = FullTokenizer(
        vocab_file=FLAGS["vocab_file"], do_lower_case=FLAGS["do_lower_case"])

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        master=FLAGS["master"],
        tpu_config=tf.contrib.tpu.TPUConfig(
            num_shards=FLAGS["num_tpu_cores"],
            per_host_input_for_training=is_per_host))

    for index, file_path in enumerate(multiline_report_paths):
        print(f'Getting feature vectors for {file_path}')
        feature_vector = _getFeaturesForFile(file_path, layer_indexes, bert_config, tokenizer, run_config)
        feature_vector = [str(i) for i in feature_vector]

        seperator = '\t'
        features_as_string = seperator.join(feature_vector)
        features_as_string = f'{features_as_string}'

        row = f'{cik_column[index]}\t{ticker_column[index]}\t{company_column[index]}\t{date_column[index]}\t{type_column[index]}\t{features_as_string}\n'

        report_features_file = open('./data/report_features.csv', 'a')
        report_features_file.write(row)
        report_features_file.close()


def _getFeaturesForFile(input_file, layer_indexes, bert_config, tokenizer, run_config):
    FLAGS = {}
    # FLAGS["input_file"] = './data/multiline_reports/multiline_report*'
    FLAGS["max_seq_length"] = 512
    FLAGS["init_checkpoint"] = './data/BERT/uncased_L-12_H-768_A-12/bert_model.ckpt'
    FLAGS["batch_size"] = 32
    FLAGS["use_tpu"] = False
    FLAGS["use_one_hot_embeddings"] = False

    examples = read_examples(input_file)

    features = convert_examples_to_features(
        examples=examples, seq_length=FLAGS["max_seq_length"], tokenizer=tokenizer)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS["init_checkpoint"],
        layer_indexes=layer_indexes,
        use_tpu=FLAGS["use_tpu"],
        use_one_hot_embeddings=FLAGS["use_one_hot_embeddings"])

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS["use_tpu"],
        model_fn=model_fn,
        config=run_config,
        predict_batch_size=FLAGS["batch_size"])

    input_fn = input_fn_builder(
        features=features, seq_length=FLAGS["max_seq_length"])

    vectorized_text_segments = []
    for result in estimator.predict(input_fn, yield_single_examples=True):
        layer_output = result["layer_output_0"]
        feature_vec = [
            round(float(x), 6) for x in layer_output[0:(0 + 1)].flat
        ]
        vectorized_text_segments.append(feature_vec)

    vectorized_text_segments = np.array(vectorized_text_segments)

    output = np.mean(vectorized_text_segments, axis = 0)

    return output

if __name__ == "__main__":
    extractFeatures()
