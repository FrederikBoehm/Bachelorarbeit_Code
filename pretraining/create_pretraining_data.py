import tensorflow as tf

from absl import logging
from absl import flags

from bert.tokenization import FullTokenizer
from bert.create_pretraining_data import create_training_instances, write_instance_to_example_files
import random
import os
import glob
from multiprocessing import Process, current_process, cpu_count
import pandas as pd

# Converts the text files, which contain the reports into TFRecord file format for pre-training

def createPretrainingData():
    FLAGS = flags.FLAGS
    FLAGS.mark_as_parsed()

    bert_object = {
        "vocab_file": "./data/BERT/uncased_L-12_H-768_A-12/vocab.txt",
        "do_lower_case": True,
        "max_seq_length": 128,
        "max_predictions_per_seq" = 20,
        "masked_lm_prob": 0.15,
        "random_seed": 12345,
        "dupe_factor": 5,
        "short_seq_prob": 0.1
    }

    logging.set_verbosity(logging.INFO)

    df_train = pd.read_csv('./data/multiline_report_index_train.csv', sep='\t')
    train_files = df_train['File_Path']
    train_output_dir = f"./data/bert_pretraining_data/seq_{bert_object["max_seq_length"]}/train"
    if not os.path.exists(train_output_dir):
        os.makedirs(train_output_dir)
    _splitDataAndSpawnProcesses(train_files, output_dir = train_output_dir, bert_object = bert_object)

    df_validate = pd.read_csv('./data/multiline_report_index_validate.csv', sep='\t')
    validate_files = df_validate['File_Path']
    validate_output_dir = f"./data/bert_pretraining_data/seq_{bert_object["max_seq_length"]}/validate"
    if not os.path.exists(validate_output_dir):
        os.makedirs(validate_output_dir)
    _splitDataAndSpawnProcesses(validate_files, output_dir = validate_output_dir, bert_object = bert_object)

    df_test = pd.read_csv('./data/multiline_report_index_test.csv', sep='\t')
    test_files = df_test['File_Path']
    test_output_dir = f"./data/bert_pretraining_data/seq_{bert_object["max_seq_length"]}/test"
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)
    _splitDataAndSpawnProcesses(test_files, output_dir = test_output_dir, bert_object = bert_object)

    print('Computation completed.')

def _splitDataAndSpawnProcesses(input_files, output_dir, bert_object):
    cpu_cores = cpu_count()
    processes = []
    print(f'Detected {cpu_cores} cores, splitting dataset...')
    for index in range(cpu_cores):
        start_index = int(len(input_files) / cpu_cores) * index
        end_index = int(len(input_files) / cpu_cores) * (index + 1)
        sObject = slice(start_index, end_index)
        splitted_files_list = input_files[sObject]
        print(splitted_files_list)

        process = Process(target=_handlePretrainingDataCreation, args=(splitted_files_list, start_index, end_index, output_dir, bert_object))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

def _handlePretrainingDataCreation(files, start_index, end_index, output_dir):
    process_id = os.getpid()
    print(f'Process {process_id} handling files from {start_index} to {end_index}')

    bert_object["output_file"] = output_dir + "/" + "tf_examples.tfrecord"
    for index, single_file in enumerate(files, start_index):

        print(f'Processing {single_file}')
        
        tokenizer = FullTokenizer(
            vocab_file=bert_object["vocab_file"], do_lower_case=bert_object["do_lower_case"])

        input_files = []
        input_files.extend(tf.io.gfile.glob(single_file))

        logging.info("*** Reading from input files ***")
        for input_file in input_files:
            logging.info("  %s", input_file)

        rng = random.Random(bert_object["random_seed"])
        instances = create_training_instances(
            input_files, tokenizer, bert_object["max_seq_length"], bert_object["dupe_factor"],
            bert_object["short_seq_prob"], bert_object["masked_lm_prob"], bert_object["max_predictions_per_seq"],
            rng)

        output_files = (bert_object["output_file"] + str(index)).split(",")
        logging.info("*** Writing to output files ***")
        for output_file in output_files:
            logging.info("  %s", output_file)

        write_instance_to_example_files(instances, tokenizer, bert_object["max_seq_length"],
                                        bert_object["max_predictions_per_seq"], output_files)

    print(f'Process {process_id} finished processing.')

if __name__ == "__main__":
    createPretrainingData()