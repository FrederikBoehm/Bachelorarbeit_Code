import tensorflow as tf

from absl import logging
from absl import flags

from official.nlp.bert.tokenization import FullTokenizer
from official.nlp.bert.create_pretraining_data import create_training_instances, write_instance_to_example_files
import random
import os
import glob
from multiprocessing import Process, current_process, cpu_count
import pandas as pd

def createPretrainingData():
    FLAGS = flags.FLAGS
    # FLAGS = {}
    # FLAGS["input_file"] = "./data/multiline_reports/multiline_report*"
    # FLAGS["output_file"] = "./data/bert_pretraining_data/seq_128/tf_examples.tfrecord"
    # FLAGS["vocab_file"] = "./data/BERT/uncased_L-12_H-768_A-12/vocab.txt"
    # FLAGS["do_lower_case"] = True
    # FLAGS["max_seq_length"] = 128
    # FLAGS["max_predictions_per_seq"] = 20
    # FLAGS["masked_lm_prob"] = 0.15
    # FLAGS["random_seed"] = 12345
    # FLAGS["dupe_factor"] = 5
    # FLAGS["do_whole_word_mask"] = False
    # FLAGS["short_seq_prob"] = 0.1

    FLAGS.vocab_file = "./data/BERT/uncased_L-12_H-768_A-12/vocab.txt"
    FLAGS.do_lower_case = True
    FLAGS.max_seq_length = 128
    FLAGS.max_predictions_per_seq = 20
    FLAGS.masked_lm_prob = 0.15
    FLAGS.random_seed = 12345
    FLAGS.dupe_factor = 5
    FLAGS.do_whole_word_mask = False
    FLAGS.short_seq_prob = 0.1
    FLAGS.mark_as_parsed()

    logging.set_verbosity(logging.INFO)
    max_seq_length = FLAGS.max_seq_length

    # all_files = glob.glob(FLAGS["input_file"])
    # random.shuffle(all_files)
    # Split array in similar sized arrays an spawn new process for each of them

    # df_train = pd.read_csv('./data/multiline_report_index_train.csv', sep='\t')
    # train_files = df_train['File_Path']
    # train_output_dir = f"./data/bert_pretraining_data/seq_{max_seq_length}/train"
    # if not os.path.exists(train_output_dir):
    #     os.makedirs(train_output_dir)
    # _createData(train_files, output_dir = train_output_dir)

    df_validate = pd.read_csv('./data/multiline_report_index_validate.csv', sep='\t')
    validate_files = df_validate['File_Path']
    validate_output_dir = f"./data/bert_pretraining_data/seq_{max_seq_length}/validate"
    if not os.path.exists(validate_output_dir):
        os.makedirs(validate_output_dir)
    _createData(validate_files, output_dir = validate_output_dir)

    df_test = pd.read_csv('./data/multiline_report_index_test.csv', sep='\t')
    test_files = df_test['File_Path']
    test_output_dir = f"./data/bert_pretraining_data/seq_{max_seq_length}/test"
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)
    _createData(test_files, output_dir = test_output_dir)

    print('Computation completed.')

def _createData(input_files, output_dir):
    cpu_cores = cpu_count()
    processes = []
    print(f'Detected {cpu_cores} cores, splitting dataset...')
    for index in range(cpu_cores):
        start_index = int(len(input_files) / cpu_cores) * index
        end_index = int(len(input_files) / cpu_cores) * (index + 1)
        sObject = slice(start_index, end_index)
        splitted_files_list = input_files[sObject]
        print(splitted_files_list)

        process = Process(target=_handlePretrainingDataCreation, args=(splitted_files_list, start_index, end_index, output_dir))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

def _handlePretrainingDataCreation(files, start_index, end_index, output_dir):
    process_id = os.getpid()
    print(f'Process {process_id} handling files from {start_index} to {end_index}')

    # FLAGS = {}
    # FLAGS["output_file"] = "./data/bert_pretraining_data/seq_128/tf_examples.tfrecord"
    # FLAGS["output_file"] = output_dir + "/" + "tf_examples.tfrecord"
    # FLAGS["vocab_file"] = "./data/BERT/uncased_L-12_H-768_A-12/vocab.txt"
    # FLAGS["do_lower_case"] = True
    # FLAGS["max_seq_length"] = 128
    # FLAGS["max_predictions_per_seq"] = 20
    # FLAGS["masked_lm_prob"] = 0.15
    # FLAGS["random_seed"] = 12345
    # FLAGS["dupe_factor"] = 5
    # FLAGS["do_whole_word_mask"] = False
    # FLAGS["short_seq_prob"] = 0.1
    FLAGS = flags.FLAGS
    FLAGS.output_file = output_dir + "/" + "tf_examples.tfrecord"
    for index, single_file in enumerate(files, start_index):

        print(f'Processing {single_file}')
        
        tokenizer = FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

        input_files = []
        input_files.extend(tf.io.gfile.glob(single_file))

        logging.info("*** Reading from input files ***")
        for input_file in input_files:
            logging.info("  %s", input_file)

        rng = random.Random(FLAGS.random_seed)
        instances = create_training_instances(
            input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
            FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
            rng)

        output_files = (FLAGS.output_file + str(index)).split(",")
        logging.info("*** Writing to output files ***")
        for output_file in output_files:
            logging.info("  %s", output_file)

        write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                        FLAGS.max_predictions_per_seq, output_files)

    print(f'Process {process_id} finished processing.')

if __name__ == "__main__":
    createPretrainingData()