import tensorflow as tf

from absl import logging
from absl import flags

from bert.tokenization import FullTokenizer
from bert.create_pretraining_data import create_training_instances, write_instance_to_example_files
import random
import os
import glob
from multiprocessing import Process, current_process, cpu_count, Queue
import pandas as pd

# Counts how many words can not be tokenized

def checkWordsInVocabulary():
    FLAGS = flags.FLAGS
    FLAGS.mark_as_parsed()

    bert_object = {
        "vocab_file": "./data/BERT/uncased_L-12_H-768_A-12/vocab.txt",
        "do_lower_case": True,
        "max_seq_length": 128,
        "max_predictions_per_seq": 20,
        "masked_lm_prob": 0.15,
        "random_seed": 12345,
        "dupe_factor": 5,
        "short_seq_prob": 0.1
    }

    _getTokensNotInVocab(glob.glob('./data/multiline_reports/multiline_report*'), bert_object)

    print('Computation completed.')

def _getTokensNotInVocab(input_files, bert_object):
    cpu_cores = cpu_count()
    processes = []
    queue = Queue()
    print(f'Detected {cpu_cores} cores, splitting dataset...')
    for index in range(cpu_cores):
        start_index = int(len(input_files) / cpu_cores) * index
        end_index = int(len(input_files) / cpu_cores) * (index + 1)
        sObject = slice(start_index, end_index)
        splitted_files_list = input_files[sObject]
        print(splitted_files_list)

        process = Process(target=_handleVocabCheck, args=(splitted_files_list, start_index, end_index, bert_object, queue))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


    masked_tokens = 0
    not_in_vocab_count = 0
    while not queue.empty():
        process_output = queue.get()
        masked_tokens += process_output["masked_tokens"]
        not_in_vocab_count += process_output["not_in_vocab_count"]

    not_in_vocab_percentage = not_in_vocab_count / masked_tokens
    f = open('./data/words_not_in_vocab.txt', 'w+')
    f.write(f'Number of masked tokens: {masked_tokens}\n')
    f.write(f'Number of tokens not in vocab: {not_in_vocab_count}\n')
    f.write(f'Percentage of tokens not in vocab: {not_in_vocab_percentage}\n')
    f.close()
    

def _handleVocabCheck(files, start_index, end_index, bert_object, queue):
    process_id = os.getpid()
    print(f'Process {process_id} handling files from {start_index} to {end_index}')

    masked_globally = []
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

        masked_in_report = map(lambda instance: instance.masked_lm_labels, instances)
        masked_in_report = [item for sublist in masked_in_report for item in sublist]
        masked_globally.extend(masked_in_report)


    masked_globally = list(set(masked_globally))

    masked_tokens = 0
    not_in_vocab_count = 0
    for token in masked_globally:
        if token == '[UNK]':
            not_in_vocab_count += 1
        
        masked_tokens += 1

    output = {
        "masked_tokens": masked_tokens,
        "not_in_vocab_count": not_in_vocab_count
    }
    queue.put(output)

    print(f'Process {process_id} finished processing.')

if __name__ == "__main__":
    checkWordsInVocabulary()