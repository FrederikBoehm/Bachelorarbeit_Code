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
import re

def getSubTokens():
    FLAGS = flags.FLAGS
    # FLAGS.vocab_file = "./data/BERT/uncased_L-12_H-768_A-12/vocab.txt"
    FLAGS.vocab_file = "./vocab_modified.txt"
    FLAGS.do_lower_case = True
    FLAGS.max_seq_length = 128
    FLAGS.max_predictions_per_seq = 0
    FLAGS.masked_lm_prob = 0
    FLAGS.random_seed = 12345
    FLAGS.dupe_factor = 5
    # FLAGS.do_whole_word_mask = False
    FLAGS.short_seq_prob = 0.1
    FLAGS.mark_as_parsed()

    _getSubTokens(glob.glob('./data/multiline_reports/multiline_report*'))

    print('Computation completed.')

def _getSubTokens(input_files):
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

        process = Process(target=_handleTokenization, args=(splitted_files_list, start_index, end_index, queue), daemon=True)
        process.start()
        processes.append(process)


    print('Combining process results...')
    sub_tokenized_sequences = {}
    finished_processes = 0
    while finished_processes < cpu_cores:
        while not queue.empty():
            process_output = queue.get()
            
            if process_output == 'FINISHED':
                finished_processes += 1
            else:
                for sub_tokenized_sequence in list(process_output.keys()):
                    if sub_tokenized_sequence in sub_tokenized_sequences:
                        sub_tokenized_sequences[sub_tokenized_sequence] += process_output[sub_tokenized_sequence]
                    else:
                        sub_tokenized_sequences[sub_tokenized_sequence] = process_output[sub_tokenized_sequence]

    for process in processes:
        process.join()
    
    print('Writing to file.')
    f = open('./data/sub_tokens.csv', 'w+')
    sub_tokenized_sequences_list = list(sub_tokenized_sequences.items())
    sub_tokenized_sequences_list.sort(key=lambda sub_tokenized_sequence: sub_tokenized_sequence[1], reverse=True)
    for sub_tokenized_sequence, number_of_occurences in sub_tokenized_sequences_list:
        output_string = " ".join(sub_tokenized_sequence) + "," + str(number_of_occurences) + "\n"
        f.write(output_string)
    f.close()
    

def _handleTokenization(files, start_index, end_index, queue):
    process_id = os.getpid()
    print(f'Process {process_id} handling files from {start_index} to {end_index}')

    FLAGS = flags.FLAGS
    sub_tokenized_sequences = {}
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


        tokens_in_file = map(lambda instance: instance.tokens, instances)

        for tokens_in_line in tokens_in_file:
            
            sub_tokens_in_line = _getSubTokenizedSequences(tokens_in_line)

            for sub_tokenized_sequence in list(sub_tokens_in_line.keys()):
                if sub_tokenized_sequence in sub_tokenized_sequences:
                    sub_tokenized_sequences[sub_tokenized_sequence] += sub_tokens_in_line[sub_tokenized_sequence]
                else:
                    sub_tokenized_sequences[sub_tokenized_sequence] = sub_tokens_in_line[sub_tokenized_sequence]


    queue.put(sub_tokenized_sequences)
    queue.put('FINISHED')

    print(f'Process {process_id} finished processing.')


def _getSubTokenizedSequences(tokens):

    sub_tokenized_sequences = {}

    index = 1
    while index < len(tokens):

        token = tokens[index]

        if re.search(r"##.+", token):
            sub_tokenized_sequence = [token]

            # Iterate back to get start of sub-tokenized sequence
            back_index = index - 1
            while back_index > 0 and re.search(r"##.+", tokens[back_index]):
                sub_tokenized_sequence.insert(0, tokens[back_index])
                back_index -= 1
            sub_tokenized_sequence.insert(0, tokens[back_index])

            # Iterate forward to get end of sub-tokenized sequence
            if index < len(tokens) - 1:
                forward_index = index + 1
                while forward_index < len(tokens) and re.search(r"##.+", tokens[forward_index]):
                    sub_tokenized_sequence.insert(len(sub_tokenized_sequence), tokens[forward_index])
                    forward_index += 1
                
                index = forward_index

            sub_tokenized_sequence = tuple(sub_tokenized_sequence)
            if sub_tokenized_sequence in sub_tokenized_sequences:
                sub_tokenized_sequences[sub_tokenized_sequence] += 1
            else:
                sub_tokenized_sequences[sub_tokenized_sequence] = 1

        index += 1

    return sub_tokenized_sequences

if __name__ == "__main__":
    getSubTokens()