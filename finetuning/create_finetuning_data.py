import pandas as pd
import random
import os
from bert.tokenization import FullTokenizer, convert_to_unicode
from bert.run_classifier import file_based_convert_examples_to_features, DataProcessor, InputExample
from multiprocessing import Process, current_process, cpu_count
from report_processor import ReportProcessor
try:
    from concat_to_max_sequence_length import concatToMaxSequenceLength
except ImportError:
    from shared.concat_to_max_sequence_length import concatToMaxSequenceLength

def createFinetuningData():
    max_seq_length = 512
    
    _createBalancedData('./data/multiline_report_index_train.csv', './data/fine_tuning_data_train.csv', max_seq_length)
    _createBalancedData('./data/multiline_report_index_validate.csv', './data/fine_tuning_data_validate.csv', max_seq_length)
    _createBalancedData('./data/multiline_report_index_test.csv', './data/fine_tuning_data_test.csv', max_seq_length)

    processor = ReportProcessor()

    train_examples = processor.get_train_examples('./data')
    f = open('./data/fine_tuning_examples_number.txt', 'w+')
    f.write(f'Number of train examples: {len(train_examples)}\n')
    f.close()

    validate_examples = processor.get_validate_examples('./data')
    f = open('./data/fine_tuning_examples_number.txt', 'a')
    f.write(f'Number of validate examples: {len(validate_examples)}\n')
    f.close()

    test_examples = processor.get_test_examples('./data')
    f = open('./data/fine_tuning_examples_number.txt', 'a')
    f.write(f'Number of test examples: {len(test_examples)}\n')
    f.close()

    vocab_file = './data/BERT/uncased_L-12_H-768_A-12/vocab.txt'

    train_process = Process(target=_convertToTfrecord, args=(train_examples, vocab_file, './data/bert_finetuning_data/train', 'train.tf_record', max_seq_length))
    train_process.start()
    validate_process = Process(target=_convertToTfrecord, args=(validate_examples, vocab_file, './data/bert_finetuning_data/validate', 'validate.tf_record', max_seq_length))
    validate_process.start()
    test_process = Process(target=_convertToTfrecord, args=(test_examples, vocab_file, './data/bert_finetuning_data/test', 'test.tf_record', max_seq_length))
    test_process.start()

    train_process.join()
    validate_process.join()
    test_process.join()


def _convertToTfrecord(examples, vocab_file, output_dir, output_file, max_seq_length):
    processor = ReportProcessor()
    label_list = processor.get_labels()
    tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file_path = os.path.join(output_dir, output_file)

    print('Converting files to tfrecord.')
    file_based_convert_examples_to_features(
            examples, label_list, max_seq_length, tokenizer, output_file_path)
    print(f'Finished tfrecord creation at {output_file_path}')



def _createBalancedData(index_file, output_file_path, max_seq_length):

    reports_index_df = pd.read_csv(index_file, sep='\t')
    reports_index = reports_index_df.to_dict('records')

    pos_sequences = []
    neg_sequences = []
    for report in reports_index:
        file_path = report['File_Path']
        print(f'Processing {file_path}')

        price_change = report['Change_Nominal']

        with open(file_path) as f:
            lines = [line.rstrip('\n') for line in f]

        lines = concatToMaxSequenceLength(lines, max_seq_length)

        if price_change == 'positive':
            pos_sequences.extend(lines)
        else:
            neg_sequences.extend(lines)

    print('Shuffling sequences...')
    random.shuffle(pos_sequences)
    random.shuffle(neg_sequences)

    print('Balancing data...')
    if len(pos_sequences) > len(neg_sequences):
        pos_sequences = pos_sequences[:len(neg_sequences)]
    else:
        neg_sequences = neg_sequences[:len(pos_sequences)]

    print('Merging positive and negative sequences...')
    labeled_sequences = []

    for pos_sequence in pos_sequences:
        labeled_sequence = {
            'label': 1,
            'sequence': pos_sequence
        }

        labeled_sequences.append(labeled_sequence)

    for neg_sequence in neg_sequences:
        labeled_sequence = {
            'label': 0,
            'sequence': neg_sequence
        }

        labeled_sequences.append(labeled_sequence)

    print('Shuffling sequences...')
    random.shuffle(labeled_sequences)

    print(f'Writing to file {output_file_path}')
    labeled_sequences_df = pd.DataFrame(labeled_sequences)
    labeled_sequences_df.to_csv(output_file_path, sep='\t', index=False)


if __name__ == '__main__':
    createFinetuningData()
