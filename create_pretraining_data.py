import tensorflow as tf
from bert.tokenization import FullTokenizer
from bert.create_pretraining_data import create_training_instances, write_instance_to_example_files
import random
import os
import glob

def createPretrainingData():
    FLAGS = {}
    FLAGS["input_file"] = "./data/multiline_reports/multiline_report*"
    FLAGS["output_file"] = "./data/bert_pretraining_data/seq_128/tf_examples.tfrecord"
    FLAGS["vocab_file"] = "./data/BERT/uncased_L-12_H-768_A-12/vocab.txt"
    FLAGS["do_lower_case"] = True
    FLAGS["max_seq_length"] = 128
    FLAGS["max_predictions_per_seq"] = 20
    FLAGS["masked_lm_prob"] = 0.15
    FLAGS["random_seed"] = 12345
    FLAGS["dupe_factor"] = 5
    FLAGS["do_whole_word_mask"] = False
    FLAGS["short_seq_prob"] = 0.1

    tf.logging.set_verbosity(tf.logging.INFO)

    if not os.path.exists('./data/bert_pretraining_data/seq_128'):
        os.makedirs('./data/bert_pretraining_data/seq_128')

    all_files = glob.glob(FLAGS["input_file"])
    random.shuffle(all_files)
    for index, single_file in enumerate(all_files[:10]):

        tokenizer = FullTokenizer(
            vocab_file=FLAGS["vocab_file"], do_lower_case=FLAGS["do_lower_case"])

        input_files = []
        input_files.extend(tf.gfile.Glob(single_file))

        tf.logging.info("*** Reading from input files ***")
        for input_file in input_files:
            tf.logging.info("  %s", input_file)

        rng = random.Random(FLAGS["random_seed"])
        instances = create_training_instances(
            input_files, tokenizer, FLAGS["max_seq_length"], FLAGS["dupe_factor"],
            FLAGS["short_seq_prob"], FLAGS["masked_lm_prob"], FLAGS["max_predictions_per_seq"],
            rng)

        output_files = (FLAGS["output_file"] + str(index)).split(",")
        tf.logging.info("*** Writing to output files ***")
        for output_file in output_files:
            tf.logging.info("  %s", output_file)

        write_instance_to_example_files(instances, tokenizer, FLAGS["max_seq_length"],
                                        FLAGS["max_predictions_per_seq"], output_files)

if __name__ == "__main__":
    createPretrainingData()