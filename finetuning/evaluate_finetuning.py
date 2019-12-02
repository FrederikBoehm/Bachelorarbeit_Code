import tensorflow as tf
from absl import flags
from absl import logging
from bert.tokenization import validate_case_matches_checkpoint, convert_to_unicode
from bert.modeling import BertConfig
from bert.run_classifier import DataProcessor, model_fn_builder, file_based_convert_examples_to_features, file_based_input_fn_builder, PaddingInputExample, InputExample
from report_processor import ReportProcessor
import os
import time
import glob
import re
import pandas as pd

def evaluateFinetuning():

    FLAGS = flags.FLAGS
    FLAGS.mark_as_parsed()

    bert_config_file = './data/BERT/uncased_L-12_H-768_A-12/bert_config.json'
    # vocab_file = './data/BERT/uncased_L-12_H-768_A-12/vocab.txt'
    output_dir = './data'
    max_seq_length = 512
    train_batch_size = 6
    eval_batch_size = 6
    learning_rate = 2e-5
    num_train_epochs = 4
    warmup_proportion = 0.1
    save_checkpoints_steps = 22406
    iterations_per_loop = 1000

    number_of_train_examples = 672176
    number_of_validate_examples = 231762

    eval_file = "./data/bert_finetuning_data/test/test.tf_record"

    num_train_steps = int(
            number_of_train_examples / train_batch_size * num_train_epochs)
    num_warmup_steps = int(num_train_steps * warmup_proportion)

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    logging.set_verbosity(logging.INFO)

    bert_config = BertConfig.from_json_file(bert_config_file)

    label_list = ReportProcessor().get_labels()

    tpu_cluster_resolver = None

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=None,
        model_dir=output_dir,
        save_checkpoints_steps=save_checkpoints_steps,
        keep_checkpoint_max=30,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=iterations_per_loop,
            num_shards=None,
            per_host_input_for_training=is_per_host))


    finetuned_checkpoints = './data/bert_finetuning_checkpoints/model.ckpt-*'
    finetuned_checkpoints_list = glob.glob(finetuned_checkpoints)
    output_df = pd.DataFrame()

    for index in range(0, num_train_steps + 1):

        if [x for x in finetuned_checkpoints_list if re.search(f'.*model\.ckpt-{index}\..*', x)]:

            checkpoint = f'./data/bert_finetuning_checkpoints/model.ckpt-{index}'

            result = _evaluateModel(checkpoint, label_list, number_of_validate_examples, num_train_steps, num_warmup_steps, bert_config, run_config, eval_file, max_seq_length, train_batch_size, eval_batch_size, learning_rate)
            result["step"] = index
            result["checkpoint"] = checkpoint
            output_df = output_df.append(result, ignore_index=True)
            logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logging.info("  %s = %s", key, str(result[key]))

    output_df.to_csv('./data/finetuning_evaluation_validation_data.csv', index=False)
            


def _evaluateModel(checkpoint, label_list, number_of_validate_examples, num_train_steps, num_warmup_steps, bert_config, run_config, eval_file, max_seq_length, train_batch_size, eval_batch_size, learning_rate):
    FLAGS = flags.FLAGS
    model_fn = model_fn_builder(
                bert_config=bert_config,
                num_labels=len(label_list),
                init_checkpoint=checkpoint,
                learning_rate=learning_rate,
                num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps,
                use_tpu=False,
                use_one_hot_embeddings=False)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    
    logging.info("***** Running evaluation *****")
    logging.info(f"  Num examples = {number_of_validate_examples}")
    logging.info("  Batch size = %d", eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None

    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=max_seq_length,
        is_training=False,
        drop_remainder=False)

    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

    return result



if __name__ == '__main__':
    evaluateFinetuning()