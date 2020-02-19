import os
from bert.run_pretraining import model_fn_builder, input_fn_builder
from bert.modeling import BertConfig

from absl import flags
from absl import logging
import tensorflow as tf
import glob
import re
import pandas as pd
import numpy as np

# Evaluates the trained model either for parameter optimization with the validation dataset or for final accuracy with test dataset

def evaluatePretraining():
    FLAGS = flags.FLAGS
    FLAGS.mark_as_parsed()

    input_file = "./data/bert_pretraining_data/seq_128/validate/tf_examples.tfrecord*"
    output_dir = "./data"
    bert_config_file = './data/BERT/uncased_L-12_H-768_A-12/bert_config.json'
    train_batch_size = 32
    max_seq_length = 128
    max_predictions_per_seq = 20
    num_train_steps = 185000 # We take that number to iterate over the created checkpoints
    num_warmup_steps = 18500
    learning_rate = 2e-5
    eval_batch_size = 32
    save_checkpoints_steps = 18500
    iterations_per_loop = 1000
    max_eval_steps = 30560


    
    logging.set_verbosity(logging.INFO)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    bert_config = BertConfig.from_json_file(bert_config_file)

    input_files = []
    for input_pattern in input_file.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))

    logging.info("*** Input Files ***")
    for input_file in input_files:
        logging.info("  %s" % input_file)

    tpu_cluster_resolver = None

    is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.compat.v1.estimator.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=None,
        model_dir=output_dir,
        save_checkpoints_steps=save_checkpoints_steps,
        tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
            iterations_per_loop=iterations_per_loop,
            num_shards=None,
            per_host_input_for_training=is_per_host))

    output_df = pd.DataFrame()

    checkpoints_path = './data/bert_pretraining_checkpoints_2'
    pretrained_checkpoints = checkpoints_path + '/model.ckpt-*'
    pretrained_checkpoints_list = glob.glob(pretrained_checkpoints)

    for index in range(0, num_train_steps + 1):

        if [x for x in pretrained_checkpoints_list if re.search(f'.*model\.ckpt-{index}\..*', x)]:
            is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
            run_config = tf.compat.v1.estimator.tpu.RunConfig(
                cluster=tpu_cluster_resolver,
                master=None,
                model_dir=output_dir,
                save_checkpoints_steps=save_checkpoints_steps,
                tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
                    iterations_per_loop=iterations_per_loop,
                    num_shards=None,
                    per_host_input_for_training=is_per_host))

            checkpoint = f'{checkpoints_path}/model.ckpt-{index}'
            result = _evaluateModel(checkpoint, bert_config, run_config, input_files, max_seq_length, num_train_steps, num_warmup_steps, learning_rate, train_batch_size, eval_batch_size, max_eval_steps, max_predictions_per_seq)
            result["checkpoint"] = checkpoint
            result["step"] = index
            output_df = output_df.append(result, ignore_index=True)
            logging.info("***Eval results***")
            for key in sorted(result.keys()):
                logging.info("  %s = %s", key, str(result[key]))

    output_df.to_csv('./data/pretraining_evaluation_validation_data_modified_vocab_short_warmup.csv', index=False)


def _evaluateModel(checkpoint, bert_config, run_config, input_files, max_seq_length, num_train_steps, num_warmup_steps, learning_rate, train_batch_size, eval_batch_size, max_eval_steps, max_predictions_per_seq):

    model_fn = model_fn_builder(
            bert_config=bert_config,
            init_checkpoint=checkpoint,
            learning_rate=learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=False,
            use_one_hot_embeddings=False)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size)

    logging.info("***** Running evaluation *****")
    logging.info(f"Checkpoint: {checkpoint}")
    logging.info("  Batch size = %d", eval_batch_size)

    eval_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=max_seq_length,
        max_predictions_per_seq=max_predictions_per_seq,
        is_training=False)

    result = estimator.evaluate(
        input_fn=eval_input_fn, steps=max_eval_steps)

    return result



if __name__ == "__main__":
    evaluatePretraining()