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
from multiprocessing import Process, current_process, cpu_count, Queue

def evaluatePretraining():
    FLAGS = flags.FLAGS

    input_file = "./data/bert_pretraining_data/seq_128/validate/tf_examples.tfrecord*"
    FLAGS.output_dir = "./data"
    # FLAGS.do_train = True
    # FLAGS.do_eval = True
    FLAGS.bert_config_file = './data/BERT/uncased_L-12_H-768_A-12/bert_config.json'
    # FLAGS.init_checkpoint = './data/BERT/uncased_L-12_H-768_A-12/bert_model.ckpt'
    FLAGS.train_batch_size = 32
    FLAGS.max_seq_length = 128
    FLAGS.max_predictions_per_seq = 20
    FLAGS.num_train_steps = 185000 # We take that number to iterate over the created checkpoints
    FLAGS.num_warmup_steps = 92500
    FLAGS.learning_rate = 2e-5
    FLAGS.eval_batch_size = 32
    FLAGS.save_checkpoints_steps = 18500
    FLAGS.iterations_per_loop = 1000
    FLAGS.max_eval_steps = 30560
    FLAGS.use_tpu = False
    FLAGS.tpu_name = None
    FLAGS.tpu_zone = None
    FLAGS.gcp_project = None
    FLAGS.master = None
    FLAGS.num_tpu_cores = None

    # index_file = './data/multiline_report_index_train.csv'

    FLAGS.mark_as_parsed()
    
    logging.set_verbosity(logging.INFO)

    # init_checkpoint = './data/BERT/uncased_L-12_H-768_A-12/bert_model.ckpt'

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # if not FLAGS.do_train and not FLAGS.do_eval:
    #     raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = BertConfig.from_json_file(FLAGS.bert_config_file)

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
        master=FLAGS.master,
        # model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    output_df = pd.DataFrame()

    checkpoint_path = './data/BERT/uncased_L-12_H-768_A-12/bert_model.ckpt'
    result = _evaluateModel(checkpoint_path, bert_config, run_config, input_files)
    result["epoch"] = 0
    result["checkpoint"] = checkpoint_path
    output_df = output_df.append(result, ignore_index=True)
    logging.info("***Eval results***")
    for key in sorted(result.keys()):
        logging.info("  %s = %s", key, str(result[key]))

    pretrained_checkpoints = './data/bert_pretraining_checkpoints/model.ckpt-*'
    pretrained_checkpoints_list = glob.glob(pretrained_checkpoints)

    for index in range(0, FLAGS.num_train_steps + 1):

        if [x for x in pretrained_checkpoints_list if re.search(f'.*model\.ckpt-{index}\..*', x)]:
            is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
            run_config = tf.compat.v1.estimator.tpu.RunConfig(
                cluster=tpu_cluster_resolver,
                master=FLAGS.master,
                model_dir=FLAGS.output_dir,
                save_checkpoints_steps=FLAGS.save_checkpoints_steps,
                tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
                    iterations_per_loop=FLAGS.iterations_per_loop,
                    num_shards=FLAGS.num_tpu_cores,
                    per_host_input_for_training=is_per_host))

            checkpoint = f'./data/bert_pretraining_checkpoints/model.ckpt-{index}'
            result = _evaluateModel(checkpoint, bert_config, run_config, input_files)
            result["epoch"] = int((index)/FLAGS.num_train_steps)
            result["checkpoint"] = checkpoint
            output_df = output_df.append(result, ignore_index=True)
            logging.info("***Eval results***")
            for key in sorted(result.keys()):
                logging.info("  %s = %s", key, str(result[key]))

    output_df.to_csv('./data/pretraining_evaluation_validation_data.csv', index=False)


def _evaluateModel(checkpoint, bert_config, run_config, input_files):

    FLAGS = flags.FLAGS
    model_fn = model_fn_builder(
            bert_config=bert_config,
            init_checkpoint=checkpoint,
            learning_rate=FLAGS.learning_rate,
            num_train_steps=FLAGS.num_train_steps,
            num_warmup_steps=FLAGS.num_warmup_steps,
            use_tpu=FLAGS.use_tpu,
            use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size)

    logging.info("***** Running evaluation *****")
    logging.info(f"Checkpoint: {checkpoint}")
    logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    eval_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=False)

    result = estimator.evaluate(
        input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)

    # output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    # with tf.io.gfile.GFile(output_eval_file, "w") as writer:
    #     logging.info("***** Eval results *****")
    # for key in sorted(result.keys()):
    #     logging.info("  %s = %s", key, str(result[key]))
    #     writer.write("%s = %s\n" % (key, str(result[key])))
    return result



if __name__ == "__main__":
    evaluatePretraining()