import os
from bert.run_pretraining import model_fn_builder, input_fn_builder, main
from bert.modeling import BertConfig

from absl import flags
from absl import logging
import tensorflow as tf
import time

def runPretraining():
    start_time = time.time()
    _initializeBert()

    input_file = "./data/bert_pretraining_data/seq_128/train/tf_examples.tfrecord*"
    output_dir = "./data/bert_pretraining_checkpoints"
    do_train = True
    do_eval = True
    bert_config_file = './data/BERT/uncased_L-12_H-768_A-12/bert_config.json'
    init_checkpoint = './data/BERT/uncased_L-12_H-768_A-12/bert_model.ckpt'
    train_batch_size = 32
    max_seq_length = 128
    max_predictions_per_seq = 20
    num_train_steps = 185000
    num_warmup_steps = 18500
    learning_rate = 2e-5
    eval_batch_size = 32
    save_checkpoints_steps = 18500
    max_eval_steps = 1000
    iterations_per_loop = 1000

    
    logging.set_verbosity(logging.INFO)

    

    bert_config = BertConfig.from_json_file(bert_config_file)

    tf.io.gfile.makedirs(output_dir)

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
        keep_checkpoint_max=30,
        tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
            iterations_per_loop=iterations_per_loop,
            num_shards=None,
            per_host_input_for_training=is_per_host))

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=init_checkpoint,
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

    if do_train:
        logging.info("***** Running training *****")
        logging.info("  Batch size = %d", train_batch_size)
        train_input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=max_seq_length,
            max_predictions_per_seq=max_predictions_per_seq,
            is_training=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if do_eval:
        logging.info("***** Running evaluation *****")
        logging.info("  Batch size = %d", eval_batch_size)

        eval_input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=max_seq_length,
            max_predictions_per_seq=max_predictions_per_seq,
            is_training=False)

        result = estimator.evaluate(
            input_fn=eval_input_fn, steps=max_eval_steps)

        output_eval_file = os.path.join(output_dir, "eval_results.txt")
        with tf.io.gfile.GFile(output_eval_file, "w") as writer:
            logging.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    end_time = time.time()
    elapsed_time = end_time - start_time
    f = open('./data/pretraining_time.txt', 'a+')
    f.write(f"{elapsed_time}\n")
    f.close()

def _initializeBert():
    FLAGS = flags.FLAGS
    FLAGS.mark_as_parsed()

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'


if __name__ == "__main__":
    runPretraining()