import os
from bert.run_pretraining import model_fn_builder, input_fn_builder
from bert.modeling import BertConfig

from absl import flags
from absl import logging
import tensorflow as tf

def runPretraining():
    FLAGS = flags.FLAGS

    FLAGS.input_file = "./data/bert_pretraining_data/seq_128/train/tf_examples.tfrecord*"
    FLAGS.output_dir = "./data/bert_pretraining_checkpoints"
    FLAGS.do_train = True
    FLAGS.do_eval = True
    FLAGS.bert_config_file = './data/BERT/uncased_L-12_H-768_A-12/bert_config.json'
    FLAGS.init_checkpoint = './data/BERT/uncased_L-12_H-768_A-12/bert_model.ckpt'
    FLAGS.train_batch_size = 32
    FLAGS.max_seq_length = 128
    FLAGS.max_predictions_per_seq = 20
    FLAGS.num_train_steps = 459
    FLAGS.num_warmup_steps = 100
    FLAGS.learning_rate = 2e-5
    FLAGS.eval_batch_size = 32
    FLAGS.save_checkpoints_steps = 100
    FLAGS.iterations_per_loop = 1000
    FLAGS.max_eval_steps = 459
    FLAGS.use_tpu = False
    FLAGS.tpu_name = None
    FLAGS.tpu_zone = None
    FLAGS.gcp_project = None
    FLAGS.master = None
    FLAGS.num_tpu_cores = None

    FLAGS.mark_as_parsed()
    
    logging.set_verbosity(logging.INFO)

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = BertConfig.from_json_file(FLAGS.bert_config_file)

    tf.io.gfile.makedirs(FLAGS.output_dir)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))

    logging.info("*** Input Files ***")
    for input_file in input_files:
        logging.info("  %s" % input_file)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

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

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
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

    if FLAGS.do_train:
        logging.info("***** Running training *****")
        logging.info("  Batch size = %d", FLAGS.train_batch_size)
        train_input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=True)
        estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

    if FLAGS.do_eval:
        logging.info("***** Running evaluation *****")
        logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        eval_input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=False)

        result = estimator.evaluate(
            input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.io.gfile.GFile(output_eval_file, "w") as writer:
            logging.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
    runPretraining()