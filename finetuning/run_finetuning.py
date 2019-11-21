import tensorflow as tf
from absl import flags
from absl import logging
from bert.tokenization import validate_case_matches_checkpoint, FullTokenizer, convert_to_unicode
from bert.modeling import BertConfig
from bert.run_classifier import DataProcessor, model_fn_builder, file_based_convert_examples_to_features, file_based_input_fn_builder, PaddingInputExample, InputExample
from report_processor import ReportProcessor
import os
import time

def runFinetuning():
    start_time = time.time()

    FLAGS = flags.FLAGS
    # FLAGS.data_dir = './data'
    FLAGS.bert_config_file = './data/BERT/uncased_L-12_H-768_A-12/bert_config.json'
    # FLAGS.task_name = 'mrpc'
    FLAGS.vocab_file = './data/BERT/uncased_L-12_H-768_A-12/vocab.txt'
    FLAGS.output_dir = './data/bert_finetuning_checkpoints'
    FLAGS.init_checkpoint = './data/bert_pretraining_512/model.ckpt-18500'
    FLAGS.do_lower_case = True
    FLAGS.max_seq_length = 512
    FLAGS.do_train = True
    FLAGS.do_eval = True
    FLAGS.do_predict = False
    FLAGS.train_batch_size = 6
    FLAGS.eval_batch_size = 6
    # FLAGS.predict_batch_size = 
    FLAGS.learning_rate = 2e-5
    FLAGS.num_train_epochs = 4
    FLAGS.warmup_proportion = 0.1
    FLAGS.save_checkpoints_steps = 22406
    FLAGS.iterations_per_loop = 1000
    FLAGS.use_tpu = False
    FLAGS.tpu_name = False
    FLAGS.mark_as_parsed()

    # number_of_train_examples = 11298966
    number_of_train_examples = 672176
    # number_of_validate_examples = 3828636
    number_of_validate_examples = 228060
    # number_of_test_examples = 0
    train_file = "./data/bert_finetuning_data/train/train.tf_record"
    eval_file = "./data/bert_finetuning_data/validate/validate.tf_record"

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    logging.set_verbosity(logging.INFO)

    validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                    FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    label_list = ReportProcessor().get_labels()

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=30,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        num_train_steps = int(
            number_of_train_examples / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", number_of_train_examples)
        logging.info("  Batch size = %d", FLAGS.train_batch_size)
        logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:

        logging.info("***** Running evaluation *****")
        logging.info(f"  Num examples = {number_of_validate_examples}")
        logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    end_time = time.time()
    elapsed_time = end_time - start_time
    f = open('./data/finetuning_time.txt', 'a+')
    f.write(f"{elapsed_time}\n")
    f.close()

if __name__ == "__main__":
    runFinetuning()