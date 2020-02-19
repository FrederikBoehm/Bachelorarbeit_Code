import tensorflow as tf
from absl import flags
from absl import logging
from bert.tokenization import validate_case_matches_checkpoint, FullTokenizer, convert_to_unicode
from bert.modeling import BertConfig
from bert.run_classifier import DataProcessor, model_fn_builder, file_based_convert_examples_to_features, file_based_input_fn_builder, PaddingInputExample, InputExample
from report_processor import ReportProcessor
import os
import time


# Runs Fine-tuning and saves a checkpoint every 22406 steps, which gives 20 checkpoints
# The checkpoints are later used to evaluate the model

def runFinetuning():
    start_time = time.time()

    FLAGS = flags.FLAGS
    FLAGS.mark_as_parsed()

    bert_config_file = './data/BERT/uncased_L-12_H-768_A-12/bert_config.json'
    vocab_file = './data/BERT/uncased_L-12_H-768_A-12/vocab.txt'
    output_dir = './data/bert_finetuning_checkpoints'
    init_checkpoint = './data/bert_pretraining_512/model.ckpt-18500'
    do_lower_case = True
    max_seq_length = 512
    do_train = True
    do_eval = True
    train_batch_size = 6
    eval_batch_size = 6
    learning_rate = 2e-5
    num_train_epochs = 4
    warmup_proportion = 0.1
    save_checkpoints_steps = 22406
    iterations_per_loop = 1000
    
    number_of_train_examples = 672176
    number_of_validate_examples = 228060
    train_file = "./data/bert_finetuning_data/train/train.tf_record"
    eval_file = "./data/bert_finetuning_data/validate/validate.tf_record"

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    logging.set_verbosity(logging.INFO)

    validate_case_matches_checkpoint(do_lower_case, init_checkpoint)

    if not do_train and not do_eval:
        raise ValueError(
            "At least one of `do_train`, `do_eval` must be True.")

    bert_config = BertConfig.from_json_file(bert_config_file)

    if max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(output_dir)

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
            num_shards=8,
            per_host_input_for_training=is_per_host))

    num_train_steps = None
    num_warmup_steps = None
    if do_train:
        num_train_steps = int(
            number_of_train_examples /train_batch_size * num_train_epochs)
        num_warmup_steps = int(num_train_steps * warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=init_checkpoint,
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
        predict_batch_size=None)

    if do_train:
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", number_of_train_examples)
        logging.info("  Batch size = %d", train_batch_size)
        logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if do_eval:

        logging.info("***** Running evaluation *****")
        logging.info(f"  Num examples = {number_of_validate_examples}")
        logging.info("  Batch size = %d", eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None

        eval_drop_remainder = False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(output_dir, "eval_results.txt")
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