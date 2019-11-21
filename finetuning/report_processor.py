import os
from bert.run_classifier import DataProcessor, InputExample
from bert.tokenization import convert_to_unicode

class ReportProcessor(DataProcessor):

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "fine_tuning_data_train.csv")), "train")

  def get_validate_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "fine_tuning_data_validate.csv")), "validate")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "fine_tuning_data_test.csv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        guid = "%s-%s" % (set_type, i)
        
        text_a = convert_to_unicode(line[1])
        label = convert_to_unicode(line[0])
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples