import pandas as pd
import numpy as np


# Splits the data for BERT training, since no cross-validation is applied a seperate validation dataset is created

def splitData():
    train_size = 0.6
    validate_size = 0.2

    df_multiline_report_index = pd.read_csv('./data/multiline_report_index_with_price_changes.csv', sep='\t')

    multiline_report_index_array = df_multiline_report_index.values
    column_names = list(df_multiline_report_index.columns.values)

    np.random.shuffle(multiline_report_index_array)

    array_length, _ = multiline_report_index_array.shape

    train_interval_end = int(array_length*train_size)
    multiline_report_index_train = multiline_report_index_array[:train_interval_end]
    validate_interval_end = train_interval_end + int(array_length*validate_size)
    multiline_report_index_validate = multiline_report_index_array[train_interval_end:validate_interval_end]
    multiline_report_index_test = multiline_report_index_array[validate_interval_end:]

    df_train = pd.DataFrame(data=multiline_report_index_train, columns=column_names)
    df_validate = pd.DataFrame(data=multiline_report_index_validate, columns=column_names)
    df_test = pd.DataFrame(data=multiline_report_index_test, columns=column_names)

    df_train.to_csv('./data/multiline_report_index_train.csv', sep='\t', index=False)
    df_validate.to_csv('./data/multiline_report_index_validate.csv', sep='\t', index=False)
    df_test.to_csv('./data/multiline_report_index_test.csv', sep='\t', index=False)

if __name__ == '__main__':
    splitData()