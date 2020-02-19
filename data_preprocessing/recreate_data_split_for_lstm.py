import pandas as pd
import random
from prepare_data_for_classifier import _balanceData
import re


# To gain comparable results, this function recreates the train test datasplit for the LSTM

def recreateDataSplitForLSTM():
    print("Reading data files...")
    std_deviation_index = pd.read_csv("./data/multiline_report_features_index.csv", sep="\t").to_dict('records')
    train_files = pd.read_csv("./data/report_features_std_train.csv", sep="\t")["File_Path"]
    test_files = pd.read_csv("./data/report_features_std_test.csv", sep="\t")["File_Path"]

    file_name_regex = r'multiline_report\d+'
    train_files = list(map(lambda train_file: re.findall(file_name_regex, train_file)[0], train_files))
    test_files = list(map(lambda test_file: re.findall(file_name_regex, test_file)[0], test_files))

    recreated_train_files = []
    recreated_test_files = []
    print("Recreating data split...")
    for index_entry in std_deviation_index:

        file_name = re.findall(file_name_regex, index_entry["File_Path"])[0]

        if file_name in train_files:
            recreated_train_files.append(index_entry)
        elif file_name in test_files:
            recreated_test_files.append(index_entry)
        else: # Because train and test files are balanced, it is possible that the original index file contains entries that are neither in train nor in test
            print(f'File {file_name} was neither in train nor in test dataset. Skipping...')

    df_train_output = pd.DataFrame(recreated_train_files)
    df_test_output = pd.DataFrame(recreated_test_files)

    print("Writing to output files...")
    df_train_output.to_csv("./data/multiline_report_features_train.csv", index=False, sep="\t")
    df_test_output.to_csv("./data/multiline_report_features_test.csv", index=False, sep="\t")

if __name__ == "__main__":
    recreateDataSplitForLSTM()


