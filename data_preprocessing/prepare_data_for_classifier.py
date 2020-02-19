import random
import pandas as pd
import numpy as np


# Splits the data into a train and a test dataset and balances them to have equal positive and negative reports
# Since cross-validation is applied for the classifiers, a seperate validation dataset is not needed

def prepareDataForClassifier():
    train_size = 0.8

    print('Reading file...')
    df_report_features = pd.read_csv('./data/report_features_with_std.csv', sep='\t')

    report_features = df_report_features.to_dict('records')

    random.shuffle(report_features)

    array_length = len(report_features)

    train_interval_end = int(array_length*train_size)
    report_features_train = _balanceData(report_features[:train_interval_end], 'Change_Nominal')
    report_features_test = _balanceData(report_features[train_interval_end:], 'Change_Nominal')

    df_train = pd.DataFrame(data=report_features_train)
    df_test = pd.DataFrame(data=report_features_test)

    df_train.to_csv('./data/report_features_std_train.csv', sep='\t', index=False)
    df_test.to_csv('./data/report_features_std_test.csv', sep='\t', index=False)


def _balanceData(data, balance_key):
    print(f'Balancing data based on key {balance_key}...')
    data = list(data)
    classes = {}
    for entry in data:
        entry_class = entry[f'{balance_key}']

        if not entry_class in classes:
            classes[f'{entry_class}'] = []
            classes[f'{entry_class}'].append(entry)
        else:
            classes[f'{entry_class}'].append(entry)

    # get minimum
    class_occurences = list(map(lambda x: len(x), classes.values()))
    minimum_number_of_occurences = min(class_occurences)
    print('Occurences in classes:')
    print([(item[0], len(item[1])) for item in classes.items()])

    output_data = []
    for key in classes.keys():
        occurences_of_class = classes[f'{key}']
        random.shuffle(occurences_of_class)
        output_data.extend(occurences_of_class[:minimum_number_of_occurences])

    random.shuffle(output_data)

    print(f'Output data has {len(data) - len(output_data)} less items.')

    return output_data

if __name__ == "__main__":
    prepareDataForClassifier()

    
        