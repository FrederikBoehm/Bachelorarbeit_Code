import pandas as pd
import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, LSTM, Flatten
from tensorflow.keras.utils import Sequence
import os

def runLSTM():
    feature_vector_dimension = 768

    index_df = pd.read_csv('./data/multiline_report_features_index.csv', sep='\t')

    # Get x values
    X = _getFeatures(index_df['File_Path'][:100])
    X = _padToSameLength(X, feature_vector_dimension)

    split_index = int(0.75*len(list(X))) # This splitting is only possible because the dataset was randomized previously (prepare_data_for_classifier.py)
    X_train = np.array(X[:split_index])
    X_validate = np.array(X[split_index:])

    _, y = np.unique(index_df['Change_Nominal'].values[:100], return_inverse=True)
    y_train = np.array(y[:split_index])
    y_validate = np.array(y[split_index:])

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    print(f'SHAPE: {X_train.shape}')

    samples, time_steps, dimension = X_train.shape

    print(f'Initializing LSTM with {time_steps} time steps and neurons and a dimension of {dimension}')

    model = Sequential()

    forward_layer = LSTM(time_steps, return_sequences=True)
    backward_layer = LSTM(time_steps, input_shape=(time_steps, dimension), go_backwards=True, return_sequences=True)

    model.add(Bidirectional(forward_layer, backward_layer=backward_layer, input_shape=(time_steps, dimension)))
    model.add(Flatten())
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=2, batch_size=32)

    print('Accuracy Train:')
    print(model.evaluate(X_train, y_train))

    print('Accuracy Validate:')
    print(model.evaluate(X_validate, y_validate))



def _getFeatures(file_paths):

    features_all_files = []

    for file_path in file_paths:
        print(f'Loading {file_path}')
        report_features_df = pd.read_csv(file_path, sep='\t')
        report_features = list(report_features_df.values)
        features_all_files.append(report_features)

    return features_all_files

def _padToSameLength(feature_vectors, feature_vector_dimension):
    report_lengths = map(lambda feature_vector_single_report: len(feature_vector_single_report), feature_vectors)
    max_length = max(report_lengths)
    print(f'Padding to a maximum report length of {max_length}')

    for index, feature_vector_single_report in enumerate(feature_vectors):
        report_length = len(feature_vector_single_report)

        if report_length < max_length:
            zero_list = [[0 for col in range(feature_vector_dimension)] for row in range(max_length - report_length)]
            feature_vector_single_report.extend(zero_list)

    return feature_vectors

class ReportSequence(Sequence):

    def __init__(self, X, y):

        dataset = zip(X, y)
        self.X, self.y = self.__groupByTimesteps(dataset)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (np.array(self.X[idx]), np.array(self.y[idx]))

    def __groupByTimesteps(self, dataset):

        print('Grouping feature vectors by time steps...')

        grouped_reports = {}

        for report_X, report_y in dataset:
            time_steps = len(report_X)

            if time_steps > 0 and len(report_X[0]) > 0:
                if not time_steps in grouped_reports:
                    grouped_reports[time_steps] = {
                        'X': [],
                        'y': []
                    }

                grouped_reports[time_steps]['X'].append(report_X)
                grouped_reports[time_steps]['y'].append(report_y)

        batches_X = []
        batches_y = []

        for key in grouped_reports.keys():
            batches_X.append(grouped_reports[key]['X'])
            batches_y.append(grouped_reports[key]['y'])

        return (batches_X, batches_y)


if __name__ == "__main__":
    runLSTM()
