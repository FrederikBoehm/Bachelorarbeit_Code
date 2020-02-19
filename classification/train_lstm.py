import pandas as pd
import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, LSTM, Flatten, GlobalAveragePooling1D
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import os


# Trains the LSTM and evaluates it on the test dataset

def runLSTM():
    feature_vector_dimension = 768
    checkpoints_path = "./data/lstm_checkpoints"

    train_df = pd.read_csv('./data/multiline_report_features_train.csv', sep='\t')
    test_df = pd.read_csv('./data/multiline_report_features_test.csv', sep='\t')

    # Get x values
    X_train = _getFeatures(train_df['File_Path'])
    X_train = np.array(X_train)
    X_train = _scaleFeatures(X_train)

    X_test = _getFeatures(test_df['File_Path'])
    X_test = np.array(X_test)
    X_test = _scaleFeatures(X_test, X_train)

    _, y_train = np.unique(train_df['Change_Nominal'].values, return_inverse=True)
    _, y_test = np.unique(test_df['Change_Nominal'].values, return_inverse=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    train_sequence = ReportSequence(X_train, y_train)
    test_sequence = ReportSequence(X_test, y_test)
    model = Sequential()

    forward_layer = LSTM(616, return_sequences=True)
    backward_layer = LSTM(616, go_backwards=True, return_sequences=True)

    model.add(Bidirectional(forward_layer, backward_layer=backward_layer, input_shape=(None, feature_vector_dimension)))
    model.add(Bidirectional(LSTM(62, return_sequences=True), backward_layer=LSTM(62, go_backwards=True, return_sequences=True)))
    model.add(Bidirectional(LSTM(6, return_sequences=True), backward_layer=LSTM(6, go_backwards=True, return_sequences=True)))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    ModelCheckpoint(checkpoints_path + "/checkpoint-{epoch}.hdf5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    model.fit_generator(train_sequence, epochs=20)

    accuracy_result = model.evaluate_generator(test_sequence)
    print('Accuracy Test:')
    print(accuracy_result)

    output_file = open("./data/lstm_accuracy.txt", "w+")
    output_file.write(str(accuracy_result))
    output_file.close()



def _getFeatures(file_paths):

    features_all_files = []

    for file_path in file_paths:
        print(f'Loading {file_path}')
        report_features_df = pd.read_csv(file_path, sep='\t')
        report_features = list(report_features_df.values)
        features_all_files.append(report_features)

    return features_all_files


def _scaleFeatures(transform_values, fit_values=None):
    if fit_values is None:
        fit_values = transform_values

    # We first flatten the 3d array to 2d array, because StandardScaler can only operate on 2d array
    flattened_list_fit = []
    split_indexes_fit = []
    for fit_value in fit_values:
        flattened_list_fit.extend(fit_value)
        if len(split_indexes_fit) == 0:
            split_indexes_fit.append(len(fit_value))
        else:
            split_indexes_fit.append(split_indexes_fit[len(split_indexes_fit)-1] + len(fit_value))

    flattened_list_transform = []
    split_indexes_transform = []
    for transform_value in transform_values:
        flattened_list_transform.extend(transform_value)
        if len(split_indexes_transform) == 0:
            split_indexes_transform.append(len(transform_value))
        else:
            split_indexes_transform.append(split_indexes_transform[len(split_indexes_transform)-1] + len(transform_value))
    print(f"Whole length: {len(flattened_list_transform)}")
    print(f"Single element: {len(flattened_list_transform[0])}")
    # Scaling
    scaler = StandardScaler()
    scaler.fit(flattened_list_fit)

    flattened_list_transform = scaler.transform(flattened_list_transform)

    # Back to 3d
    scaled_list = []
    for index, value in enumerate(split_indexes_transform):
        if index == 0:
            scaled_list.append(flattened_list_transform[:value])
        else:
            scaled_list.append(flattened_list_transform[split_indexes_transform[index - 1]:value])

    return scaled_list

class AccuracyMeasurement(Callback):

    def __init__(self, model, train_sequence, validation_sequence, accuracy_results):
        self.model = model
        self.train_sequence = train_sequence
        self.validation_sequence = validation_sequence
        self.accuracy_results = accuracy_results

    def on_epoch_end(self, epoch, logs):
        _, train_accuracy = self.model.evaluate_generator(self.train_sequence)
        _, validation_accuracy = self.model.evaluate_generator(self.validation_sequence)

        train_results = self.accuracy_results['train']
        if len(train_results) == epoch:
            train_results.append([])

        train_results[epoch].append(train_accuracy)

        validate_results = self.accuracy_results['validate']
        if len(validate_results) == epoch:
            validate_results.append([])

        validate_results[epoch].append(validation_accuracy)


# This class groups the the reports by the number of sequences
# This is necessary because the LSTM requires the input to have the same number of time steps on batch level
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
