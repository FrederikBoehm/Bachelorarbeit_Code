import pandas as pd
import numpy as np
import argparse
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM, Bidirectional, GlobalAveragePooling1D, TimeDistributed, Concatenate
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import os
from train_lstm import _getFeatures, _padToSameLength, ReportSequence, _scaleFeatures, AccuracyMeasurement

def createLearningCurveNB():

    df_train = pd.read_csv('./data/report_features_std_train.csv', sep='\t')

    X_train = df_train.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values[:, :feature_vector_dimension]
    _, y_train = np.unique(df_train['Change_Nominal'].values, return_inverse=True)

    # train_sizes = np.linspace(1, len(y_train), 10).astype(int).tolist()
    train_sizes = np.linspace(0.01, 1.0, 10)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('nb', GaussianNB())
    ])

    train_sizes_abs, train_scores, valid_scores = learning_curve(pipeline, X_train, y_train, train_sizes=train_sizes, cv=5, n_jobs=20, verbose=5)

    results = {
        'train_sizes': train_sizes_abs,
        'train_scores_mean': list(np.mean(train_scores, axis=1)),
        'train_scores_std': list(np.std(train_scores, axis=1)),
        'valid_scores_mean': list(np.mean(valid_scores, axis=1)),
        'valid_scores_std': list(np.std(valid_scores, axis=1))
    }

    output_df = pd.DataFrame(results)

    print(output_df)

    output_df.to_csv(f'./data/nb_learning_curve_{feature_vector_dimension}.csv', index=False)

def createLearningCurveKNN(feature_vector_dimension):

    df_train = pd.read_csv('./data/report_features_std_train.csv', sep='\t')

    X_train = df_train.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values[:, :feature_vector_dimension]
    _, y_train = np.unique(df_train['Change_Nominal'].values, return_inverse=True)

    # train_sizes = np.linspace(1, len(y_train), 10).astype(int).tolist()
    train_sizes = np.linspace(0.1, 1.0, 10)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=233))
    ])

    train_sizes_abs, train_scores, valid_scores = learning_curve(pipeline, X_train, y_train, train_sizes=train_sizes, cv=5, n_jobs=20, verbose=5)

    results = {
        'train_sizes': train_sizes_abs,
        'train_scores_mean': list(np.mean(train_scores, axis=1)),
        'train_scores_std': list(np.std(train_scores, axis=1)),
        'valid_scores_mean': list(np.mean(valid_scores, axis=1)),
        'valid_scores_std': list(np.std(valid_scores, axis=1))
    }

    output_df = pd.DataFrame(results)

    print(output_df)

    output_df.to_csv(f'./data/knn_learning_curve_{feature_vector_dimension}.csv', index=False)

def createLearningCurveSVM(feature_vector_dimension):

    df_train = pd.read_csv('./data/report_features_std_train.csv', sep='\t')

    X_train = df_train.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values[:, :feature_vector_dimension]
    _, y_train = np.unique(df_train['Change_Nominal'].values, return_inverse=True)

    # train_sizes = np.linspace(1, len(y_train), 10).astype(int).tolist()
    train_sizes = np.linspace(0.1, 1.0, 10)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=100, gamma=0.0001)),
        # ('svm', SVC(kernel='rbf', C=100000, gamma=1e-07))
    ])

    train_sizes_abs, train_scores, valid_scores = learning_curve(pipeline, X_train, y_train, train_sizes=train_sizes, cv=5, n_jobs=20, verbose=5)

    results = {
        'train_sizes': train_sizes_abs,
        'train_scores_mean': list(np.mean(train_scores, axis=1)),
        'train_scores_std': list(np.std(train_scores, axis=1)),
        'valid_scores_mean': list(np.mean(valid_scores, axis=1)),
        'valid_scores_std': list(np.std(valid_scores, axis=1))
    }

    output_df = pd.DataFrame(results)

    print(output_df)

    output_df.to_csv(f'./data/svm_learning_curve_{feature_vector_dimension}.csv', index=False)

# def createLearningCurve():
    
#     os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#     df_train = pd.read_csv('./data/report_features_train.csv', sep='\t')

#     X = df_train.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values
#     _, y = np.unique(df_train['Change_Nominal'].values, return_inverse=True)

#     kf = KFold(n_splits=5)

#     accuracy_results_cv = {
#         'train': [],
#         'validate': []
#     }


#     for train_index, valid_index in kf.split(X):
#         X_train = X[train_index]
#         X_valid = X[valid_index]

#         y_train = y[train_index]
#         y_valid = y[valid_index]

#         scaler = StandardScaler()
#         scaler.fit(X_train)
#         X_train = scaler.transform(X_train)
#         X_valid = scaler.transform(X_valid)

#         model = Sequential()

#         model.add(Dense(768, activation='sigmoid', input_shape=(768,)))
#         model.add(Dropout(0.25))
#         model.add(Dense(100, activation='sigmoid'))
#         model.add(Dropout(0.25))
#         model.add(Dense(10, activation='sigmoid'))
#         model.add(Dense(1, activation='sigmoid'))

#         accuracy_measurement = AccuracyMeasurement(model, X_train, y_train, X_valid, y_valid, accuracy_results_cv)

#         model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#         model.fit(X_train, y_train, epochs=40, batch_size=1000, callbacks=[accuracy_measurement])

#     train_scores = np.array(accuracy_results_cv['train'])
#     valid_scores = np.array(accuracy_results_cv['validate'])

#     print(accuracy_results_cv['train'])

#     results = {
#         'epochs': range(len(accuracy_results_cv['train'])),
#         'train_scores_mean': list(np.mean(train_scores, axis=1)),
#         'train_scores_std': list(np.std(train_scores, axis=1)),
#         'valid_scores_mean': list(np.mean(valid_scores, axis=1)),
#         'valid_scores_std': list(np.std(valid_scores, axis=1))
#     }

#     output_df = pd.DataFrame(results)

#     output_df.to_csv('./data/deep_learning_curve.csv', index=False)

def createLearningCurveBLSTM(feature_vector_dimension):
    
    index_df = pd.read_csv('./data/multiline_report_features_train.csv', sep='\t')

    # Get x values
    X = _getFeatures(index_df['File_Path'])
    # X = _padToSameLength(X, feature_vector_dimension)
    X = np.array(X)

    _, y = np.unique(index_df['Change_Nominal'].values, return_inverse=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = '5'


    kf = KFold(n_splits=5)

    accuracy_results_cv = {
        'train': [],
        'validate': []
    }

    for train_index, valid_index in kf.split(X):
        X_train = X[train_index]
        X_valid = X[valid_index]

        X_train = _scaleFeatures(X_train)
        X_valid = _scaleFeatures(X_valid, X_train)

        y_train = y[train_index]
        y_valid = y[valid_index]

        train_sequence = ReportSequence(X_train, y_train)
        validation_sequence = ReportSequence(X_valid, y_valid)

        # samples, time_steps, dimension = X_train.shape

        print(f'Initializing LSTM ...')

        model = Sequential()

        forward_layer = LSTM(300, return_sequences=True)
        backward_layer = LSTM(300, go_backwards=True, return_sequences=True)

        model.add(Bidirectional(forward_layer, backward_layer=backward_layer, input_shape=(None, feature_vector_dimension)))
        # model.add(GlobalAveragePooling1D())
        # model.add(Dense(1232, activation='sigmoid'))
        # model.add(Dense(120, activation='sigmoid'))
        # model.add(Dense(12, activation='sigmoid'))

        # input2 = Sequential()
        # input2.add(TimeDistributed(Dense(500, activation="sigmoid", input_shape=(None, feature_vector_dimension))))

        # model.add(Concatenate([input1, input2]))
        # model.add(LSTM(500, return_sequences=False, input_shape=(None, feature_vector_dimension)))
        # model.add(LSTM(500, input_shape=(None, feature_vector_dimension), return_sequences=True))
        # model.add(Dropout(0.2))
        # model.add(TimeDistributed(Dense(1000, activation='sigmoid')))
        # model.add(TimeDistributed(Dense(100, activation='sigmoid')))
        # model.add(TimeDistributed(Dense(10, activation='sigmoid')))
        # model.add(GlobalAveragePooling1D())
        # model.add(Dense(1000, activation='sigmoid'))
        # model.add(Dense(100, activation='sigmoid'))
        # model.add(Dense(10, activation='sigmoid'))
        model.add(Bidirectional(LSTM(30, return_sequences=True), backward_layer=LSTM(30, go_backwards=True, return_sequences=True)))
        model.add(Bidirectional(LSTM(3, return_sequences=True), backward_layer=LSTM(3, go_backwards=True, return_sequences=True)))
        model.add(GlobalAveragePooling1D())
        # model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        optimizer = Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        summary_file = open(f'./data/lstm_summary_{feature_vector_dimension}.txt', 'w+')
        model.summary(print_fn=lambda x: summary_file.write(x + '\n'))
        summary_file.close()


        accuracy_measurement = AccuracyMeasurement(model, train_sequence, validation_sequence, accuracy_results_cv)

        model.fit_generator(train_sequence, epochs=20, callbacks=[accuracy_measurement], validation_data=validation_sequence)

    train_scores = np.array(accuracy_results_cv['train'])
    valid_scores = np.array(accuracy_results_cv['validate'])

    results = {
        'epochs': range(len(accuracy_results_cv['train'])),
        'train_scores_mean': list(np.mean(train_scores, axis=1)),
        'train_scores_std': list(np.std(train_scores, axis=1)),
        'valid_scores_mean': list(np.mean(valid_scores, axis=1)),
        'valid_scores_std': list(np.std(valid_scores, axis=1))
    }

    output_df = pd.DataFrame(results)

    print(output_df)

    output_df.to_csv(f'./data/lstm_learning_curve_{feature_vector_dimension}.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process which model should be selected for the learning curve.')
    parser.add_argument('--model', dest='model', type=str)
    parser.add_argument('--mode', dest='mode', type=str)
    args = parser.parse_args()
    feature_dimension = None
    if args.mode == 'AVG_ONLY':
        feature_dimension = 768
    elif args.mode == 'USE_STD':
        feature_dimension = 1536
    else:
        print('Parameter --mode not set, use AVG_ONLY or USE_STD')

    if args.model == 'BLSTM':
        createLearningCurveBLSTM(feature_dimension)
    elif args.model == 'KNN':
        createLearningCurveKNN(feature_dimension)
    elif args.model == 'NB':
        createLearningCurveNB(feature_dimension)
    elif args.model == 'SVM':
        createLearningCurveSVM(feature_dimension)
    else:
        print(f'No model found with the key {args.model}')