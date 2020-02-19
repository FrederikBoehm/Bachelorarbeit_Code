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
from train_lstm import _getFeatures, ReportSequence, _scaleFeatures, AccuracyMeasurement


# Allows to create learning curves for different classifiers

def createLearningCurveNB(feature_vector_dimension):

    df_train = pd.read_csv('./data/report_features_std_train.csv', sep='\t')

    X_train = df_train.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values[:, :feature_vector_dimension]
    _, y_train = np.unique(df_train['Change_Nominal'].values, return_inverse=True)

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

def createLearningCurveKNN(feature_vector_dimension, K):

    df_train = pd.read_csv('./data/report_features_std_train.csv', sep='\t')

    X_train = df_train.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values[:, :feature_vector_dimension]
    _, y_train = np.unique(df_train['Change_Nominal'].values, return_inverse=True)

    train_sizes = np.linspace(0.1, 1.0, 10)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=K))
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

def createLearningCurveSVM(feature_vector_dimension, C, gamma):

    df_train = pd.read_csv('./data/report_features_std_train.csv', sep='\t')

    X_train = df_train.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values[:, :feature_vector_dimension]
    _, y_train = np.unique(df_train['Change_Nominal'].values, return_inverse=True)

    train_sizes = np.linspace(0.1, 1.0, 10)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=C, gamma=gamma)),
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


def createLearningCurveBLSTM(feature_vector_dimension):
    
    index_df = pd.read_csv('./data/multiline_report_features_train.csv', sep='\t')

    X = _getFeatures(index_df['File_Path'])
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


        print(f'Initializing LSTM ...')

        model = Sequential()

        forward_layer = LSTM(300, return_sequences=True)
        backward_layer = LSTM(300, go_backwards=True, return_sequences=True)

        model.add(Bidirectional(forward_layer, backward_layer=backward_layer, input_shape=(None, feature_vector_dimension)))
        model.add(Bidirectional(LSTM(30, return_sequences=True), backward_layer=LSTM(30, go_backwards=True, return_sequences=True)))
        model.add(Bidirectional(LSTM(3, return_sequences=True), backward_layer=LSTM(3, go_backwards=True, return_sequences=True)))
        model.add(GlobalAveragePooling1D())
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
    parser.add_argument('--K', dest='K', type=int)
    parser.add_argument('--C', dest='C', type=float)
    parser.add_argument('--gamma', dest='gamma', type=float)
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
        createLearningCurveKNN(feature_dimension, args.K)
    elif args.model == 'NB':
        createLearningCurveNB(feature_dimension)
    elif args.model == 'SVM':
        createLearningCurveSVM(feature_dimension, args.C, args.gamma)
    else:
        print(f'No model found with the key {args.model}')